"""v3 run-detector calibration, whole-run flagging, profile repair, and weather."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_onboarding.checks import (
    QualityConfig,
    causal_anomaly_masks,
    detector_calibration_table,
    hour_of_week_profile,
)
from energy_onboarding.faults import inject_downstream_faults
from energy_onboarding.pipeline import run_pipeline, verify_result_dir
from energy_onboarding.remediation import remediate


def _building(building_id: str, loads: np.ndarray, start: str = "2016-01-04") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(loads), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "building_id": building_id,
            "load": loads.astype(float),
            "site_id": "s1",
            "primary_use": "Office",
            "timezone": "US/Eastern",
        }
    )


def test_whole_run_flagging_marks_every_member_including_the_head() -> None:
    loads = np.arange(1, 41, dtype=float)
    loads[12:20] = 7.5  # eight identical non-zero readings
    frame = _building("fine", loads)
    tail = causal_anomaly_masks(frame, QualityConfig(stuck_threshold=8, causal_window=12))
    whole = causal_anomaly_masks(
        frame, QualityConfig(stuck_threshold=8, causal_window=12, run_flagging="whole_run")
    )
    assert list(frame.index[tail["stuck"]]) == [19]
    assert list(frame.index[whole["stuck"]]) == list(range(12, 20))


def test_coarse_meter_is_exempted_by_calibration_and_fine_meter_is_not() -> None:
    rng = np.random.default_rng(7)
    weeks = 6
    hours = 24 * 7 * weeks
    # A coarsely quantized meter: readings only change every ~12 hours.
    coarse = np.repeat(rng.integers(40, 60, size=hours // 12), 12).astype(float)
    fine = 50 + rng.normal(0, 3, size=hours)
    frame = pd.concat(
        [_building("coarse", coarse), _building("fine", fine)], ignore_index=True
    )
    # Inject an 8-hour stuck segment into both buildings inside the last week.
    for building_id in ("coarse", "fine"):
        rows = frame.index[frame["building_id"] == building_id][-100:-92]
        frame.loc[rows, "load"] = 55.123
    config = QualityConfig(
        stuck_threshold=8,
        causal_window=168,
        run_flagging="whole_run",
        stuck_calibration_quantile=0.999,
        calibration_end=str(frame["timestamp"].iloc[hours - 24 * 7]),
    )
    table = detector_calibration_table(frame, config).set_index("building_id")
    assert table.loc["coarse", "stuck_threshold_effective"] > 8
    assert table.loc["fine", "stuck_threshold_effective"] == 8
    masks = causal_anomaly_masks(frame, config)
    flagged_fine = frame.loc[masks["stuck"] & frame["building_id"].eq("fine")]
    flagged_coarse = frame.loc[masks["stuck"] & frame["building_id"].eq("coarse")]
    assert len(flagged_fine) == 8
    assert flagged_coarse.empty


def test_reference_calibrated_thresholds_override_quantile_on_inspected_frame() -> None:
    loads = 50 + np.zeros(24 * 7 * 3)
    loads[::2] += np.arange(len(loads[::2])) % 5  # changing values
    frame = _building("b", loads)
    frame.loc[100:107, "load"] = 9.0
    fixed = QualityConfig(
        stuck_threshold=8,
        run_flagging="whole_run",
        stuck_calibration_quantile=0.999,
        stuck_thresholds_by_building={"b": 8},
    )
    table = detector_calibration_table(frame, fixed).set_index("building_id")
    assert table.loc["b", "calibration_source"] == "reference_calibration"
    assert table.loc["b", "stuck_threshold_effective"] == 8


def test_zero_runs_during_normally_idle_hours_are_not_flagged() -> None:
    weeks = 10
    hours = 24 * 7 * weeks
    timestamps = pd.date_range("2016-01-04", periods=hours, freq="h", tz="UTC")
    active = (timestamps.hour >= 8) & (timestamps.hour < 18) & (timestamps.dayofweek < 5)
    # Varying daytime load so the synthetic meter is not itself a stuck sensor.
    loads = np.where(active, 100.0 + (timestamps.hour.to_numpy() % 10), 0.0)
    frame = _building("school", loads)
    # An unexpected outage covering the whole working day in the last week.
    day = timestamps[-24 * 3]
    outage = frame.index[(frame["timestamp"] >= day.replace(hour=8)) & (frame["timestamp"] < day.replace(hour=18))]
    frame.loc[outage, "load"] = 0.0
    config = QualityConfig(
        long_zero_threshold=6,
        run_flagging="whole_run",
        zero_profile_min_ratio=0.25,
        profile_weeks=4,
    )
    masks = causal_anomaly_masks(frame, config)
    flagged = frame.loc[masks["long_zero"]]
    # The working-day outage must be flagged. Nights and weekends are zero by
    # design and must not be flagged once a profile exists, except the night
    # hours that are contiguous with the outage: a zero run is judged and
    # flagged as one unit.
    assert set(outage).issubset(set(flagged.index))
    late = flagged.loc[flagged["timestamp"] >= timestamps[24 * 7 * 5]]
    # The Friday outage joins Thursday night and the weekend into one zero run
    # that ends when the building comes back on Monday morning.
    run_start = (day - pd.Timedelta(days=1)).replace(hour=18)
    run_end = (day + pd.Timedelta(days=3)).replace(hour=8)
    assert ((late["timestamp"] >= run_start) & (late["timestamp"] < run_end)).all()


def test_hour_of_week_profile_is_strictly_causal() -> None:
    frame = _building("b", np.arange(24 * 7 * 4, dtype=float))
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    profile = hour_of_week_profile(frame["load"], timestamps, weeks=3)
    # First occurrence of every hour-of-week has no history; the second has one
    # value only (min_periods=2), so the profile starts at the third week.
    assert profile.iloc[: 24 * 7 * 2].isna().all()
    assert np.isclose(profile.iloc[24 * 7 * 2], np.median([0.0, 24 * 7]))


def test_profile_fill_uses_only_earlier_valid_values_and_respects_limits() -> None:
    weeks = 5
    hours = 24 * 7 * weeks
    timestamps = pd.date_range("2016-01-04", periods=hours, freq="h", tz="UTC")
    loads = 10.0 + timestamps.hour.to_numpy(dtype=float)
    frame = _building("b", loads)
    gap = list(range(hours - 60, hours - 40))  # 20-hour gap in the last week
    frame.loc[gap, "load"] = np.nan
    result = remediate(
        frame,
        quality=QualityConfig(causal_window=24),
        maximum_forward_fill_hours=3,
        repair_strategy="forward_then_profile",
        maximum_profile_fill_hours=48,
        profile_weeks=4,
    )
    repaired = result.log[result.log["status"] == "repaired"]
    assert (repaired["rule"].str.startswith("past_only_forward_fill")).sum() == 3
    assert (repaired["rule"].str.startswith("past_only_hour_of_week_median")).sum() == 17
    assert result.quarantine.empty
    # Every profile-filled value equals the deterministic hour pattern, which
    # is what the earlier weeks of the same hour-of-week contain.
    for index in gap[3:]:
        assert result.frame.loc[index, "load"] == 10.0 + timestamps[index].hour
    # Forward-fill-then-profile never changes values after the repair boundary.
    later = remediate(
        frame,
        quality=QualityConfig(causal_window=24),
        maximum_forward_fill_hours=3,
        repair_strategy="forward_then_profile",
        repair_before=str(timestamps[hours - 50]),
    )
    protected = [index for index in gap if timestamps[index] >= timestamps[hours - 50]]
    assert later.frame.loc[protected, "load"].isna().all()


def test_ambiguous_scale_is_quarantined_unless_profile_fill_is_allowed() -> None:
    hours = 24 * 7 * 4
    timestamps = pd.date_range("2016-01-04", periods=hours, freq="h", tz="UTC")
    loads = 50.0 + 5 * np.sin(2 * np.pi * timestamps.hour.to_numpy() / 24)
    frame = _building("b", loads)
    shifted = list(range(hours - 30, hours - 22))
    frame.loc[shifted, "load"] = frame.loc[shifted, "load"] * 100
    quality = QualityConfig(causal_window=48, level_shift_ratio=8.0)
    strict = remediate(frame, quality=quality, repair_strategy="forward_then_profile")
    assert set(shifted).issubset(set(strict.quarantine.index)) or strict.frame.loc[shifted, "load"].isna().all()
    allowed = remediate(
        frame,
        quality=quality,
        repair_strategy="forward_then_profile",
        profile_fill_ambiguous_scale=True,
    )
    assert allowed.frame.loc[shifted, "load"].notna().all()
    assert (allowed.frame.loc[shifted, "load"] < 100).all()


def test_events_per_type_scales_the_downstream_suite(sample_inputs: tuple[Path, Path, Path]) -> None:
    from energy_onboarding.ingest import SelectionConfig, ingest_selected_dataset

    meter, metadata, _ = sample_inputs
    reference = ingest_selected_dataset(
        meter, metadata, selection=SelectionConfig(building_ids=("b1", "b2", "b3"))
    ).frame
    single = inject_downstream_faults(reference, cutoff="2016-01-10T00:00:00Z", seed=3, severity="low")
    triple = inject_downstream_faults(
        reference, cutoff="2016-01-10T00:00:00Z", seed=3, severity="low", events_per_type=3
    )
    assert len({record["fault_id"] for record in single.faults}) == 6
    assert len({record["fault_id"] for record in triple.faults}) == 18
    keys = {(record["timestamp"], record["building_id"]) for record in triple.faults}
    assert len(keys) == len(triple.faults)  # no key is corrupted twice
    with pytest.raises(ValueError):
        inject_downstream_faults(reference, cutoff="2016-01-10T00:00:00Z", seed=3, events_per_type=0)


def test_pipeline_publishes_weather_and_calibration_artifacts(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config_path = sample_inputs
    config = json.loads(config_path.read_text())
    config["quality"].update(
        {"run_flagging": "whole_run", "stuck_calibration_quantile": 0.999, "zero_profile_min_ratio": 0.25}
    )
    config["remediation"]["repair_strategy"] = "forward_then_profile"
    config["faults"]["events_per_type"] = 2
    config["weather"] = {"required": True}
    v3_config = tmp_path / "v3.json"
    v3_config.write_text(json.dumps(config), encoding="utf-8")
    timestamps = pd.date_range("2016-01-01", periods=300, freq="h")
    weather = pd.DataFrame(
        {
            "timestamp": list(timestamps.strftime("%Y-%m-%d %H:%M:%S")) * 3,
            "site_id": ["site_a"] * 300 + ["site_b"] * 300 + ["site_c"] * 300,
            "airTemperature": np.tile(20 + 5 * np.sin(np.arange(300) / 24 * 2 * np.pi), 3),
            "cloudCoverage": np.nan,
            "dewTemperature": 10.0,
            "precipDepth1HR": 0.0,
            "seaLvlPressure": 1015.0,
            "windSpeed": 2.0,
        }
    )
    weather = weather.iloc[5:]  # a few grid hours are missing at the start
    weather_path = tmp_path / "weather.csv"
    weather.to_csv(weather_path, index=False)

    with pytest.raises(ValueError, match="requires --weather-csv"):
        run_pipeline(v3_config, meter, metadata, tmp_path / "missing", fixture_mode=True)

    output = tmp_path / "result"
    run_pipeline(v3_config, meter, metadata, output, weather_csv=weather_path, fixture_mode=True)
    assert verify_result_dir(output) == []
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert "weather.csv.gz" in manifest["files"]
    assert "detector_calibration.csv" in manifest["files"]
    assert manifest["detector_policy"]["run_flagging"] == "whole_run"
    assert manifest["remediation_policy"]["repair_strategy"] == "forward_then_profile"
    assert manifest["fault_events_per_type"] == 2
    assert manifest["weather"]["grid_hours_materialized_as_null"] >= 5
    published = pd.read_csv(output / "weather.csv.gz")
    assert list(published.columns) == [
        "timestamp", "site_id", "air_temperature", "dew_temperature", "wind_speed",
        "cloud_coverage", "precip_depth_1hr", "sea_lvl_pressure",
    ]
    assert set(published["site_id"]) == {"site_a", "site_b", "site_c"}
    fault_manifest = json.loads((output / "fault_manifest.json").read_text())
    assert len(fault_manifest["fault_events"]) == 12
    calibration = pd.read_csv(output / "detector_calibration.csv")
    assert set(calibration["building_id"]) == {"b1", "b2", "b3"}
    assert (calibration["calibration_source"] == "reference_calibration").all()
    # Tampering with the weather artifact is caught.
    (output / "weather.csv.gz").write_bytes(b"not a gzip")
    assert any("hash mismatch: weather.csv.gz" in error for error in verify_result_dir(output))


def test_zero_run_exemption_is_decided_per_run_and_never_zero_filled() -> None:
    """A zero run spanning idle and active hours is flagged in full and is not
    forward-filled with its own zero."""

    weeks = 10
    hours = 24 * 7 * weeks
    timestamps = pd.date_range("2016-01-04", periods=hours, freq="h", tz="UTC")
    active = (timestamps.hour >= 8) & (timestamps.hour < 18) & (timestamps.dayofweek < 5)
    loads = np.where(active, 100.0 + (timestamps.hour.to_numpy() % 10), 0.0)
    frame = _building("school", loads)
    # An outage that starts in the idle early morning and runs into the working day.
    day = timestamps[-24 * 3]
    start = frame.index[(frame["timestamp"] >= day.replace(hour=4)) & (frame["timestamp"] < day.replace(hour=12))]
    frame.loc[start, "load"] = 0.0
    config = QualityConfig(long_zero_threshold=6, run_flagging="whole_run", zero_profile_min_ratio=0.25, profile_weeks=4)
    masks = causal_anomaly_masks(frame, config)
    run = frame.index[(frame["timestamp"] >= day.replace(hour=0)) & (frame["timestamp"] < day.replace(hour=12))]
    # The whole natural-night-plus-outage zero run is flagged, not only its working hours.
    assert masks["long_zero"].loc[run].all()
    result = remediate(frame, quality=config, maximum_forward_fill_hours=3, repair_strategy="forward_then_profile", profile_weeks=4)
    repaired = result.log[result.log["status"] == "repaired"]
    assert not ((repaired["rule"].str.startswith("past_only_forward_fill")) & (repaired["remediated_value"] == 0.0)).any()
    # Working-day hours of the outage received the profile value (100), not zero.
    working = [index for index in start if bool(active[index])]
    assert (result.frame.loc[working, "load"] > 0).all()
