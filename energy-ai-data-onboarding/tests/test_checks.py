from __future__ import annotations

import numpy as np
import pandas as pd

from energy_onboarding.checks import QualityConfig, causal_anomaly_masks, run_quality_checks


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2016-01-01", periods=30, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "building_id": "b1",
            "load": np.arange(1, 31, dtype=float),
            "site_id": "s1",
            "primary_use": "Office",
            "timezone": "US/Eastern",
        }
    )


def test_missing_negative_infinite_duplicate_and_metadata_checks() -> None:
    frame = _frame()
    frame.loc[5, "load"] = np.nan
    frame.loc[6, "load"] = -2
    frame.loc[7, "load"] = np.inf
    frame.loc[8, "site_id"] = pd.NA
    frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    outcomes = {record.code: record for record in run_quality_checks(frame)}
    assert outcomes["LOAD_MISSING"].count == 1
    assert outcomes["NEGATIVE_LOAD"].count == 1
    assert outcomes["LOAD_INFINITE"].count == 1
    assert outcomes["DUPLICATE_OBSERVATION_KEY"].status == "fail"
    assert outcomes["METADATA_JOIN_MISSING"].status == "fail"


def test_long_zero_stuck_and_causal_outlier_masks() -> None:
    frame = _frame()
    frame.loc[10:15, "load"] = 0.0
    frame.loc[18:24, "load"] = 7.123
    frame.loc[29, "load"] = 100000.0
    masks = causal_anomaly_masks(
        frame,
        QualityConfig(
            long_zero_threshold=4,
            stuck_threshold=5,
            causal_window=12,
            level_shift_ratio=5,
        ),
    )
    assert masks["long_zero"].any()
    assert masks["stuck"].any()
    assert masks["outlier"].loc[29]
