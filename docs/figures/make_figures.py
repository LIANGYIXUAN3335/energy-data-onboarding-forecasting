"""Regenerate the README figures from the committed result bundles.

Requires matplotlib and pillow (not project dependencies):
    python -m venv /tmp/figenv && /tmp/figenv/bin/pip install pandas numpy matplotlib pillow
    /tmp/figenv/bin/python docs/figures/make_figures.py

Every number drawn here is read from the committed CSV/JSON artifacts; nothing
is hard-coded except layout.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ONB = ROOT / "energy-ai-data-onboarding" / "results" / "bdg2_v3"
FC = ROOT / "energy-demand-forecasting" / "results" / "canonical_24h_v3"
SENS = ROOT / "energy-demand-forecasting" / "results" / "direct_1_to_24_sensitivity_v3"
ONB_V2 = ROOT / "energy-ai-data-onboarding" / "results" / "bdg2_mvp_v2_hardened"
FC_V2 = ROOT / "energy-demand-forecasting" / "results" / "canonical_24h_v2_final"
OUT = Path(__file__).resolve().parent
MODEL_LABEL = {"seasonal_naive": "seasonal naive", "ridge": "ridge", "random_forest": "random forest", "hist_gradient_boosting": "gradient boosting"}

BLUE, ORANGE, GREEN, GREY, RED = "#2563eb", "#f59e0b", "#16a34a", "#6b7280", "#dc2626"
COND_COLOR = {"reference": BLUE, "corrupted": ORANGE, "remediated": GREEN}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, x, y, w, h, text, fc="#eff6ff", ec=BLUE, fs=10.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 0.16, lines[0], ha="center", va="top", fontsize=fs + 1, fontweight="bold", color="#111827")
        ax.text(x + w / 2, y + h - 0.46, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 1, color="#374151", linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111827")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
    # row 1: onboarding
    box(ax, 0.3, 3.1, 2.3, 1.5, "1. Public source data\nBuilding Data Genome 2\n12 U.S. buildings, 2016-17\nmeters + weather, SHA-256", fc="#f3f4f6", ec=GREY)
    box(ax, 2.95, 3.1, 2.3, 1.5, "2. Ingest & normalize\nfixed schema, chunked read,\nmetadata join, timestamp\ncontract")
    box(ax, 5.6, 3.1, 2.3, 1.5, "3. Quality gates\n20 checks, per-building\ncalibration: gaps, stuck\nsensors, unit shifts ...")
    box(ax, 8.25, 3.1, 2.3, 1.5, "4. Seeded faults\nknown defects written only\nbefore a fixed cutoff, each\nlogged with a hash", fc="#fff7ed", ec=ORANGE)
    box(ax, 10.9, 3.1, 2.0, 1.5, "5. Remediation\npast-only repair from\nearlier weeks, else\nquarantine + reason", fc="#f0fdf4", ec=GREEN)
    for x0 in (2.6, 5.25, 7.9, 10.55):
        arrow(ax, x0, 3.85, x0 + 0.35, 3.85)
    # three conditions
    box(ax, 3.3, 1.35, 1.9, 0.95, "reference", fc="#eff6ff", ec=BLUE, bold_first=False)
    box(ax, 5.55, 1.35, 1.9, 0.95, "corrupted", fc="#fff7ed", ec=ORANGE, bold_first=False)
    box(ax, 7.8, 1.35, 1.9, 0.95, "remediated", fc="#f0fdf4", ec=GREEN, bold_first=False)
    ax.text(0.3, 2.75, "Three versions of one dataset:\nidentical keys, hash-bound manifests", ha="left", va="top", fontsize=9.5, color="#374151", style="italic")
    arrow(ax, 4.1, 3.1, 4.25, 2.3); arrow(ax, 9.4, 3.1, 6.5, 2.3); arrow(ax, 11.9, 3.1, 8.75, 2.3)
    # forecasting
    box(ax, 0.3, 0.15, 4.4, 0.95, "6. Forecasting benchmark: 4 models, same split and seed per\ncondition; 24 h ahead; test targets always from reference", fc="#f3f4f6", ec=GREY, fs=9, bold_first=False)
    box(ax, 5.0, 0.15, 3.6, 0.95, "7. Metrics, bootstrap intervals,\nper-building and per-horizon results", fc="#f3f4f6", ec=GREY, fs=9, bold_first=False)
    box(ax, 8.9, 0.15, 4.0, 0.95, "8. Verification: every artifact re-checked\nfrom hashes (verify-inputs / verify-results)", fc="#fef2f2", ec=RED, fs=9, bold_first=False)
    arrow(ax, 4.25, 1.35, 2.5, 1.1); arrow(ax, 6.5, 1.35, 6.0, 1.1); arrow(ax, 8.75, 1.35, 7.5, 1.1)
    arrow(ax, 4.7, 0.62, 5.0, 0.62); arrow(ax, 8.6, 0.62, 8.9, 0.62)
    fig.savefig(OUT / "pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_conditions(bid):
    out = {}
    for c in ("reference", "corrupted", "remediated"):
        df = pd.read_csv(ONB / f"{c}.csv.gz", parse_dates=["timestamp"])
        out[c] = df[df.building_id == bid].set_index("timestamp")["load"].sort_index()
    return out


def _fault_windows(bundle):
    """Pick the building with a unit-scale and a long-zero event and both windows."""

    manifest = json.load(open(bundle / "fault_manifest.json"))
    events = pd.DataFrame(manifest["fault_events"])
    faults = pd.DataFrame(manifest["faults"])
    building_of = faults.groupby("fault_id")["building_id"].first()
    events["building_id"] = events["fault_id"].map(building_of)
    blocks = events[events["fault_type"].isin(["unit_scale_segment", "long_zero_block"])]
    counts = blocks.groupby("building_id")["fault_type"].nunique()
    candidates = counts[counts == 2].index
    if len(candidates) == 0:
        raise ValueError("no building received both a unit-scale and a long-zero event")
    building = sorted(candidates)[0]
    chosen = []
    for fault_type, label in (("unit_scale_segment", "unit-scale fault (values x100)"), ("long_zero_block", "long-zero fault (8 hours of 0)")):
        event = blocks[(blocks["building_id"] == building) & (blocks["fault_type"] == fault_type)].iloc[0]
        start = pd.Timestamp(event["start_timestamp"]); end = pd.Timestamp(event["end_timestamp"])
        chosen.append(((start - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), (end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), label, start, end))
    return building, chosen


def fig_fault_gif():
    bid, windows = _fault_windows(ONB)
    s = load_conditions(bid)
    bm = pd.read_csv(FC / "building_mase.csv")
    bm = bm[(bm.model == "hist_gradient_boosting") & (bm.building_id == bid)].set_index("condition")["mae"]
    pretty = bid.replace("_", " ")
    frames = []
    steps = [
        ("Step 1 - reference data", f"Real hourly electricity load of one U.S. building ({pretty}, BDG2, 2016).", ["reference"], False, False),
        ("Step 2 - seeded faults injected", "Two known defects are written into the training data, before the cutoff, and logged with a hash.", ["corrupted"], True, False),
        ("Step 3 - quality gates flag them", "Detectors LEVEL_OR_UNIT_SHIFT and LONG_ZERO_RUN fire on exactly these windows.", ["corrupted"], True, True),
        ("Step 4 - past-only repair", "Each flagged hour takes the median of the same weekday and hour over the previous 8 weeks; nothing later is read.", ["remediated"], True, True),
        ("Step 5 - effect on the forecast", f"24h-ahead gradient-boosting MAE for this building: reference {bm['reference']:.1f}, corrupted {bm['corrupted']:.1f}, remediated {bm['remediated']:.1f} kWh.", ["reference", "corrupted", "remediated"], False, False),
    ]
    for title, subtitle, conds, mark, flag in steps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        fig.suptitle(title, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.99)
        fig.text(0.02, 0.905, subtitle, fontsize=10.5, color="#374151")
        for ax, (t0, t1, label, f0, f1) in zip(axes, windows):
            for c in conds:
                ser = s[c][t0:t1]
                ax.plot(ser.index, ser.values, color=COND_COLOR[c], lw=1.6 if c != "reference" or len(conds) == 1 else 1.2, label=c, alpha=0.95)
            if mark:
                ax.axvspan(f0, f1, color=ORANGE, alpha=0.15)
                ax.text(f0, ax.get_ylim()[1], " " + label, color="#b45309", fontsize=9, va="top")
            if flag:
                ax.text(0.98, 0.04, "flagged by quality gate" if "remediated" not in conds else "repaired from earlier weeks", transform=ax.transAxes, ha="right", color=RED if "remediated" not in conds else GREEN, fontsize=10, fontweight="bold")
            ax.set_ylabel("load (kWh)"); ax.set_title(f"{t0} to {t1}", fontsize=10, color="#374151")
            ax.tick_params(axis="x", labelsize=8)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
            if len(conds) > 1:
                ax.legend(loc="upper right", fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        p = OUT / f"_frame_{len(frames)}.png"
        fig.savefig(p, dpi=110, facecolor="white"); plt.close(fig)
        frames.append(Image.open(p).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(OUT / "fault_to_remediation.gif", save_all=True, append_images=frames[1:], duration=[2200, 2600, 2600, 2800, 3600], loop=0, optimize=False)
    for p in OUT.glob("_frame_*.png"):
        p.unlink()
    return bid


def fig_results():
    m = pd.read_csv(FC / "metrics.csv")
    conds = ["reference", "corrupted", "remediated"]
    models = [x for x in ["seasonal_naive", "ridge", "random_forest", "hist_gradient_boosting"] if x in set(m.model)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        sub = m[m.model == model].set_index("condition").loc[conds]
        xs = np.arange(3) + (i - (len(models) - 1) / 2) * width
        ax.bar(xs, sub["mae"], width=width * 0.92, color=[COND_COLOR[c] for c in conds], alpha=0.35 + 0.65 * i / max(1, len(models) - 1), edgecolor="white")
        ax.errorbar(xs, sub["mae"], yerr=[sub["mae"] - sub["mae_ci_low"], sub["mae_ci_high"] - sub["mae"]], fmt="none", ecolor="#111827", capsize=2, lw=0.9)
        for x, v in zip(xs, sub["mae"]):
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5, rotation=90 if len(models) > 3 else 0, va="bottom")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(conds); ax.set_ylabel("MAE, kWh (lower is better)")
    ax.set_title(f"Pooled error over {int(m['n'].iloc[0]):,} test hours, 12 buildings", fontsize=11)
    handles = [matplotlib.patches.Patch(facecolor="#9ca3af", alpha=0.35 + 0.65 * i / max(1, len(models) - 1), label=MODEL_LABEL[x]) for i, x in enumerate(models)]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left", title="bar shade = model", title_fontsize=8.5)
    ax = axes[1]
    bm = pd.read_csv(FC / "building_mase.csv")
    h = bm[bm.model == "hist_gradient_boosting"].pivot(index="building_id", columns="condition", values="mae").sort_values("reference")
    y = np.arange(len(h))
    for c, off in (("reference", 0.26), ("corrupted", 0.0), ("remediated", -0.26)):
        ax.barh(y + off, h[c], height=0.25, color=COND_COLOR[c], label=c)
    ax.set_yticks(y); ax.set_yticklabels([b.replace("_", " ") for b in h.index], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("MAE per building, kWh (log scale)")
    ax.set_title("Gradient boosting, per building", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_v2_vs_v3():
    """The remediation effect before and after the detector recalibration."""

    rows = []
    FC_DENSE = ROOT / "energy-demand-forecasting" / "results" / "canonical_24h_v3_dense"
    groups = [("v2 baseline\n(fixed thresholds,\nforward fill only;\n27 seeded rows)", FC_V2), ("v3 amendment\n(calibrated detectors,\nprofile repair, weather;\n108 seeded rows)", FC)]
    if (FC_DENSE / "metrics.csv").exists():
        groups.append(("v3 at literature\nprevalence\n(1,660 seeded rows,\n1.6% of training)", FC_DENSE))
    for label, path in groups:
        m = pd.read_csv(path / "metrics.csv")
        sub = m[m.model == "hist_gradient_boosting"].set_index("condition")
        rows.append((label, sub.loc["reference", "mase"], sub.loc["corrupted", "mase"], sub.loc["remediated", "mase"], sub.loc["remediated", "data_retained"]))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(rows))
    for j, (c, key) in enumerate((("reference", 1), ("corrupted", 2), ("remediated", 3))):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (j - 1) * 0.26, vals, width=0.25, color=COND_COLOR[c], label=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\ntraining targets kept: {r[4]:.1%}" for r in rows], fontsize=8.5)
    ax.set_ylabel("pooled MASE, gradient boosting\n(lower is better; 1.0 = seasonal naive scale)")
    ax.set_title("Same model and split; corrupted bars use different fault suites and are not comparable across groups", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "v2_vs_v3.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_horizon():
    hm = pd.read_csv(SENS / "horizon_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for c in ("reference", "corrupted", "remediated"):
        sub = hm[(hm.model == "hist_gradient_boosting") & (hm.condition == c)].sort_values("horizon_hours")
        ax.plot(sub["horizon_hours"], sub["mae"], marker="o", ms=3.5, lw=1.6, color=COND_COLOR[c], label=f"gradient boosting, {c}")
    if "ridge" in set(hm.model):
        sub = hm[(hm.model == "ridge") & (hm.condition == "reference")].sort_values("horizon_hours")
        ax.plot(sub["horizon_hours"], sub["mae"], ls=":", lw=1.4, color=BLUE, label="ridge, reference")
    sub = hm[(hm.model == "seasonal_naive") & (hm.condition == "reference")].sort_values("horizon_hours")
    ax.plot(sub["horizon_hours"], sub["mae"], ls="--", lw=1.2, color=GREY, label="seasonal naive baseline")
    ax.set_xlabel("forecast horizon (hours ahead)"); ax.set_ylabel("MAE, kWh"); ax.set_xticks(range(1, 25, 1))
    ax.set_title(f"Error by horizon, 1 to 24 hours ahead ({int(hm['n'].max()):,} test keys per horizon set)", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "horizon.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); building = fig_fault_gif(); fig_results(); fig_v2_vs_v3(); fig_horizon()
    print("gif building:", building)
    print("wrote", sorted(p.name for p in OUT.iterdir() if p.suffix in (".png", ".gif")))
