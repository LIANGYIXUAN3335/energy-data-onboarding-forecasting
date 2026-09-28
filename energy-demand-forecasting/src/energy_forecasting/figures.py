from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_ORDER = ["reference", "corrupted", "remediated"]


def _ordered_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(CONDITION_ORDER)}
    shown = metrics.copy()
    shown["_condition_order"] = shown["condition"].map(order).fillna(len(order))
    return shown.sort_values(["model", "_condition_order", "condition"]).drop(
        columns="_condition_order"
    )


def render_metric_bars_svg(
    metrics: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    axis_label: str,
) -> str:
    """Render a deterministic, dependency-free horizontal bar chart."""

    required = {"condition", "model", value_column}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Cannot render {value_column}: missing {sorted(missing)}")
    shown = _ordered_metrics(metrics)
    width = 960
    left = 310
    right = 55
    top = 92
    row_height = 42
    bottom = 82
    height = top + len(shown) * row_height + bottom
    plot_width = width - left - right
    finite = shown[value_column].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    maximum = maximum if maximum > 0 else 1.0
    colors = {
        "reference": "#2f6b3b",
        "corrupted": "#b64b3c",
        "remediated": "#3568a8",
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{escape(title)}</title>",
        (
            "<desc id=\"desc\">Deterministic chart generated from metrics.csv; "
            f"horizontal bars show {escape(axis_label)}.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="36" font-family="system-ui, sans-serif" '
            f'font-size="24" font-weight="700" fill="#17202a">{escape(title)}</text>'
        ),
        (
            f'<text x="{left}" y="63" font-family="system-ui, sans-serif" '
            f'font-size="13" fill="#4d5656">{escape(axis_label)}; lower is better</text>'
        ),
    ]
    for tick in range(5):
        ratio = tick / 4
        x = left + ratio * plot_width
        value = ratio * maximum
        lines.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{top - 12}" x2="{x:.2f}" '
                    f'y2="{top + len(shown) * row_height}" stroke="#e5e7e9"/>'
                ),
                (
                    f'<text x="{x:.2f}" y="{height - 43}" text-anchor="middle" '
                    f'font-family="ui-monospace, monospace" font-size="12" '
                    f'fill="#566573">{value:.4g}</text>'
                ),
            ]
        )
    for position, row in enumerate(shown.itertuples(index=False)):
        condition = str(row.condition)
        model = str(row.model)
        value = float(getattr(row, value_column))
        y = top + position * row_height
        label = f"{model} / {condition}"
        lines.append(
            f'<text x="{left - 12}" y="{y + 22}" text-anchor="end" '
            f'font-family="system-ui, sans-serif" font-size="13" '
            f'fill="#17202a">{escape(label)}</text>'
        )
        if np.isfinite(value) and value >= 0:
            bar_width = value / maximum * plot_width
            lines.extend(
                [
                    (
                        f'<rect x="{left}" y="{y + 7}" width="{bar_width:.2f}" '
                        f'height="21" rx="2" fill="{colors.get(condition, "#707b7c")}"/>'
                    ),
                    (
                        f'<text x="{min(left + bar_width + 7, width - 48):.2f}" '
                        f'y="{y + 22}" font-family="ui-monospace, monospace" '
                        f'font-size="12" fill="#17202a">{value:.4f}</text>'
                    ),
                ]
            )
        else:
            lines.append(
                f'<text x="{left + 7}" y="{y + 22}" font-family="system-ui, sans-serif" '
                f'font-size="12" fill="#7b7d7d">NA</text>'
            )
    lines.extend(
        [
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 13}" '
                f'text-anchor="middle" font-family="system-ui, sans-serif" '
                f'font-size="13" fill="#34495e">{escape(axis_label)}</text>'
            ),
            "</svg>",
            "",
        ]
    )
    return "\n".join(lines)


def render_horizon_lines_svg(horizon_metrics: pd.DataFrame) -> str:
    """Render deterministic per-horizon MAE lines for sensitivity outputs."""

    required = {"condition", "model", "horizon_hours", "mae"}
    missing = required - set(horizon_metrics.columns)
    if missing:
        raise ValueError(f"Cannot render horizon chart: missing {sorted(missing)}")
    frame = horizon_metrics.copy()
    width, height = 1040, 600
    left, right, top, bottom = 92, 45, 75, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite = frame["mae"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    maximum = maximum if maximum > 0 else 1.0
    colors = {
        "reference": "#2f6b3b",
        "corrupted": "#b64b3c",
        "remediated": "#3568a8",
    }
    dashes = {"seasonal_naive": "8,5", "hist_gradient_boosting": "none"}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">Direct 1–24 hour sensitivity MAE by horizon</title>',
        '<desc id="desc">Deterministic lines generated from horizon_metrics.csv.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="system-ui, sans-serif" font-size="24" font-weight="700" fill="#17202a">Direct 1–24 hour sensitivity</text>',
        f'<text x="{left}" y="58" font-family="system-ui, sans-serif" font-size="13" fill="#4d5656">MAE by target horizon; lower is better</text>',
    ]
    for tick in range(5):
        ratio = tick / 4
        y = top + plot_height - ratio * plot_height
        value = ratio * maximum
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7e9"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="ui-monospace, monospace" font-size="12" fill="#566573">{value:.4g}</text>',
            ]
        )
    for horizon in (1, 4, 8, 12, 16, 20, 24):
        x = left + (horizon - 1) / 23 * plot_width
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f2f3f4"/>',
                f'<text x="{x:.2f}" y="{height - 46}" text-anchor="middle" font-family="ui-monospace, monospace" font-size="12" fill="#566573">{horizon}</text>',
            ]
        )
    legend_y = height - 15
    for index, ((condition, model), group) in enumerate(
        frame.groupby(["condition", "model"], observed=True, sort=True)
    ):
        ordered = group.sort_values("horizon_hours")
        points = []
        for row in ordered.itertuples(index=False):
            value = float(row.mae)
            if not np.isfinite(value):
                continue
            x = left + (int(row.horizon_hours) - 1) / 23 * plot_width
            y = top + plot_height - value / maximum * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        color = colors.get(str(condition), "#707b7c")
        dash = dashes.get(str(model), "none")
        if points:
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                f'stroke-width="2.5" stroke-dasharray="{dash}"/>'
            )
        legend_x = left + index * 145
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 25}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>',
                f'<text x="{legend_x + 30}" y="{legend_y}" font-family="system-ui, sans-serif" font-size="10" fill="#34495e">{escape(str(condition))}/{escape(str(model))}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 46}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="13" fill="#34495e">Horizon (hours)</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="system-ui, sans-serif" font-size="13" fill="#34495e">MAE</text>',
            "</svg>",
            "",
        ]
    )
    return "\n".join(lines)


def write_canonical_figures(metrics: pd.DataFrame, output_dir: Path) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            "figures/mae_by_condition.svg",
            render_metric_bars_svg(
                metrics,
                value_column="mae",
                title="Test MAE by condition and model",
                axis_label="MAE (reference-only test targets)",
            ),
        ),
        (
            "figures/macro_building_mase.svg",
            render_metric_bars_svg(
                metrics,
                value_column="mase_macro_building",
                title="Supplementary macro building MASE",
                axis_label="Unweighted mean of finite per-building MASE",
            ),
        ),
    ]
    for relative, content in specifications:
        (output_dir / relative).write_text(content, encoding="utf-8")
    return [relative for relative, _ in specifications]
