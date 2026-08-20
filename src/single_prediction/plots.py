"""Figures for the training report.

All chart text is English on purpose: matplotlib has no CJK font in a default
Linux/CI container, and silently rendering every label as tofu boxes is worse
than writing them in English.

Colour rules (kept deliberately boring and consistent):

* One categorical hue per **model**, held fixed across every figure, so
  "the blue line" means the same estimator in the ROC panel and the latency
  panel. Colour follows the entity, never its rank.
* Line charts carry a second, redundant encoding (dash pattern) so the figures
  survive colour-vision deficiency and black-and-white printing.
* Single-series panels use one hue plus direct value labels rather than a
  rainbow.
* No dual-axis charts anywhere; two measures means two panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import IMAGE_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdbd6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
DASHES = [(None, None), (6, 2), (2, 2), (8, 2, 2, 2), (1, 2), (4, 1)]
POSITIVE = "#1baf7a"
NEGATIVE = "#e34948"


def _style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="600", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def _figure(rows: int = 1, cols: int = 1, width: float = 10.0, height: float = 4.4):
    figure, axes = plt.subplots(rows, cols, figsize=(width, height), facecolor=SURFACE)
    return figure, axes


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def _model_colors(names: list[str]) -> dict[str, str]:
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(sorted(names))}


# ---------------------------------------------------------------------------
def plot_model_comparison(report: dict[str, Any], path: Path) -> Path:
    """Three panels, one per decision axis: quality, calibration, cost.

    The quality panel is a dot plot with 95% bootstrap intervals, not a bar
    chart. Bars must start at zero, and at zero these four AUCs are visually
    identical — which hides the actual finding. The dot plot can zoom in
    honestly, and the overlapping whiskers *are* the argument: the models are
    statistically tied, so the decision moves to the other two panels.
    """
    candidates = report["candidates"]
    significance = report.get("selection", {}).get("significance", {})
    names = sorted(candidates, key=lambda n: candidates[n]["validation"]["roc_auc"])
    colors = _model_colors(list(candidates))
    labels = [n.replace("_", " ") for n in names]
    positions = np.arange(len(names))
    figure, axes = _figure(1, 3, width=14.0, height=4.3)

    ax = axes[0]
    for position, name in zip(positions, names, strict=True):
        point = candidates[name]["validation"]["roc_auc"]
        stats = significance.get(name)
        if stats:
            low, high = stats["auc_ci_low"], stats["auc_ci_high"]
            ax.plot([low, high], [position, position], color=colors[name], linewidth=2.5,
                    solid_capstyle="round", alpha=0.55)
        ax.plot([point], [position], marker="o", markersize=10, color=colors[name],
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(point, position + 0.30, f"{point:.4f}", ha="center", color=MUTED, fontsize=9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, color=INK, fontsize=9)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    _style(ax, "Validation ROC-AUC (95% bootstrap CI)", "higher is better — whiskers overlap = tied")

    panels = [
        (axes[1], "Brier score", [candidates[n]["validation"]["brier"] for n in names],
         "lower is better", "{:.4f}"),
        (axes[2], "Latency per prediction (ms)", [candidates[n]["latency_ms"] for n in names],
         "lower is better", "{:.1f}"),
    ]
    for ax, title, values, note, fmt in panels:
        ax.barh(positions, values, color=[colors[n] for n in names], height=0.55)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, color=INK, fontsize=9)
        ax.grid(axis="y", visible=False)
        span = max(values) or 1
        for position, value in zip(positions, values, strict=True):
            ax.text(value + span * 0.02, position, fmt.format(value), va="center",
                    color=MUTED, fontsize=9)
        ax.set_xlim(0, span * 1.22)
        _style(ax, title, note)

    figure.suptitle("Candidate scorecard — one panel per decision axis",
                    color=INK, fontsize=13, fontweight="700", x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(figure, path)


def plot_roc_pr(report: dict[str, Any], path: Path) -> Path:
    curves = report["roc_pr_curves"]
    candidates = report["candidates"]
    colors = _model_colors(list(curves))
    order = sorted(curves, key=lambda n: -candidates[n]["test"]["roc_auc"])
    figure, axes = _figure(1, 2, width=11.5, height=4.8)
    for i, name in enumerate(order):
        dash = DASHES[i % len(DASHES)]
        label_roc = f"{name.replace('_', ' ')} (AUC {candidates[name]['test']['roc_auc']:.3f})"
        line, = axes[0].plot(curves[name]["fpr"], curves[name]["tpr"], color=colors[name],
                             linewidth=2, label=label_roc)
        if dash[0]:
            line.set_dashes(list(dash))
        label_pr = f"{name.replace('_', ' ')} (AP {candidates[name]['test']['pr_auc']:.3f})"
        line2, = axes[1].plot(curves[name]["recall"], curves[name]["precision"], color=colors[name],
                              linewidth=2, label=label_pr)
        if dash[0]:
            line2.set_dashes(list(dash))
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="random")
    base_rate = report["dataset"]["positive_rate"]
    axes[1].axhline(base_rate, color=MUTED, linewidth=1, linestyle=":",
                    label=f"base rate {base_rate:.3f}")
    _style(axes[0], "ROC — separating engaged from not", "False positive rate", "True positive rate")
    _style(axes[1], "Precision–Recall — the view that respects imbalance", "Recall", "Precision")
    for ax in axes:
        legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right" if ax is axes[0] else "upper right")
        for text in legend.get_texts():
            text.set_color(MUTED)
    return _save(figure, path)


def plot_calibration(report: dict[str, Any], path: Path) -> Path:
    calibration = report["calibration"]
    raw = calibration["reliability_raw"]
    fixed = calibration["reliability_calibrated"]
    figure, axes = _figure(1, 2, width=11.5, height=4.8)

    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="perfect")
    axes[0].plot(raw["predicted"], raw["observed"], color=SERIES[1], linewidth=2,
                 marker="o", markersize=5, label="raw model")
    line, = axes[0].plot(fixed["predicted"], fixed["observed"], color=SERIES[0], linewidth=2,
                         marker="s", markersize=5, label="after isotonic calibration")
    line.set_dashes([6, 2])
    _style(axes[0], "Reliability — does 0.30 mean 30%?", "Predicted probability", "Observed frequency")

    # Three proper scoring rules on one axis would be unreadable (log loss ~0.5,
    # ECE ~0.01), so this panel shows the *relative* change each metric saw.
    # Negative bars mean calibration helped; positive means it hurt.
    labels = ["Brier", "ECE", "Log loss"]
    keys = ("brier", "ece", "log_loss")
    before = [calibration["raw_test"][k] for k in keys]
    after = [calibration["calibrated_test"][k] for k in keys]
    change = [(a - b) / b * 100 if b else 0.0 for b, a in zip(before, after, strict=True)]
    positions = np.arange(len(labels))
    colors = [POSITIVE if value < 0 else NEGATIVE for value in change]
    axes[1].barh(positions, change, color=colors, height=0.5)
    axes[1].axvline(0, color=MUTED, linewidth=1)
    for position, (value, b, a) in enumerate(zip(change, before, after, strict=True)):
        offset = 0.6 if value >= 0 else -0.6
        axes[1].text(value + offset, position, f"{value:+.1f}%   ({b:.4f} → {a:.4f})",
                     va="center", ha="left" if value >= 0 else "right",
                     color=MUTED, fontsize=9)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(labels, color=INK, fontsize=9)
    span = max(abs(min(change)), abs(max(change)), 1.0)
    axes[1].set_xlim(-span * 3.2, span * 3.2)
    axes[1].grid(axis="y", visible=False)
    _style(axes[1], "Did calibration help? (change vs raw model)",
           "negative = better after calibration")

    legend = axes[0].legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(MUTED)
    return _save(figure, path)


def plot_lift(report: dict[str, Any], path: Path) -> Path:
    lift = report["decile_lift"]
    deciles = [f"D{int(row['bucket']) + 1}" for row in lift]
    values = [row["lift"] for row in lift]
    capture = [row["cumulative_capture"] for row in lift]
    figure, axes = _figure(1, 2, width=11.5, height=4.4)

    axes[0].bar(deciles, values, color=SERIES[0], width=0.62)
    axes[0].axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    for i, value in enumerate(values):
        axes[0].text(i, value, f"{value:.2f}x", ha="center", va="bottom", color=MUTED, fontsize=8.5)
    axes[0].grid(axis="x", visible=False)
    _style(axes[0], "Lift by score decile (1.0x = random)", "Score decile, best first", "Lift")

    axes[1].plot(deciles, capture, color=SERIES[2], linewidth=2, marker="o", markersize=6)
    axes[1].plot(deciles, [(i + 1) / len(deciles) for i in range(len(deciles))],
                 color=MUTED, linewidth=1, linestyle=":")
    for i, value in enumerate(capture):
        axes[1].annotate(f"{value:.0%}", xy=(i, value), xytext=(0, 7),
                         textcoords="offset points", ha="center", color=MUTED, fontsize=8.5)
    axes[1].grid(axis="x", visible=False)
    _style(axes[1], "Cumulative share of engaged users captured", "Score decile, best first", "Captured")
    return _save(figure, path)


def plot_threshold(report: dict[str, Any], path: Path) -> Path:
    sweep = report["threshold_sweep"]
    best = report["recommended_threshold"]
    figure, ax = _figure(width=8.6, height=4.6)
    ax.plot(sweep["threshold"], sweep["precision"], color=SERIES[0], linewidth=2, label="precision")
    recall_line, = ax.plot(sweep["threshold"], sweep["recall"], color=SERIES[1], linewidth=2,
                           label="recall")
    recall_line.set_dashes([6, 2])
    f1_line, = ax.plot(sweep["threshold"], sweep["f1"], color=SERIES[2], linewidth=2, label="F1")
    f1_line.set_dashes([2, 2])
    ax.axvline(best, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate(f"F1-optimal {best:.2f}", xy=(best, 0.05), xytext=(6, 0),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=":", alpha=0.5)
    ax.annotate("default 0.5", xy=(0.5, 0.95), xytext=(-64, 0),
                textcoords="offset points", color=MUTED, fontsize=9)
    _style(ax, "Choosing an operating point", "Decision threshold", "Score")
    legend = ax.legend(frameon=False, fontsize=9, loc="center right")
    for text in legend.get_texts():
        text.set_color(MUTED)
    return _save(figure, path)


def render_all(report: dict[str, Any], image_dir: Path = IMAGE_DIR) -> list[Path]:
    """Render every figure referenced by the README."""
    jobs = [
        (plot_model_comparison, "model_comparison.png"),
        (plot_roc_pr, "roc_pr_curves.png"),
        (plot_calibration, "calibration.png"),
        (plot_lift, "lift_deciles.png"),
        (plot_threshold, "threshold_tradeoff.png"),
        (plot_operating_points, "operating_points.png"),
    ]
    rendered = []
    for function, filename in jobs:
        try:
            rendered.append(function(report, image_dir / filename))
        except Exception as exc:  # noqa: BLE001 - a broken figure must not fail training
            print(f"[warn] failed to render {filename}: {exc}", flush=True)
    return rendered


def plot_benchmark(payload: dict[str, Any], image_dir: Path = IMAGE_DIR) -> Path:
    """Two panels: absolute cost per batch size, and the vectorisation speed-up."""
    rows = payload["rows"]
    sizes = [row["batch_size"] for row in rows]
    labels = [str(size) for size in sizes]
    figure, axes = _figure(1, 2, width=11.5, height=4.4)

    positions = np.arange(len(sizes))
    width = 0.38
    vectorised = [row["vectorised_inference_ms"] for row in rows]
    looped = [row["row_by_row_inference_ms"] for row in rows]
    axes[0].bar(positions - width / 2 - 0.01, looped, width, color=SERIES[1], label="row-by-row loop")
    axes[0].bar(positions + width / 2 + 0.01, vectorised, width, color=SERIES[0], label="one vectorised call")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, color=INK, fontsize=9)
    axes[0].set_yscale("log")
    axes[0].grid(axis="x", visible=False)
    for position, value in zip(positions, looped, strict=True):
        axes[0].text(position - width / 2 - 0.01, value, f"{value:.0f}", ha="center", va="bottom",
                     color=MUTED, fontsize=8)
    for position, value in zip(positions, vectorised, strict=True):
        axes[0].text(position + width / 2 + 0.01, value, f"{value:.1f}", ha="center", va="bottom",
                     color=MUTED, fontsize=8)
    _style(axes[0], "Inference cost, same rows, two strategies", "Rows in the batch",
           "Milliseconds (log scale)")
    legend = axes[0].legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(MUTED)

    per_row = [row["ms_per_row"] for row in rows]
    axes[1].plot(labels, per_row, color=SERIES[2], linewidth=2, marker="o", markersize=7)
    for i, value in enumerate(per_row):
        axes[1].annotate(f"{value:.2f}", xy=(i, value), xytext=(0, 8), textcoords="offset points",
                         ha="center", color=MUTED, fontsize=8.5)
    axes[1].grid(axis="x", visible=False)
    _style(axes[1], "Amortised cost per row — why bigger batches are cheaper",
           "Rows in the batch", "Milliseconds per row")
    return _save(figure, image_dir / "benchmark_batch.png")


def plot_operating_points(report: dict[str, Any], path: Path) -> Path:
    """Lift and captured positives at each traffic budget.

    This replaces the F1-optimal threshold as the operating-point guidance. On a
    weak-signal problem F1 degenerates to flagging everything, so the useful
    question is what a budget buys.
    """
    points = report.get("operating_points", [])
    if not points:
        return path
    labels = [f"{row['actual_flag_rate']:.0%}" for row in points]
    lifts = [row["lift"] for row in points]
    recalls = [row["recall"] for row in points]
    figure, axes = _figure(1, 2, width=11.5, height=4.4)

    axes[0].bar(labels, lifts, color=SERIES[0], width=0.6)
    axes[0].axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    for i, value in enumerate(lifts):
        axes[0].text(i, value, f"{value:.2f}x", ha="center", va="bottom", color=MUTED, fontsize=9)
    axes[0].grid(axis="x", visible=False)
    _style(axes[0], "Lift at each traffic budget (1.0x = random)",
           "Share of traffic flagged", "Lift over base rate")

    axes[1].plot(labels, recalls, color=SERIES[2], linewidth=2, marker="o", markersize=7)
    axes[1].plot(labels, [row["actual_flag_rate"] for row in points], color=MUTED,
                 linewidth=1, linestyle=":")
    for i, value in enumerate(recalls):
        axes[1].annotate(f"{value:.0%}", xy=(i, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", color=MUTED, fontsize=9)
    axes[1].grid(axis="x", visible=False)
    _style(axes[1], "Engaged users captured (dotted = random targeting)",
           "Share of traffic flagged", "Recall")
    return _save(figure, path)
