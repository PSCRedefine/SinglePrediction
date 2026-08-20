"""Evaluation metrics.

Three families, each answering a different question:

* **Discrimination** (ROC-AUC, PR-AUC) — does the model rank engaged
  interactions above non-engaged ones?
* **Calibration** (log loss, Brier, ECE) — when it says 0.30, do 30% of that
  group actually engage?
* **Operating point** (precision/recall/F1 across thresholds) — where should the
  probability be cut, given what the downstream action costs?

Accuracy is deliberately not a headline. At a 27.8% positive rate a
constant-negative classifier is 72% accurate and completely useless, which is
exactly the failure mode the previous model exhibited (F1 = 0.0 at threshold
0.5) without anything flagging it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, bins: int = 15) -> float:
    """Size-weighted average gap between predicted probability and observed rate."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, bins - 1)
    total, error = len(y_prob), 0.0
    for bucket in range(bins):
        mask = index == bucket
        count = int(mask.sum())
        if count:
            error += count / total * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(error)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """The full scorecard for one model on one split."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 6),
        "ece": round(expected_calibration_error(y_true, y_prob), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "flagged_rate": round(float(y_pred.mean()), 6),
        "mean_probability": round(float(y_prob.mean()), 6),
    }


def reliability_curve(y_true, y_prob, bins: int = 15):
    """Points for a reliability diagram: (mean predicted, observed, count)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, bins - 1)
    predicted, observed, counts = [], [], []
    for bucket in range(bins):
        mask = index == bucket
        if mask.any():
            predicted.append(float(y_prob[mask].mean()))
            observed.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(predicted), np.array(observed), np.array(counts)


def threshold_sweep(y_true, y_prob, steps: int = 99) -> pd.DataFrame:
    """Precision, recall and F1 at every operating point.

    0.5 is an inheritance from balanced textbook problems. With a 27.8% base
    rate it is almost never the business-optimal cut, so the whole curve is
    reported and the F1-optimal point is stored in the model metadata.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for threshold in np.linspace(0.01, 0.99, steps):
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "flagged_rate": float(y_pred.mean()),
        })
    return pd.DataFrame(rows)


def decile_lift(y_true, y_prob, buckets: int = 10) -> pd.DataFrame:
    """Sort by score, cut into equal buckets, report lift over the base rate."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": np.asarray(y_prob, dtype=float)})
    frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
    frame["bucket"] = np.minimum((np.arange(len(frame)) * buckets) // len(frame), buckets - 1)
    base = frame["y"].mean()
    grouped = frame.groupby("bucket").agg(rows=("y", "size"), positives=("y", "sum"),
                                          mean_probability=("p", "mean"))
    grouped["actual_rate"] = grouped["positives"] / grouped["rows"]
    grouped["lift"] = grouped["actual_rate"] / base
    grouped["cumulative_capture"] = grouped["positives"].cumsum() / frame["y"].sum()
    return grouped.reset_index()


def paired_bootstrap_auc(y_true, probabilities: dict, resamples: int = 300, seed: int = 0):
    """Is model A really better than model B, or did it get lucky on this split?

    Resamples the evaluation rows with replacement and re-scores **every model on
    the same resample**, so the shared noise cancels. If a model's gap to the
    leader has an interval containing zero, the two are statistically tied and
    the choice between them cannot be made on accuracy.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    names = list(probabilities)
    leader = max(names, key=lambda n: roc_auc_score(y_true, probabilities[n]))
    draws = {n: [] for n in names}
    gaps = {n: [] for n in names}
    size = len(y_true)
    for _ in range(resamples):
        index = rng.integers(0, size, size)
        sample_y = y_true[index]
        if sample_y.min() == sample_y.max():
            continue
        scores = {n: float(roc_auc_score(sample_y, probabilities[n][index])) for n in names}
        for n in names:
            draws[n].append(scores[n])
            gaps[n].append(scores[leader] - scores[n])
    summary = {}
    for n in names:
        own, gap = np.array(draws[n]), np.array(gaps[n])
        low, high = float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))
        summary[n] = {
            "auc": round(float(roc_auc_score(y_true, probabilities[n])), 6),
            "auc_std_error": round(float(own.std(ddof=1)), 6),
            "auc_ci_low": round(float(np.percentile(own, 2.5)), 6),
            "auc_ci_high": round(float(np.percentile(own, 97.5)), 6),
            "gap_to_best": round(float(gap.mean()), 6),
            "gap_ci_low": round(low, 6),
            "gap_ci_high": round(high, 6),
            "statistically_tied_with_best": bool(low <= 0.0 <= high),
        }
    summary[leader]["statistically_tied_with_best"] = True
    return summary


def operating_points(y_true, y_prob, targets=(0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """Thresholds chosen by how much traffic they flag, not by F1.

    On a weak-signal problem F1 optimisation degenerates. Here the base rate is
    27.8%, so "flag everything" scores F1 = 0.438 and no threshold beats it —
    F1 is telling us the model cannot outperform blanket treatment on that
    particular trade-off, which is true and useless as guidance.

    What a product team can actually act on is: *if I can afford to treat X% of
    traffic, how much better than random is that group?* Each row below answers
    that question for one budget, with the threshold that implements it.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    base = y_true.mean()
    rows = []
    seen: set[float] = set()
    for target in targets:
        threshold = float(np.quantile(y_prob, 1 - target))
        # Isotonic calibration is a step function. With this little signal it
        # collapses to a handful of distinct probabilities, so several budgets
        # resolve to the same cut. Duplicates are dropped rather than shown as
        # separate options that are in fact the same operating point.
        if round(threshold, 6) in seen:
            continue
        seen.add(round(threshold, 6))
        flagged = y_prob >= threshold
        if flagged.sum() == 0:
            continue
        precision = float(y_true[flagged].mean())
        rows.append({
            "target_flag_rate": target,
            "threshold": round(threshold, 6),
            "actual_flag_rate": round(float(flagged.mean()), 6),
            "precision": round(precision, 6),
            "recall": round(float(y_true[flagged].sum() / max(y_true.sum(), 1)), 6),
            "lift": round(precision / base if base else 0.0, 4),
        })
    return pd.DataFrame(rows)


def recommend_threshold(y_true, y_prob, minimum_flag_rate: float = 0.10) -> dict:
    """Pick the operating point with the highest lift that still has volume.

    A volume floor is required: the highest-lift threshold is usually one that
    flags a handful of rows, which is precise and operationally pointless. The
    floor makes the recommendation something a campaign could actually run.
    """
    points = operating_points(y_true, y_prob)
    usable = points[points["actual_flag_rate"] >= minimum_flag_rate]
    chosen = (usable if not usable.empty else points).sort_values("lift", ascending=False).iloc[0]
    return {
        "threshold": float(chosen["threshold"]),
        "flag_rate": float(chosen["actual_flag_rate"]),
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "lift": float(chosen["lift"]),
        "rule": f"highest lift among operating points flagging at least "
                f"{minimum_flag_rate:.0%} of traffic",
    }
