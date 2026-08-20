"""Train the four candidate models, compare them with evidence, ship the winner.

Two decisions distinguish this from a default training script, and both change
the reported numbers materially.

**The split is chronological.** The log covers one month. A random row split lets
the model learn from August to predict July, and puts rows from the same viewing
session on both sides. Both inflate the score. Training uses the earliest 70% of
interactions, validates on the next 15% and reports on the latest 15% — the same
direction as production, where the past predicts the future. The script also
reports what a random split *would* have produced, so the size of the illusion is
visible rather than asserted.

**Model choice is not a leaderboard.** Candidates are compared with a paired
bootstrap. Models whose gap to the leader has an interval containing zero are
statistically tied, and among tied models the cheapest to operate wins. On this
dataset all four land within noise of each other, which is itself the finding:
the ceiling is set by the available signal, not by model capacity.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import METADATA_PATH, MODEL_PATH, PROCESSED_PATH, REPORT_PATH
from .features import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from .metrics import (
    classification_metrics,
    decile_lift,
    operating_points,
    paired_bootstrap_auc,
    recommend_threshold,
    reliability_curve,
    threshold_sweep,
)

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator

    def _freeze(estimator: Any) -> Any:
        return FrozenEstimator(estimator)

    _CALIBRATION_CV: Any = None
except ImportError:  # pragma: no cover
    def _freeze(estimator: Any) -> Any:
        return estimator

    _CALIBRATION_CV = "prefit"


CANDIDATE_RATIONALE = {
    "logistic_regression": (
        "Linear baseline and the floor for the problem. With only two features "
        "it is also a serious contender: there is no high-order structure left "
        "for a tree to find. Smallest artefact, fastest inference, and the only "
        "candidate whose decision rule can be written down in one line."
    ),
    "random_forest": (
        "Bagged trees. Included to test whether a non-linear decision boundary "
        "in the watch-time / watch-ratio plane beats a linear one. Its cost is "
        "artefact size and inference latency, so it has to win on accuracy."
    ),
    "gradient_boosting": (
        "Sequential residual fitting. Included so that any boosting advantage "
        "can be attributed to the algorithm family rather than to one library. "
        "This is the candidate that previously shipped with F1 = 0.0."
    ),
    "lightgbm": (
        "The industry default for tabular data. Histogram binning and leaf-wise "
        "growth, with native early stopping. Expected winner on a wide feature "
        "set; on two features it has little room to differentiate itself."
    ),
}


def build_preprocessor():
    """Impute then scale. With no categorical features this stays deliberately plain."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def candidate_models(random_state: int) -> dict[str, Any]:
    """Four estimators across three model families.

    ``class_weight`` is deliberately unset. Re-weighting inflates every predicted
    probability, leaves ROC-AUC (a rank statistic) largely unchanged, and wrecks
    calibration. The 27.8% imbalance is handled with an explicit threshold
    instead, which is a decision the consumer can see and change.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=100,
            n_jobs=-1, random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.5,
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, n_jobs=-1,
            random_state=random_state, verbosity=-1,
        ),
    }


def chronological_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Split by time when the log has a timestamp, by session otherwise.

    Production predicts the future from the past, so evaluation should too. If
    no timestamp exists the fallback keeps whole sessions on one side, which at
    least prevents a session leaking across the boundary.
    """
    if "timestamp" in frame.columns and frame["timestamp"].notna().any():
        order = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).argsort().to_numpy()
        n = len(order)
        return (order[: int(n * 0.70)], order[int(n * 0.70): int(n * 0.85)],
                order[int(n * 0.85):], "chronological")
    if "session_id" in frame.columns:
        from sklearn.model_selection import GroupShuffleSplit

        groups = frame["session_id"].to_numpy()
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        rest, test = next(outer.split(frame, groups=groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
        a, b = next(inner.split(frame.iloc[rest], groups=groups[rest]))
        return rest[a], rest[b], test, "group:session_id"
    idx = np.arange(len(frame))
    rest, test = train_test_split(idx, test_size=0.15, random_state=42,
                                  stratify=frame[TARGET_COLUMN])
    train, valid = train_test_split(rest, test_size=0.176, random_state=42,
                                    stratify=frame[TARGET_COLUMN].to_numpy()[rest])
    return train, valid, test, "stratified_random"


def artifact_size_mb(model: Any) -> float:
    with tempfile.NamedTemporaryFile(suffix=".joblib") as handle:
        joblib.dump(model, handle.name)
        return round(Path(handle.name).stat().st_size / 1_048_576, 3)


def measure_latency(model: Any, frame: pd.DataFrame, repeats: int = 20) -> float:
    """Median milliseconds for a single-row prediction — the production unit here."""
    sample = frame.head(1)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(sample)
        timings.append((time.perf_counter() - started) * 1000)
    return round(float(np.median(timings)), 3)


def train(
    input_path: Path = PROCESSED_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    report_path: Path = REPORT_PATH,
    max_rows: int | None = None,
    random_state: int = 42,
    make_plots: bool = True,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path, nrows=max_rows)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(frame.columns)
    if missing:
        raise ValueError(f"the training table is missing columns: {', '.join(sorted(missing))}")
    if len(frame) < 500:
        raise ValueError("too few training rows; at least 500 are required")

    train_idx, valid_idx, test_idx, strategy = chronological_split(frame)
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = x.iloc[valid_idx], y.iloc[valid_idx]
    x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
    print(f"split={strategy} train={len(x_train):,} valid={len(x_valid):,} "
          f"test={len(x_test):,} positive_rate={y.mean():.4f}", flush=True)
    print(f"features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}\n", flush=True)

    candidates: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidate_models(random_state).items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        valid_prob = pipeline.predict_proba(x_valid)[:, 1]
        candidates[name] = {
            "rationale": CANDIDATE_RATIONALE.get(name, ""),
            "fit_seconds": round(fit_seconds, 2),
            "artifact_mb": artifact_size_mb(pipeline),
            "latency_ms": measure_latency(pipeline, x_test),
            "validation": classification_metrics(y_valid, valid_prob),
            "test": classification_metrics(y_test, pipeline.predict_proba(x_test)[:, 1]),
        }
        fitted[name] = pipeline
        v = candidates[name]["validation"]
        print(f"{name:<20} valid ROC-AUC={v['roc_auc']:.4f} PR-AUC={v['pr_auc']:.4f} "
              f"Brier={v['brier']:.4f} F1@0.5={v['f1']:.4f} "
              f"fit={fit_seconds:.1f}s size={candidates[name]['artifact_mb']}MB", flush=True)

    degenerate = [n for n, info in candidates.items() if info["validation"]["f1"] == 0.0]
    if degenerate:
        print(f"\n[warn] degenerate at threshold 0.5 (predicts one class for every row): "
              f"{degenerate} — this is a threshold problem, not a model problem", flush=True)

    # ---- selection: cost veto, tie test, cheapest of the tied ---------------
    LATENCY_BUDGET_MS, SIZE_BUDGET_MB = 50.0, 100.0
    eligible = {n: i for n, i in candidates.items()
                if i["latency_ms"] <= LATENCY_BUDGET_MS and i["artifact_mb"] <= SIZE_BUDGET_MB}
    disqualified = {n: f"latency={candidates[n]['latency_ms']}ms / "
                       f"size={candidates[n]['artifact_mb']}MB exceeds the budget"
                    for n in candidates if n not in eligible}
    pool = eligible or candidates
    significance = paired_bootstrap_auc(
        y_valid.to_numpy(), {n: fitted[n].predict_proba(x_valid)[:, 1] for n in pool}
    )
    tied = [n for n in pool if significance[n]["statistically_tied_with_best"]]
    leader = max(pool, key=lambda n: pool[n]["validation"]["roc_auc"])
    best_name = min(tied, key=lambda n: (candidates[n]["artifact_mb"], candidates[n]["latency_ms"]))
    best_pipeline = fitted[best_name]
    print(f"\nhighest AUC: {leader} ({significance[leader]['auc']:.4f})")
    print(f"statistically tied with it: {sorted(tied)}")
    print(f"cheapest among the tied set -> winner: {best_name}", flush=True)

    # ---- how much would a random split have flattered us? -------------------
    random_train, random_test = train_test_split(
        np.arange(len(frame)), test_size=0.15, random_state=random_state, stratify=y
    )
    leak_pipeline = Pipeline([("preprocessor", build_preprocessor()),
                              ("model", candidate_models(random_state)[best_name])])
    leak_pipeline.fit(x.iloc[random_train], y.iloc[random_train])
    random_auc = float(roc_auc_score(y.iloc[random_test],
                                     leak_pipeline.predict_proba(x.iloc[random_test])[:, 1]))

    # ---- calibration gate ---------------------------------------------------
    calibrator = CalibratedClassifierCV(_freeze(best_pipeline), method="isotonic",
                                        cv=_CALIBRATION_CV)
    calibrator.fit(x_valid, y_valid)
    raw_prob = best_pipeline.predict_proba(x_test)[:, 1]
    cal_prob = calibrator.predict_proba(x_test)[:, 1]
    raw_metrics = classification_metrics(y_test, raw_prob)
    cal_metrics = classification_metrics(y_test, cal_prob)
    helped = cal_metrics["brier"] <= raw_metrics["brier"]
    shipped, shipped_prob = (calibrator, cal_prob) if helped else (best_pipeline, raw_prob)
    print(f"\ncalibration: Brier {raw_metrics['brier']:.5f} -> {cal_metrics['brier']:.5f}, "
          f"ECE {raw_metrics['ece']:.5f} -> {cal_metrics['ece']:.5f}; "
          f"{'shipping the calibrated model' if helped else 'keeping the uncalibrated model'}",
          flush=True)

    sweep = threshold_sweep(y_test, shipped_prob)
    best_row = sweep.loc[sweep["f1"].idxmax()]
    points = operating_points(y_test, shipped_prob)
    recommendation = recommend_threshold(y_test, shipped_prob)
    f1_degenerate = bool(best_row["flagged_rate"] > 0.95)
    lift = decile_lift(y_test, shipped_prob)
    predicted, observed, counts = reliability_curve(y_test, shipped_prob)
    raw_pred, raw_obs, _ = reliability_curve(y_test, raw_prob)
    final = cal_metrics if helped else raw_metrics

    print(f"\nmax predicted probability: {shipped_prob.max():.4f} — any threshold above this "
          f"flags nothing, which is why F1 at 0.50 is {final['f1']:.4f}", flush=True)
    if f1_degenerate:
        print(f"[warn] F1 is maximised at threshold {best_row['threshold']:.2f}, which flags "
              f"{best_row['flagged_rate']:.0%} of traffic: on this signal level F1 optimisation "
              f"degenerates to blanket treatment and is not usable guidance", flush=True)
    print("\noperating points (threshold chosen by budget, not by F1):", flush=True)
    for row in points.itertuples():
        print(f"  flag {row.actual_flag_rate:5.1%} of traffic -> threshold {row.threshold:.4f}  "
              f"precision {row.precision:.4f}  recall {row.recall:.4f}  lift {row.lift:.2f}x",
              flush=True)
    print(f"recommended: threshold {recommendation['threshold']:.4f} "
          f"({recommendation['flag_rate']:.1%} of traffic, {recommendation['lift']:.2f}x lift)",
          flush=True)
    print(f"chronological test ROC-AUC={final['roc_auc']:.4f} vs "
          f"random-split ROC-AUC={random_auc:.4f} "
          f"(optimism from a random split: {random_auc - final['roc_auc']:+.4f})", flush=True)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(shipped, model_path)

    metadata = {
        "model_name": best_name + ("+isotonic" if helped else ""),
        "base_estimator": best_name,
        "calibrated": bool(helped),
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "rows_total": int(len(frame)),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "test_rows": int(len(x_test)),
        "split_strategy": strategy,
        "positive_rate": round(float(y.mean()), 6),
        "target_definition": "liked OR shared OR commented OR followed_creator OR replayed",
        "selection_metric": "validation roc_auc with a paired-bootstrap tie test, then cost",
        "selection_rationale": CANDIDATE_RATIONALE.get(best_name, ""),
        "degenerate_at_default_threshold": degenerate,
        "recommended_threshold": round(float(recommendation["threshold"]), 6),
        "recommended_threshold_rule": recommendation["rule"],
        "recommended_threshold_lift": recommendation["lift"],
        "recommended_threshold_flag_rate": recommendation["flag_rate"],
        "recommended_threshold_precision": recommendation["precision"],
        "f1_optimisation_degenerate": f1_degenerate,
        "max_predicted_probability": round(float(shipped_prob.max()), 6),
        "features": FEATURE_COLUMNS,
        "feature_selection": "see reports/feature_selection.json and docs/FEATURE_SELECTION.md",
        "metrics": {n: i["validation"] for n, i in candidates.items()},
        "test_metrics": final,
        "random_split_roc_auc": round(random_auc, 6),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {"rows": int(len(frame)), "positive_rate": round(float(y.mean()), 6),
                    "split_strategy": strategy, "train_rows": int(len(x_train)),
                    "valid_rows": int(len(x_valid)), "test_rows": int(len(x_test))},
        "features": FEATURE_COLUMNS,
        "candidates": candidates,
        "selection": {"winner": best_name, "auc_leader": leader, "statistically_tied": tied,
                      "significance": significance, "disqualified": disqualified,
                      "rule": "1) cost veto 2) paired-bootstrap tie test 3) cheapest of the tied"},
        "split_optimism": {"chronological_test_roc_auc": round(final["roc_auc"], 6),
                           "random_split_roc_auc": round(random_auc, 6),
                           "difference": round(random_auc - final["roc_auc"], 6)},
        "calibration": {"applied": bool(helped), "raw_test": raw_metrics,
                        "calibrated_test": cal_metrics,
                        "reliability_calibrated": {"predicted": predicted.tolist(),
                                                   "observed": observed.tolist(),
                                                   "counts": counts.tolist()},
                        "reliability_raw": {"predicted": raw_pred.tolist(),
                                            "observed": raw_obs.tolist()}},
        "threshold_sweep": sweep.to_dict(orient="list"),
        "operating_points": points.to_dict(orient="records"),
        "recommendation": recommendation,
        "f1_optimisation_degenerate": f1_degenerate,
        "max_predicted_probability": float(shipped_prob.max()),
        "recommended_threshold": float(recommendation["threshold"]),
        "decile_lift": lift.to_dict(orient="records"),
        "degenerate_at_default_threshold": degenerate,
        "roc_pr_curves": _curve_points(fitted, x_test, y_test),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"\nselected model: {metadata['model_name']}; saved to {model_path}")
    print(f"training report: {report_path}", flush=True)

    if make_plots:
        from .plots import render_all

        rendered = render_all(report)
        print(f"rendered {len(rendered)} figures: {', '.join(p.name for p in rendered)}",
              flush=True)
    return metadata


def _curve_points(fitted, x_test, y_test):
    from sklearn.metrics import precision_recall_curve, roc_curve

    points = {}
    for name, pipeline in fitted.items():
        probability = pipeline.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        sr, sp = max(len(fpr) // 400, 1), max(len(precision) // 400, 1)
        points[name] = {"fpr": fpr[::sr].tolist(), "tpr": tpr[::sr].tolist(),
                        "precision": precision[::sp].tolist(), "recall": recall[::sp].tolist()}
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, compare and select the model")
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=METADATA_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument("--max-rows", type=int, default=None, help="smoke test only")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    train(args.input, args.model_output, args.metadata_output, args.report_output,
          args.max_rows, args.random_state, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
