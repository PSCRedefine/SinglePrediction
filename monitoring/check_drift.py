#!/usr/bin/env python3
"""Compare a recent slice of traffic against the training baselines.

Run it against a log of scored requests to answer three questions without
needing labels:

1. Has the *input* distribution moved? (PSI on the two model features)
2. Has the *output* distribution moved? (mean probability, flag rate)
3. Is the model still emitting the number of distinct probabilities it should?

Question 3 is specific to this model: isotonic calibration produces a step
function, and on this dataset it collapses to eight distinct values. A sudden
change in that count means the artefact was swapped, not that traffic changed.

Calibration drift needs labels and therefore a separate, delayed job; the
thresholds for it are in monitoring.yaml.

Usage:
    python monitoring/check_drift.py --live path/to/scored_requests.csv
    python monitoring/check_drift.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES = ["watch_time_seconds", "watch_ratio"]
PSI_WARNING, PSI_CRITICAL = 0.10, 0.25


def population_stability_index(baseline: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples, using baseline deciles as the bin edges.

    A small constant guards against empty bins, which would otherwise send the
    logarithm to infinity and turn a harmless sparse bucket into a false alarm.
    """
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    base_counts = np.histogram(baseline, bins=edges)[0].astype(float)
    live_counts = np.histogram(live, bins=edges)[0].astype(float)
    base_share = np.clip(base_counts / max(base_counts.sum(), 1), 1e-6, None)
    live_share = np.clip(live_counts / max(live_counts.sum(), 1), 1e-6, None)
    return float(np.sum((live_share - base_share) * np.log(live_share / base_share)))


def verdict(psi: float) -> str:
    if psi >= PSI_CRITICAL:
        return "CRITICAL"
    if psi >= PSI_WARNING:
        return "WARNING"
    return "ok"


def compare(baseline: pd.DataFrame, live: pd.DataFrame, metadata: dict) -> dict:
    report: dict = {"features": {}, "output": {}, "status": "ok"}
    worst = "ok"
    for feature in FEATURES:
        if feature not in live.columns:
            continue
        psi = population_stability_index(
            baseline[feature].to_numpy(dtype=float), live[feature].to_numpy(dtype=float)
        )
        state = verdict(psi)
        report["features"][feature] = {
            "psi": round(psi, 5),
            "status": state,
            "baseline_mean": round(float(baseline[feature].mean()), 5),
            "live_mean": round(float(live[feature].mean()), 5),
        }
        if state != "ok":
            worst = "CRITICAL" if state == "CRITICAL" or worst == "CRITICAL" else "WARNING"

    if "probability" in live.columns:
        probability = live["probability"].to_numpy(dtype=float)
        threshold = float(metadata.get("recommended_threshold", 0.5))
        baseline_mean = float(metadata.get("positive_rate", 0.2779))
        shift = abs(float(probability.mean()) - baseline_mean)
        distinct = int(len(np.unique(np.round(probability, 6))))
        report["output"] = {
            "mean_probability": round(float(probability.mean()), 5),
            "baseline_mean_probability": round(baseline_mean, 5),
            "shift": round(shift, 5),
            "flag_rate_at_recommended_threshold": round(float((probability >= threshold).mean()), 5),
            "distinct_values": distinct,
            "status": "WARNING" if shift > 0.05 else "ok",
        }
        if report["output"]["status"] != "ok" and worst == "ok":
            worst = "WARNING"

    report["status"] = worst
    return report


def _self_test() -> dict:
    """Prove the detector actually fires: shift the input and watch PSI move."""
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({
        "watch_time_seconds": rng.gamma(2.0, 16.0, 50_000),
        "watch_ratio": rng.beta(2.0, 2.0, 50_000),
    })
    unchanged = pd.DataFrame({
        "watch_time_seconds": rng.gamma(2.0, 16.0, 10_000),
        "watch_ratio": rng.beta(2.0, 2.0, 10_000),
    })
    shifted = pd.DataFrame({
        "watch_time_seconds": rng.gamma(2.0, 26.0, 10_000),   # users watching longer
        "watch_ratio": rng.beta(3.0, 1.6, 10_000),
    })
    metadata = {"recommended_threshold": 0.3812, "positive_rate": 0.2779}
    quiet = compare(baseline, unchanged, metadata)
    loud = compare(baseline, shifted, metadata)
    print("unchanged traffic ->", quiet["status"],
          {k: v["psi"] for k, v in quiet["features"].items()})
    print("shifted traffic   ->", loud["status"],
          {k: v["psi"] for k, v in loud["features"].items()})
    assert quiet["status"] == "ok", "detector produced a false alarm on unchanged traffic"
    assert loud["status"] != "ok", "detector missed a genuine distribution shift"
    print("self-test passed: quiet traffic stays quiet, shifted traffic alerts")
    return loud


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift check against training baselines")
    parser.add_argument("--baseline", type=Path,
                        default=PROJECT_ROOT / "data" / "processed_interactions.csv")
    parser.add_argument("--live", type=Path, help="CSV of recently scored requests")
    parser.add_argument("--metadata", type=Path,
                        default=PROJECT_ROOT / "models" / "model_metadata.json")
    parser.add_argument("--self-test", action="store_true",
                        help="run the detector against synthetic quiet and shifted traffic")
    args = parser.parse_args()

    if args.self_test or args.live is None:
        _self_test()
        return

    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata.exists() else {}
    baseline = pd.read_csv(args.baseline, usecols=FEATURES)
    live = pd.read_csv(args.live)
    report = compare(baseline, live, metadata)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "CRITICAL":
        raise SystemExit(2)
    if report["status"] == "WARNING":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
