#!/usr/bin/env python3
"""Measure which features carry signal, and prune to the ones that do.

The pruned feature set in this project is not an opinion. It is the output of
this script, which is committed so the decision can be re-derived on new data.

Procedure
---------
1. Restrict the candidate pool to features a live request can actually supply.
   The service receives ``user_id``, ``video_id``, ``watch_time`` and
   ``hour_of_day``. Anything else must be derivable from those, from the request
   clock, or from the user/video snapshot stores the backend can read. A feature
   that is observed only *during* the interaction (scroll velocity, whether the
   user skipped) is unavailable at scoring time and is excluded regardless of
   how predictive it looks offline.
2. Score every candidate univariately with ROC-AUC and a bootstrap interval.
   A feature whose interval includes 0.50 is indistinguishable from noise.
3. Score every candidate multivariately by permutation importance on a held-out
   split, so features that matter only in combination are not missed.
4. Keep a feature when either test clears its threshold. Report everything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
REPORTS = PROJECT_ROOT / "reports"

ACTIONS = ["liked", "shared", "commented", "followed_creator", "replayed"]

# Observed only during or after the interaction. Not available when a live
# request arrives, therefore not candidates however predictive they appear.
NOT_SERVABLE = [
    "watch_percentage",        # derivable from watch_time / duration -> rebuilt as watch_ratio
    "skipped_quickly",
    "scroll_velocity",
    "sound_on",
    "time_since_last_interaction",
    "session_position",
    "recommendation_source",
    "app_version",
    "engagement_score",        # computed from the outcome: label leakage
]

# Supplied by the caller or derived from the request clock.
REQUEST_FEATURES = ["watch_time_seconds", "hour_of_day", "day_of_week", "is_weekend", "watch_ratio"]

# Resolvable from the user snapshot store at request time.
USER_SNAPSHOT = [
    "user_age", "user_gender", "user_country", "user_language",
    "user_account_age_days", "user_subscriber_count", "user_primary_device",
    "user_primary_platform", "user_is_premium",
]

# Resolvable from the video snapshot store at request time.
VIDEO_SNAPSHOT = [
    "video_category", "video_duration_seconds", "video_view_count",
    "video_like_count", "video_comment_count", "video_share_count",
    "video_trending_score", "video_has_music", "video_has_text_overlay",
    "video_has_effects", "video_days_since_upload",
]


def bootstrap_auc(y: np.ndarray, score: np.ndarray, resamples: int = 200, seed: int = 0):
    """Univariate ROC-AUC with a percentile interval.

    The AUC is folded above 0.5 because a feature that is perfectly
    anti-correlated is just as useful as one that is correlated; only the
    distance from 0.5 matters for feature screening.
    """
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, score)
    point = max(point, 1 - point)
    draws = []
    n = len(y)
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        a = roc_auc_score(y[idx], score[idx])
        draws.append(max(a, 1 - a))
    draws = np.array(draws)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def build_frame(rows: int | None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the interaction log and attach the snapshot stores the API would read."""
    interactions = pd.read_csv(DATA / "interactions.csv", nrows=rows)
    users = pd.read_csv(DATA / "users.csv")
    videos = pd.read_csv(DATA / "videos.csv")

    y = interactions[ACTIONS].astype(bool).any(axis=1).astype(int)
    frame = interactions[["user_id", "video_id", "watch_time_seconds", "hour_of_day",
                          "day_of_week", "is_weekend", "session_id", "timestamp"]].copy()

    user_cols = {
        "age": "user_age", "gender": "user_gender", "country": "user_country",
        "language": "user_language", "account_age_days": "user_account_age_days",
        "subscriber_count": "user_subscriber_count", "primary_device": "user_primary_device",
        "primary_platform": "user_primary_platform", "is_premium": "user_is_premium",
    }
    video_cols = {
        "category": "video_category", "duration_seconds": "video_duration_seconds",
        "view_count": "video_view_count", "like_count": "video_like_count",
        "comment_count": "video_comment_count", "share_count": "video_share_count",
        "trending_score": "video_trending_score", "has_music": "video_has_music",
        "has_text_overlay": "video_has_text_overlay", "has_effects": "video_has_effects",
        "days_since_upload": "video_days_since_upload",
    }
    frame = frame.merge(users[["user_id", *user_cols]].rename(columns=user_cols),
                        on="user_id", how="left")
    frame = frame.merge(videos[["video_id", *video_cols]].rename(columns=video_cols),
                        on="video_id", how="left")
    duration = frame["video_duration_seconds"].replace(0, np.nan)
    frame["watch_ratio"] = (frame["watch_time_seconds"] / duration).fillna(0).clip(0, 1)
    return frame, y, interactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure and prune the feature set")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--auc-threshold", type=float, default=0.505,
                        help="a feature must beat this AND have an interval excluding it")
    args = parser.parse_args()

    frame, y, _ = build_frame(args.rows)
    candidates = [c for c in REQUEST_FEATURES + USER_SNAPSHOT + VIDEO_SNAPSHOT if c in frame]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in candidates if c not in numeric]

    print(f"rows={len(frame):,}  positive_rate={y.mean():.4f}  candidates={len(candidates)}")
    print(f"excluded as not servable at request time: {', '.join(NOT_SERVABLE)}\n")

    # ---- univariate screen -------------------------------------------------
    univariate = {}
    yv = y.to_numpy()
    for column in candidates:
        series = frame[column]
        if column in categorical:
            codes = pd.factorize(series.astype(str))[0]
            score = pd.Series(yv).groupby(codes).transform("mean").to_numpy()
        else:
            score = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        point, low, high = bootstrap_auc(yv, score)
        univariate[column] = {"auc": round(point, 5), "ci_low": round(low, 5),
                              "ci_high": round(high, 5),
                              "beats_noise": bool(low > args.auc_threshold)}

    # ---- multivariate screen ----------------------------------------------
    split = int(len(frame) * 0.8)
    x_train, x_valid = frame[candidates].iloc[:split], frame[candidates].iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                           min_frequency=20))]), categorical),
    ])
    model = Pipeline([("pre", pre), ("model", LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        n_jobs=-1, random_state=42, verbosity=-1))])
    model.fit(x_train, y_train)
    baseline = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
    print(f"full-feature validation ROC-AUC: {baseline:.5f}\n")

    importance = permutation_importance(
        model, x_valid, y_valid, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1
    )
    for i, column in enumerate(candidates):
        univariate[column]["permutation_drop"] = round(float(importance.importances_mean[i]), 6)
        univariate[column]["permutation_std"] = round(float(importance.importances_std[i]), 6)
        univariate[column]["matters_in_combination"] = bool(
            importance.importances_mean[i] > 2 * importance.importances_std[i]
            and importance.importances_mean[i] > 0.001
        )

    keep = [c for c, v in univariate.items()
            if v["beats_noise"] or v["matters_in_combination"]]

    ordered = sorted(univariate.items(), key=lambda kv: -kv[1]["auc"])
    print(f"{'feature':32s} {'AUC':>7s} {'95% CI':>18s} {'perm drop':>10s}  verdict")
    for column, v in ordered:
        verdict = "KEEP" if column in keep else "drop (noise)"
        print(f"  {column:30s} {v['auc']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
              f"  {v['permutation_drop']:+.5f}  {verdict}")

    print(f"\nkept {len(keep)} of {len(candidates)} candidates: {keep}")

    # ---- does the pruned model match the full model? -----------------------
    numeric_keep = [c for c in keep if c in numeric]
    categorical_keep = [c for c in keep if c in categorical]
    pre_keep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric_keep),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                           min_frequency=20))]), categorical_keep),
    ])
    pruned = Pipeline([("pre", pre_keep), ("model", LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        n_jobs=-1, random_state=42, verbosity=-1))])
    pruned.fit(x_train[keep], y_train)
    pruned_auc = roc_auc_score(y_valid, pruned.predict_proba(x_valid[keep])[:, 1])
    print(f"\nfull model  ({len(candidates)} features): {baseline:.5f}")
    print(f"pruned model ({len(keep)} features): {pruned_auc:.5f}")
    print(f"difference: {pruned_auc - baseline:+.5f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "feature_selection.json"
    out.write_text(json.dumps({
        "rows": int(len(frame)),
        "positive_rate": round(float(y.mean()), 6),
        "auc_threshold": args.auc_threshold,
        "excluded_not_servable": NOT_SERVABLE,
        "candidates": univariate,
        "kept": keep,
        "full_model_auc": round(float(baseline), 6),
        "pruned_model_auc": round(float(pruned_auc), 6),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
