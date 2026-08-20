"""Shared fixtures: a miniature but real model and identifier store on disk.

The API tests do not mock the model. They fit a real pipeline through the real
preprocessing, dump it with joblib and load it through the real code path, so a
dtype or serialisation regression fails the suite. A mock would pass while the
service was broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from single_prediction.features import FEATURE_COLUMNS
from single_prediction.train import build_preprocessor

USER_IDS = [f"user_{i:06d}" for i in range(1, 6)]
VIDEO_IDS = [f"video_{i:07d}" for i in range(1, 6)]


@pytest.fixture
def feature_csvs(tmp_path: Path) -> tuple[Path, Path]:
    users_path = tmp_path / "users.csv"
    videos_path = tmp_path / "videos.csv"
    pd.DataFrame({"user_id": USER_IDS}).to_csv(users_path, index=False)
    pd.DataFrame(
        {"video_id": VIDEO_IDS, "duration_seconds": [30, 60, 90, 120, 0]}
    ).to_csv(videos_path, index=False)
    return users_path, videos_path


@pytest.fixture
def trained_artifacts(tmp_path: Path, feature_csvs: tuple[Path, Path]) -> dict[str, Path]:
    users_path, videos_path = feature_csvs
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "watch_time_seconds": rng.uniform(0, 120, 200),
            "watch_ratio": rng.uniform(0, 1, 200),
        }
    )[FEATURE_COLUMNS]
    target = (frame["watch_ratio"] + rng.normal(0, 0.2, 200) > 0.6).astype(int)

    pipeline = Pipeline(
        [("preprocessor", build_preprocessor()), ("model", LogisticRegression(max_iter=200))]
    )
    pipeline.fit(frame, target)

    model_path = tmp_path / "best_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "test_model",
                "model_version": "test",
                "split_strategy": "chronological",
                "recommended_threshold": 0.35,
                "features": FEATURE_COLUMNS,
                "metrics": {},
                "test_metrics": {"roc_auc": 0.58},
            }
        ),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "metadata": metadata_path,
        "users": users_path,
        "videos": videos_path,
    }
