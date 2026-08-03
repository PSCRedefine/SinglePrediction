"""Paths and environment-backed application settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def env_path(name: str, default: Path) -> Path:
    """Return an absolute path from an environment variable or a project default."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


USERS_PATH = env_path("SINGLE_PREDICTION_USERS_PATH", DATA_DIR / "users.csv")
VIDEOS_PATH = env_path("SINGLE_PREDICTION_VIDEOS_PATH", DATA_DIR / "videos.csv")
INTERACTIONS_PATH = env_path(
    "SINGLE_PREDICTION_INTERACTIONS_PATH", DATA_DIR / "interactions.csv"
)
PROCESSED_PATH = env_path(
    "SINGLE_PREDICTION_PROCESSED_PATH", DATA_DIR / "processed_interactions.csv"
)
MODEL_PATH = env_path(
    "SINGLE_PREDICTION_MODEL_PATH", MODELS_DIR / "best_model.joblib"
)
METADATA_PATH = env_path(
    "SINGLE_PREDICTION_METADATA_PATH", MODELS_DIR / "model_metadata.json"
)
