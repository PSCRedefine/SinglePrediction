"""Feature contract shared by offline training and online prediction.

The feature set here is small on purpose. It is the output of
``scripts/feature_selection.py``, which screens every candidate a live request
could supply and keeps only those whose predictive contribution is
distinguishable from noise. On this dataset that is two features out of
twenty-five, and the pruned model scores *higher* than the full one — dropping
twenty-three noise columns removed variance without removing signal.

See docs/FEATURE_SELECTION.md for the measurements.

One distinction matters and is easy to miss: the **API contract** and the
**feature set** are different things. ``hour_of_day`` is part of the request
because the specification defines it as an input, and the service accepts,
validates and echoes it. It is not a model feature, because measurement says it
carries no signal (AUC 0.5010, interval [0.5000, 0.5034]). Accepting a field and
training on it are separate decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The model's inputs
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["watch_time_seconds", "watch_ratio"]
CATEGORICAL_FEATURES: list[str] = []
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "target_engaged"
ID_COLUMNS = ["interaction_id", "user_id", "video_id"]

# Carried into the training table but never used as features. ``session_id``
# keeps a viewing session on one side of a split; ``timestamp`` enables the
# chronological split that a production model actually needs.
SPLIT_COLUMNS = ["session_id", "timestamp"]

# The five explicit actions that define the label.
ACTION_COLUMNS = ["liked", "shared", "commented", "followed_creator", "replayed"]

# Computed from the outcome of the interaction being predicted. Training on any
# of these is label leakage: they do not exist when a live request arrives.
# ``engagement_score`` alone scores ROC-AUC 0.836 — including it would produce an
# impressive offline number and a worthless service.
LEAKY_COLUMNS = [*ACTION_COLUMNS, "engagement_score", "skipped_quickly", "watch_percentage"]

# Observed during the interaction, so unavailable at scoring time.
UNSERVABLE_COLUMNS = [
    "scroll_velocity", "sound_on", "time_since_last_interaction",
    "session_position", "recommendation_source", "app_version",
]

MAX_WATCH_TIME_SECONDS = 3600.0
USER_ID_PATTERN = re.compile(r"user_[A-Za-z0-9_-]+")
VIDEO_ID_PATTERN = re.compile(r"video_[A-Za-z0-9_-]+")


def validate_ids(user_id: str, video_id: str) -> None:
    """Validate identifier shape without assuming a fixed digit count."""
    if not USER_ID_PATTERN.fullmatch(str(user_id)):
        raise ValueError(
            "user_id must start with 'user_' and contain only letters, digits, "
            "underscores or hyphens"
        )
    if not VIDEO_ID_PATTERN.fullmatch(str(video_id)):
        raise ValueError(
            "video_id must start with 'video_' and contain only letters, digits, "
            "underscores or hyphens"
        )


def watch_ratio(watch_seconds, duration_seconds) -> pd.Series:
    """Fraction of the video watched, clipped to [0, 1].

    A zero-length video would divide by zero. Those rows get ratio 0 rather than
    being dropped, so one corrupt video row degrades a single feature instead of
    failing the request.
    """
    duration = pd.to_numeric(duration_seconds, errors="coerce").replace(0, np.nan)
    ratio = pd.to_numeric(watch_seconds, errors="coerce") / duration
    return ratio.fillna(0.0).clip(0.0, 1.0)


def _required_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")


def build_processed_table(
    interactions: pd.DataFrame, videos: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the leakage-safe training table from the raw interaction log.

    ``watch_ratio`` is computed from the duration recorded **on the interaction
    row**, not from the current ``videos.csv``. The two disagree on 90% of rows,
    and using the interaction row's own duration measurably improves the feature
    (0.579 against 0.558 univariate AUC). Serving falls back to the video
    snapshot because that is all a live request can reach; the discrepancy is
    documented in docs/FEATURE_SELECTION.md rather than hidden.
    """
    _required_columns(
        interactions,
        set(ID_COLUMNS) | {"watch_time_seconds"} | set(ACTION_COLUMNS),
        "interactions.csv",
    )
    frame = interactions.reset_index(drop=True)
    result = frame[ID_COLUMNS].copy()
    for column in SPLIT_COLUMNS:
        if column in frame.columns:
            result[column] = frame[column]

    result["watch_time_seconds"] = pd.to_numeric(
        frame["watch_time_seconds"], errors="coerce"
    ).fillna(0.0)

    if "video_duration_seconds" in frame.columns:
        duration = frame["video_duration_seconds"]
    elif videos is not None:
        duration = frame[["video_id"]].merge(
            videos[["video_id", "duration_seconds"]], on="video_id", how="left"
        )["duration_seconds"]
    else:
        raise ValueError("no video duration available; pass videos.csv or use the full log")
    result["watch_ratio"] = watch_ratio(result["watch_time_seconds"], duration)

    result[TARGET_COLUMN] = frame[ACTION_COLUMNS].astype(bool).any(axis=1).astype("int8")

    present_split = [c for c in SPLIT_COLUMNS if c in result.columns]
    return result[ID_COLUMNS + present_split + FEATURE_COLUMNS + [TARGET_COLUMN]]


@dataclass
class FeatureStore:
    """Identifier resolution and online feature construction.

    The store exists for two reasons even though the model uses only two
    features: an unknown identifier must produce a 404 rather than a prediction,
    and ``watch_ratio`` needs the video's duration, which a request does not
    carry.
    """

    users: pd.DataFrame
    videos: pd.DataFrame
    user_ids: set = field(init=False, repr=False)
    video_ids: set = field(init=False, repr=False)
    durations: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.user_ids = set(self.users["user_id"].astype(str))
        self.video_ids = set(self.videos["video_id"].astype(str))
        self.durations = dict(
            zip(
                self.videos["video_id"].astype(str),
                pd.to_numeric(self.videos["duration_seconds"], errors="coerce").fillna(0.0),
                strict=True,
            )
        )

    @classmethod
    def from_csv(cls, users_path: str, videos_path: str) -> "FeatureStore":
        users = pd.read_csv(users_path, usecols=["user_id"])
        videos = pd.read_csv(videos_path, usecols=["video_id", "duration_seconds"])
        _required_columns(users, {"user_id"}, "users.csv")
        _required_columns(videos, {"video_id", "duration_seconds"}, "videos.csv")
        return cls(users=users, videos=videos)

    def build_one(self, user_id: str, video_id: str, watch_time: float) -> pd.DataFrame:
        """Build the single-row feature frame for one request.

        Raises ``ValueError`` for a malformed identifier and ``KeyError`` for one
        that is well formed but absent, so the API can map them to 422 and 404
        respectively.
        """
        validate_ids(user_id, video_id)
        user_id, video_id = str(user_id), str(video_id)
        if user_id not in self.user_ids:
            raise KeyError(f"unknown user_id: {user_id}")
        if video_id not in self.video_ids:
            raise KeyError(f"unknown video_id: {video_id}")
        seconds = float(watch_time)
        if not 0 <= seconds <= MAX_WATCH_TIME_SECONDS:
            raise ValueError(f"watch_time must be between 0 and {MAX_WATCH_TIME_SECONDS:.0f}")
        duration = float(self.durations.get(video_id, 0.0))
        ratio = min(max(seconds / duration, 0.0), 1.0) if duration else 0.0
        return pd.DataFrame(
            [{"watch_time_seconds": seconds, "watch_ratio": ratio}], columns=FEATURE_COLUMNS
        )
