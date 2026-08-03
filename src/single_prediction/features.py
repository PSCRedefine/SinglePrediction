"""Shared feature contract used by offline processing and online prediction."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "watch_time_seconds",
    "hour_of_day",
    "watch_ratio",
    "user_age",
    "user_account_age_days",
    "user_subscriber_count",
    "user_avg_session_length_minutes",
    "user_total_video_views",
    "user_total_likes_given",
    "video_duration_seconds",
    "video_days_since_upload",
    "video_view_count",
    "video_like_count",
    "video_comment_count",
    "video_share_count",
    "video_trending_score",
    "video_avg_watch_time_seconds",
    "video_retention_rate",
    "video_engagement_rate",
]

CATEGORICAL_FEATURES = [
    "user_gender",
    "user_country",
    "user_language",
    "user_is_premium",
    "user_primary_device",
    "user_primary_platform",
    "video_category",
    "video_language",
    "video_quality",
    "video_has_music",
    "video_has_text_overlay",
    "video_has_effects",
    "video_has_voiceover",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target_engaged"
ID_COLUMNS = ["interaction_id", "user_id", "video_id"]

USER_RENAME = {
    "age": "user_age",
    "gender": "user_gender",
    "country": "user_country",
    "language": "user_language",
    "account_age_days": "user_account_age_days",
    "subscriber_count": "user_subscriber_count",
    "avg_session_length_minutes": "user_avg_session_length_minutes",
    "total_video_views": "user_total_video_views",
    "total_likes_given": "user_total_likes_given",
    "is_premium": "user_is_premium",
    "primary_device": "user_primary_device",
    "primary_platform": "user_primary_platform",
}

VIDEO_RENAME = {
    "category": "video_category",
    "duration_seconds": "video_duration_seconds",
    "days_since_upload": "video_days_since_upload",
    "view_count": "video_view_count",
    "like_count": "video_like_count",
    "comment_count": "video_comment_count",
    "share_count": "video_share_count",
    "has_music": "video_has_music",
    "has_text_overlay": "video_has_text_overlay",
    "has_effects": "video_has_effects",
    "has_voiceover": "video_has_voiceover",
    "video_quality": "video_quality",
    "trending_score": "video_trending_score",
    "avg_watch_time_seconds": "video_avg_watch_time_seconds",
    "retention_rate": "video_retention_rate",
    "engagement_rate": "video_engagement_rate",
    "language": "video_language",
}

ACTION_COLUMNS = ["liked", "shared", "commented", "followed_creator", "replayed"]


def _required_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} 缺少字段: {', '.join(sorted(missing))}")


def validate_ids(user_id: str, video_id: str) -> None:
    """Validate the public identifier shape without assuming a fixed digit count."""
    if not re.fullmatch(r"user_[A-Za-z0-9_-]+", user_id):
        raise ValueError("user_id 必须以 'user_' 开头，且只含字母、数字、下划线或连字符")
    if not re.fullmatch(r"video_[A-Za-z0-9_-]+", video_id):
        raise ValueError("video_id 必须以 'video_' 开头，且只含字母、数字、下划线或连字符")


def build_processed_table(
    interactions: pd.DataFrame, users: pd.DataFrame, videos: pd.DataFrame
) -> pd.DataFrame:
    """Join raw snapshots and build a leakage-safe binary training table.

    The target is 1 when any explicit active action occurred. The action columns
    and the provided engagement_score are deliberately excluded from features.
    """
    _required_columns(
        interactions,
        set(ID_COLUMNS + ["watch_time_seconds", "hour_of_day"] + ACTION_COLUMNS),
        "interactions.csv",
    )
    _required_columns(users, {"user_id", *USER_RENAME}, "users.csv")
    _required_columns(videos, {"video_id", *VIDEO_RENAME}, "videos.csv")

    base = interactions[ID_COLUMNS + ["watch_time_seconds", "hour_of_day"] + ACTION_COLUMNS].copy()
    base[TARGET_COLUMN] = base[ACTION_COLUMNS].astype(bool).any(axis=1).astype("int8")
    base = base.drop(columns=ACTION_COLUMNS)

    user_features = users[["user_id", *USER_RENAME]].rename(columns=USER_RENAME)
    video_features = videos[["video_id", *VIDEO_RENAME]].rename(columns=VIDEO_RENAME)
    result = base.merge(user_features, on="user_id", how="left", validate="many_to_one")
    result = result.merge(video_features, on="video_id", how="left", validate="many_to_one")
    duration = result["video_duration_seconds"].replace(0, np.nan)
    result["watch_ratio"] = (result["watch_time_seconds"] / duration).fillna(0).clip(0, 1)
    if result[FEATURE_COLUMNS].isna().any().any():
        missing_rows = int(result[FEATURE_COLUMNS].isna().any(axis=1).sum())
        raise ValueError(f"连接用户/视频画像后有 {missing_rows} 行缺少特征")
    return result[ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]]


@dataclass
class FeatureStore:
    """In-memory user/video snapshot lookup for low-latency local inference."""

    users: pd.DataFrame
    videos: pd.DataFrame

    @classmethod
    def from_csv(cls, users_path: str, videos_path: str) -> "FeatureStore":
        users = pd.read_csv(users_path).set_index("user_id", drop=False)
        videos = pd.read_csv(videos_path).set_index("video_id", drop=False)
        _required_columns(users, {"user_id", *USER_RENAME}, "users.csv")
        _required_columns(videos, {"video_id", *VIDEO_RENAME}, "videos.csv")
        return cls(users=users, videos=videos)

    def build_one(
        self, user_id: str, video_id: str, watch_time: float, hour_of_day: int
    ) -> pd.DataFrame:
        validate_ids(user_id, video_id)
        if user_id not in self.users.index:
            raise KeyError(f"未知 user_id: {user_id}")
        if video_id not in self.videos.index:
            raise KeyError(f"未知 video_id: {video_id}")

        user = self.users.loc[user_id]
        video = self.videos.loc[video_id]
        row: dict[str, object] = {
            "watch_time_seconds": float(watch_time),
            "hour_of_day": int(hour_of_day),
        }
        row.update({new: user[old] for old, new in USER_RENAME.items()})
        row.update({new: video[old] for old, new in VIDEO_RENAME.items()})
        duration = float(row["video_duration_seconds"])
        row["watch_ratio"] = min(max(float(watch_time) / duration, 0.0), 1.0) if duration else 0.0
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)
