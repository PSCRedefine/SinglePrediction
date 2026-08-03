import pandas as pd
import pytest

from single_prediction.features import FEATURE_COLUMNS, TARGET_COLUMN, FeatureStore, build_processed_table


def test_build_processed_table_and_target():
    interactions = pd.DataFrame(
        [
            {
                "interaction_id": "i1",
                "user_id": "user_1",
                "video_id": "video_1",
                "watch_time_seconds": 10,
                "hour_of_day": 12,
                "liked": False,
                "shared": False,
                "commented": True,
                "followed_creator": False,
                "replayed": False,
            }
        ]
    )
    users = pd.DataFrame(
        [
            {
                "user_id": "user_1",
                "age": 20,
                "gender": "F",
                "country": "US",
                "language": "en",
                "account_age_days": 100,
                "subscriber_count": 10,
                "avg_session_length_minutes": 5,
                "total_video_views": 20,
                "total_likes_given": 3,
                "is_premium": False,
                "primary_device": "mobile",
                "primary_platform": "ios",
            }
        ]
    )
    videos = pd.DataFrame(
        [
            {
                "video_id": "video_1",
                "category": "News",
                "duration_seconds": 20,
                "days_since_upload": 3,
                "view_count": 100,
                "like_count": 10,
                "comment_count": 2,
                "share_count": 1,
                "has_music": True,
                "has_text_overlay": False,
                "has_effects": False,
                "has_voiceover": True,
                "video_quality": "1080p",
                "trending_score": 0.5,
                "avg_watch_time_seconds": 12,
                "retention_rate": 0.6,
                "engagement_rate": 0.1,
                "language": "en",
            }
        ]
    )
    result = build_processed_table(interactions, users, videos)
    assert result.loc[0, TARGET_COLUMN] == 1
    assert result.loc[0, "watch_ratio"] == pytest.approx(0.5)
    assert list(result.columns[-1:]) == [TARGET_COLUMN]
    assert not result[FEATURE_COLUMNS].isna().any().any()


def test_feature_store_rejects_unknown_id(tmp_path):
    users = pd.DataFrame(columns=["user_id", "age", "gender", "country", "language", "account_age_days", "subscriber_count", "avg_session_length_minutes", "total_video_views", "total_likes_given", "is_premium", "primary_device", "primary_platform"])
    videos = pd.DataFrame(columns=["video_id", "category", "duration_seconds", "days_since_upload", "view_count", "like_count", "comment_count", "share_count", "has_music", "has_text_overlay", "has_effects", "has_voiceover", "video_quality", "trending_score", "avg_watch_time_seconds", "retention_rate", "engagement_rate", "language"])
    store = FeatureStore(users.set_index("user_id", drop=False), videos.set_index("video_id", drop=False))
    with pytest.raises(KeyError):
        store.build_one("user_1", "video_1", 10, 12)
