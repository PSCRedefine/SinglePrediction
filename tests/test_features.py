"""The feature contract: leakage, watch ratio, identifier handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from single_prediction.features import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    TARGET_COLUMN,
    UNSERVABLE_COLUMNS,
    FeatureStore,
    build_processed_table,
    validate_ids,
    watch_ratio,
)


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "interaction_id": "int_1", "user_id": "user_000001",
                "video_id": "video_0000001", "session_id": "s1",
                "timestamp": "2025-07-14T22:06:00Z",
                "watch_time_seconds": 15, "video_duration_seconds": 30,
                "liked": False, "shared": False, "commented": True,
                "followed_creator": False, "replayed": False,
            },
            {
                "interaction_id": "int_2", "user_id": "user_000002",
                "video_id": "video_0000002", "session_id": "s1",
                "timestamp": "2025-07-14T22:07:00Z",
                "watch_time_seconds": 0, "video_duration_seconds": 60,
                "liked": False, "shared": False, "commented": False,
                "followed_creator": False, "replayed": False,
            },
        ]
    )


def test_target_is_any_active_action():
    result = build_processed_table(_interactions())
    assert result.loc[0, TARGET_COLUMN] == 1
    assert result.loc[1, TARGET_COLUMN] == 0


def test_no_leaky_or_unservable_column_can_become_a_feature():
    """The two exclusion lists must never intersect the feature set."""
    for column in LEAKY_COLUMNS + UNSERVABLE_COLUMNS:
        assert column not in FEATURE_COLUMNS


def test_processed_table_contains_only_the_selected_features():
    result = build_processed_table(_interactions())
    for column in LEAKY_COLUMNS:
        assert column not in result.columns
    assert set(FEATURE_COLUMNS).issubset(result.columns)


def test_split_columns_survive_for_the_chronological_split():
    result = build_processed_table(_interactions())
    assert "timestamp" in result.columns
    assert "session_id" in result.columns


def test_watch_ratio_is_clipped_and_survives_zero_duration():
    ratio = watch_ratio(pd.Series([15.0, 999.0, 10.0]), pd.Series([30.0, 30.0, 0.0]))
    assert ratio.tolist() == [0.5, 1.0, 0.0]


def test_watch_ratio_uses_the_interaction_row_duration_when_present():
    """The interaction's own duration disagrees with videos.csv on 90% of rows."""
    interactions = _interactions()
    videos = pd.DataFrame({"video_id": ["video_0000001"], "duration_seconds": [300]})
    result = build_processed_table(interactions, videos)
    # 15 / 30 from the interaction row, not 15 / 300 from videos.csv
    assert result.loc[0, "watch_ratio"] == pytest.approx(0.5)


def test_identifier_validation():
    validate_ids("user_000001", "video_0000001")
    with pytest.raises(ValueError):
        validate_ids("nope", "video_0000001")
    with pytest.raises(ValueError):
        validate_ids("user_000001", "nope")


def test_feature_store_rejects_unknown_and_malformed_ids(feature_csvs):
    store = FeatureStore.from_csv(*map(str, feature_csvs))
    frame = store.build_one("user_000001", "video_0000001", 15.0)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.loc[0, "watch_ratio"] == pytest.approx(0.5)

    with pytest.raises(KeyError):
        store.build_one("user_999999", "video_0000001", 10.0)
    with pytest.raises(KeyError):
        store.build_one("user_000001", "video_9999999", 10.0)
    with pytest.raises(ValueError):
        store.build_one("bad", "video_0000001", 10.0)
    with pytest.raises(ValueError):
        store.build_one("user_000001", "video_0000001", -1.0)


def test_feature_store_handles_zero_duration(feature_csvs):
    store = FeatureStore.from_csv(*map(str, feature_csvs))
    frame = store.build_one("user_000001", "video_0000005", 30.0)
    assert frame.loc[0, "watch_ratio"] == 0.0
