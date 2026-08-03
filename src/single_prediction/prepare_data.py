"""CLI for producing the model-ready feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import INTERACTIONS_PATH, PROCESSED_PATH, USERS_PATH, VIDEOS_PATH
from .features import FEATURE_COLUMNS, TARGET_COLUMN, build_processed_table


def _validate_processed(frame: pd.DataFrame) -> None:
    """Validate the final contract, using Pandera for field-level checks."""
    import pandera.pandas as pa

    checks = {
        "watch_time_seconds": pa.Column(float, pa.Check.ge(0), coerce=True),
        "hour_of_day": pa.Column(int, [pa.Check.ge(0), pa.Check.le(23)], coerce=True),
        "watch_ratio": pa.Column(float, [pa.Check.ge(0), pa.Check.le(1)], coerce=True),
        TARGET_COLUMN: pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    }
    pa.DataFrameSchema(checks, strict=False).validate(frame, lazy=True)
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("处理后的特征表不能包含缺失值")


def prepare(
    interactions_path: Path = INTERACTIONS_PATH,
    users_path: Path = USERS_PATH,
    videos_path: Path = VIDEOS_PATH,
    output_path: Path = PROCESSED_PATH,
    max_rows: int | None = None,
) -> pd.DataFrame:
    interactions = pd.read_csv(interactions_path, nrows=max_rows)
    users = pd.read_csv(users_path)
    videos = pd.read_csv(videos_path)
    result = build_processed_table(interactions, users, videos)
    _validate_processed(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成训练用 processed_interactions.csv")
    parser.add_argument("--interactions", type=Path, default=INTERACTIONS_PATH)
    parser.add_argument("--users", type=Path, default=USERS_PATH)
    parser.add_argument("--videos", type=Path, default=VIDEOS_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于快速烟雾验证")
    args = parser.parse_args()
    frame = prepare(args.interactions, args.users, args.videos, args.output, args.max_rows)
    print(
        f"已生成 {args.output}: {len(frame):,} 行, "
        f"正例率={frame[TARGET_COLUMN].mean():.3%}"
    )


if __name__ == "__main__":
    main()
