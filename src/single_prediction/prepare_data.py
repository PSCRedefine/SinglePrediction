"""Offline pipeline: the raw interaction log to one model-ready feature table.

The log is 139 MB and would be larger in production, so it is streamed in chunks
and appended. Peak memory stays flat regardless of file size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import INTERACTIONS_PATH, PROCESSED_PATH, VIDEOS_PATH
from .features import (
    ACTION_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    SPLIT_COLUMNS,
    TARGET_COLUMN,
    build_processed_table,
)

DEFAULT_CHUNKSIZE = 100_000

# Only these columns are read from the log. Everything else is either leaky,
# unavailable at serving time, or measured to be noise; not reading it keeps the
# pass fast and makes the exclusion explicit rather than implicit.
READ_COLUMNS = ID_COLUMNS + SPLIT_COLUMNS + [
    "watch_time_seconds", "video_duration_seconds", *ACTION_COLUMNS
]


def _validate_processed(frame: pd.DataFrame) -> None:
    """Field-level contract check with Pandera.

    Cheap insurance: it catches "watch_time is suddenly a string" the moment the
    data changes, rather than three steps later when a model trains on garbage.
    """
    import pandera.pandas as pa

    schema = pa.DataFrameSchema(
        {
            "watch_time_seconds": pa.Column(float, pa.Check.ge(0), coerce=True),
            "watch_ratio": pa.Column(float, [pa.Check.ge(0), pa.Check.le(1)], coerce=True),
            TARGET_COLUMN: pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        },
        strict=False,
    )
    schema.validate(frame, lazy=True)
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("the processed feature table must not contain missing values")


def prepare(
    interactions_path: Path = INTERACTIONS_PATH,
    videos_path: Path = VIDEOS_PATH,
    output_path: Path = PROCESSED_PATH,
    max_rows: int | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict[str, float]:
    """Stream the log, build features, write the training table."""
    header_columns = pd.read_csv(interactions_path, nrows=0).columns
    usecols = [c for c in READ_COLUMNS if c in header_columns]
    videos = None
    if "video_duration_seconds" not in usecols:
        videos = pd.read_csv(videos_path, usecols=["video_id", "duration_seconds"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rows_written = 0
    positives = 0
    remaining = max_rows
    header = True
    for chunk in pd.read_csv(interactions_path, usecols=usecols, chunksize=chunksize):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        processed = build_processed_table(chunk, videos)
        _validate_processed(processed)
        processed.to_csv(output_path, index=False, mode="a", header=header)
        header = False
        rows_written += len(processed)
        positives += int(processed[TARGET_COLUMN].sum())

    if rows_written == 0:
        raise ValueError("no training rows were produced; check interactions.csv")
    return {
        "rows": rows_written,
        "positive_rate": positives / rows_written,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the model-ready feature table")
    parser.add_argument("--interactions", type=Path, default=INTERACTIONS_PATH)
    parser.add_argument("--videos", type=Path, default=VIDEOS_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--max-rows", type=int, default=None, help="smoke test only")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = parser.parse_args()
    summary = prepare(args.interactions, args.videos, args.output, args.max_rows, args.chunksize)
    print(
        f"wrote {summary['output']}: {int(summary['rows']):,} rows, "
        f"positive rate={summary['positive_rate']:.3%}"
    )


if __name__ == "__main__":
    main()
