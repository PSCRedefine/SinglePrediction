# processed_interactions.csv — Data Preview

The output of `prepare_data.py`, and the only input `train.py` takes. This document is its "thumbnail", so the table can be checked at any time without opening the 47 MB original.

- Full file: `data/processed_interactions.csv` (gitignored; rebuild with `python -m single_prediction.prepare_data`)
- Directly openable sample: [`processed_interactions_sample.csv`](processed_interactions_sample.csv) (first 20 rows)
- Rebuilding takes about 1 minute

## Scale

| Item | Value |
|---|---|
| Rows | 500,000 |
| Columns | 8 |
| File size | 47 MB |
| Missing values | 0 (all 8 columns) |
| Positive rate `target_engaged` | 27.79% |
| Time span | 2025-07-14 → 2025-08-14 (about 1 month) |

## Column structure

8 columns = 3 IDs + 2 split keys + 2 features + 1 label. The original `interactions.csv` has 43 columns; the rest either leak the label or are unavailable in production, and are deliberately dropped by `features.py` — the reasoning is in [FEATURE_SELECTION.md](FEATURE_SELECTION.md).

| # | Column | Type | Role | Unique values | Notes |
|---|---|---|---|---|---|
| 1 | `interaction_id` | object | ID | 500,000 | Unique per row |
| 2 | `user_id` | object | ID | 25,000 | |
| 3 | `video_id` | object | ID | 35,000 | |
| 4 | `session_id` | object | Split key | 54,403 | Fallback grouping key when timestamp is absent |
| 5 | `timestamp` | object | Split key | 499,915 | ISO-8601 UTC, **the temporal split depends on this column** |
| 6 | `watch_time_seconds` | int64 | **Feature** | 61 | 0–60 seconds |
| 7 | `watch_ratio` | float64 | **Feature** | 231 | 0–1, watch time / video duration |
| 8 | `target_engaged` | int64 | **Label** | 2 | liked OR shared OR commented OR followed_creator OR replayed |

> A missing `timestamp` makes `train.py` silently degrade to a random split rather than a temporal one — both split key columns must be present.

## First 8 rows

| interaction_id | user_id | video_id | session_id | timestamp | watch_time_seconds | watch_ratio | target_engaged |
|---|---|---|---|---|---|---|---|
| int_00000000 | user_011733 | video_0001369 | session_00000001 | 2025-07-29T06:15:15.088Z | 37 | 0.740000 | 1 |
| int_00000001 | user_004478 | video_0020797 | session_00000001 | 2025-08-14T10:05:56.487Z | 50 | 1.000000 | 1 |
| int_00000002 | user_013708 | video_0011401 | session_00000001 | 2025-08-04T07:47:09.045Z | 3 | 0.200000 | 0 |
| int_00000003 | user_017762 | video_0022813 | session_00000001 | 2025-08-07T10:46:07.763Z | 8 | 0.177778 | 0 |
| int_00000004 | user_017542 | video_0007516 | session_00000001 | 2025-08-14T13:49:08.648Z | 26 | 0.742857 | 0 |
| int_00000005 | user_007288 | video_0030665 | session_00000001 | 2025-08-08T20:30:00.016Z | 29 | 0.828571 | 1 |
| int_00000006 | user_009164 | video_0034865 | session_00000001 | 2025-08-07T13:46:26.523Z | 20 | 1.000000 | 1 |
| int_00000007 | user_011672 | video_0017351 | session_00000001 | 2025-08-08T23:26:14.121Z | 40 | 1.000000 | 0 |

## Numeric distribution

| | watch_time_seconds | watch_ratio | target_engaged |
|---|---|---|---|
| mean | 19.560264 | 0.519606 | 0.277916 |
| std | 16.307897 | 0.360036 | 0.447972 |
| min | 0.000000 | 0.000000 | 0.000000 |
| 25% | 6.000000 | 0.200000 | 0.000000 |
| 50% | 15.000000 | 0.466667 | 0.000000 |
| 75% | 30.000000 | 0.950000 | 1.000000 |
| max | 60.000000 | 1.000000 | 1.000000 |

Neither feature has missing values or needs imputation, and their scales are close, which is an incidental benefit of keeping only these two in the end. The signal is weak (test set ROC-AUC 0.5796), a point covered in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

## Self-check commands

```bash
python -m single_prediction.prepare_data
head -1 data/processed_interactions.csv    # should print the 8 column names in the table above
wc -l data/processed_interactions.csv      # should be 500001 (including the header)
```
