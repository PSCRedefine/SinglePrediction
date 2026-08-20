# processed_interactions.csv — 数据预览

`prepare_data.py` 的产物，也是 `train.py` 唯一的输入。本文件是它的“缩略图”，方便随时查阅而不必打开 47 MB 的原文件。

- 完整文件：`data/processed_interactions.csv`（已 gitignore，用 `python -m single_prediction.prepare_data` 重建）
- 可直接打开的样本：[`processed_interactions_sample.csv`](processed_interactions_sample.csv)（前 20 行）
- 重建耗时约 1 分钟

## 规模

| 项 | 值 |
|---|---|
| 行数 | 500,000 |
| 列数 | 8 |
| 文件大小 | 47 MB |
| 缺失值 | 0（全部 8 列） |
| 正例率 `target_engaged` | 27.79% |
| 时间跨度 | 2025-07-14 → 2025-08-14（约 1 个月） |

## 列结构

8 列 = 3 个 ID + 2 个切分键 + 2 个特征 + 1 个标签。原始 `interactions.csv` 有 43 列，其余的要么泄漏标签、要么线上取不到，被 `features.py` 有意丢弃——理由见 [FEATURE_SELECTION.md](FEATURE_SELECTION.md)。

| # | 列名 | 类型 | 角色 | 唯一值 | 说明 |
|---|---|---|---|---|---|
| 1 | `interaction_id` | object | ID | 500,000 | 每行唯一 |
| 2 | `user_id` | object | ID | 25,000 | |
| 3 | `video_id` | object | ID | 35,000 | |
| 4 | `session_id` | object | 切分键 | 54,403 | 无 timestamp 时的兜底分组依据 |
| 5 | `timestamp` | object | 切分键 | 499,915 | ISO-8601 UTC，**时序切分依赖此列** |
| 6 | `watch_time_seconds` | int64 | **特征** | 61 | 0–60 秒 |
| 7 | `watch_ratio` | float64 | **特征** | 231 | 0–1，观看时长 / 视频时长 |
| 8 | `target_engaged` | int64 | **标签** | 2 | liked OR shared OR commented OR followed_creator OR replayed |

> `timestamp` 缺失会让 `train.py` 静默退化为随机切分而非时序切分——两列切分键必须存在。

## 前 8 行

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

## 数值分布

| | watch_time_seconds | watch_ratio | target_engaged |
|---|---|---|---|
| mean | 19.560264 | 0.519606 | 0.277916 |
| std | 16.307897 | 0.360036 | 0.447972 |
| min | 0.000000 | 0.000000 | 0.000000 |
| 25% | 6.000000 | 0.200000 | 0.000000 |
| 50% | 15.000000 | 0.466667 | 0.000000 |
| 75% | 30.000000 | 0.950000 | 1.000000 |
| max | 60.000000 | 1.000000 | 1.000000 |

两个特征都无缺失、无需插补，量纲也接近，这是最终只保留它们的一个附带好处。信号强度有限（测试集 ROC-AUC 0.5796），这一点在 [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) 里有交代。

## 自查命令

```bash
python -m single_prediction.prepare_data
head -1 data/processed_interactions.csv    # 应输出上表的 8 个列名
wc -l data/processed_interactions.csv      # 应为 500001（含表头）
```
