# Feature Selection

The model uses two features. This document is the measurement that justifies
that, and `scripts/feature_selection.py` reproduces every number in it.

## Procedure

1. **Restrict to what a request can supply.** The service receives `user_id`,
   `video_id`, `watch_time` and `hour_of_day`. Anything else must be derivable
   from those, from the request clock, or from the user/video snapshot stores.
   Columns observed *during* the interaction are excluded before measurement,
   however predictive they look offline — using them would produce a model that
   cannot be served.

2. **Screen univariately with an interval.** Each candidate is scored by ROC-AUC
   with a 200-sample bootstrap interval. A feature whose interval reaches down
   to 0.505 is not distinguishable from noise. A point estimate alone would let
   0.5065 masquerade as signal.

3. **Screen multivariately.** Permutation importance on a held-out split, so a
   feature that matters only in combination with others is not wrongly dropped.

4. **Keep on either test.** A candidate survives if it clears the univariate
   interval *or* shows a permutation drop larger than twice its own standard
   deviation.

5. **Verify the pruning.** Train the full-feature and pruned models and compare.

## Excluded before measurement

| Column | Reason |
|---|---|
| `engagement_score` | Computed from the outcome. Scores ROC-AUC **0.836** on its own — including it would produce an impressive number and a worthless service |
| `liked`, `shared`, `commented`, `followed_creator`, `replayed` | The label itself |
| `skipped_quickly` | A restatement of the same view's watch behaviour, known only afterwards |
| `watch_percentage` | Not in the request; reconstructed server-side as `watch_ratio` |
| `scroll_velocity`, `sound_on`, `time_since_last_interaction` | Observed during the interaction |
| `session_position`, `recommendation_source` | Known to the recommender, not to the scoring request |
| `app_version` | Client metadata, not carried in the contract |

## Results, 500,000 rows

| Feature | ROC-AUC | 95% interval | Permutation drop | Verdict |
|---|---:|---|---:|---|
| `watch_time_seconds` | 0.5641 | [0.5623, 0.5660] | +0.0717 | **keep** |
| `watch_ratio` | 0.5569 | [0.5551, 0.5588] | +0.0003 | **keep** |
| `video_category` | 0.5065 | [0.5038, 0.5093] | −0.0001 | drop |
| `user_language` | 0.5063 | [0.5037, 0.5091] | +0.0003 | drop |
| `user_country` | 0.5060 | [0.5033, 0.5085] | +0.0006 | drop |
| `user_gender` | 0.5029 | [0.5005, 0.5054] | +0.0004 | drop |
| `day_of_week` | 0.5022 | [0.5002, 0.5047] | +0.0001 | drop |
| `video_trending_score` | 0.5018 | [0.5000, 0.5041] | +0.0002 | drop |
| `hour_of_day` | 0.5010 | [0.5000, 0.5034] | −0.0003 | drop |
| `video_view_count` | 0.5002 | [0.5000, 0.5030] | −0.0006 | drop |
| `user_age` | 0.5001 | [0.5001, 0.5030] | −0.0005 | drop |
| *…14 further candidates* | ≤0.5024 | all reaching 0.500 | ≈0 | drop |

**Kept: 2 of 25.**

## Does pruning cost anything?

| Model | Features | ROC-AUC |
|---|---:|---:|
| Full | 25 | 0.5739 |
| Pruned | 2 | 0.5721 |

A difference of 0.0018, against a standard error of roughly 0.0045. On a
200,000-row run the sign reverses (+0.0016 in favour of the pruned model). The
two are indistinguishable: **23 additional features buy nothing measurable.**

The pruned model is preferred not because it scores higher but because equal
performance from two features is strictly better than equal performance from
twenty-five — less to serve, less to drift, less to monitor, less to explain.

## Why the snapshot features are noise

`users.csv` and `videos.csv` share identifier spaces with the interaction log
but not attribute values. For the same `user_id`, the age recorded on the
interaction row and the age in `users.csv` differ on **97.1%** of rows, with a
correlation of **0.0014**. Every other shared field behaves the same way:

| Field | Rows disagreeing | Correlation |
|---|---:|---:|
| `user_age` | 97.1% | 0.0014 |
| `user_subscriber_count` | 100.0% | 0.0033 |
| `user_account_age_days` | 99.9% | 0.0007 |
| `video_view_count` | 100.0% | −0.0032 |
| `video_like_count` | 99.6% | −0.0024 |
| `video_trending_score` | 99.9% | −0.0033 |

Joining these files therefore attaches values unrelated to the interaction being
predicted. That is why the previous version of this project trained on thirty
noise features without any metric revealing the problem: noise does not make a
model *wrong*, it makes it *pointless*, and only a per-feature measurement shows
the difference.

## `watch_ratio` and the duration disagreement

`video_duration_seconds` on the interaction row and `duration_seconds` in
`videos.csv` agree on only **9.9%** of rows. Computing `watch_ratio` from the
interaction row's own duration gives 0.5788 univariate AUC; computing it from
`videos.csv` gives 0.5578.

Training therefore uses the interaction row's duration. Serving has no choice
but to use the video snapshot, because a live request cannot know what duration
was recorded historically. This is a real, unresolved inconsistency in the
dataset rather than a modelling decision, and it is recorded here instead of
being smoothed over. On a real platform the fix is a point-in-time feature store
that returns the value as of a given timestamp.

## The ceiling

Only watch behaviour predicts engagement in this dataset. That bounds any honest
model at roughly **0.58–0.60 ROC-AUC**. A materially higher number on this data
indicates leakage — most likely `engagement_score`, which reaches 0.836 alone.

Knowing the noise floor is a result, not a disappointment. It says the next
improvement must come from data that does not exist here — genuine user history,
video embeddings, sequence context — rather than from a larger model.

## Re-running

```bash
python scripts/feature_selection.py --rows 500000
```

Writes `reports/feature_selection.json`. If the kept set changes on new data,
that is a signal worth reviewing before retraining.
