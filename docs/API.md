# API Contract

Interactive documentation at `http://127.0.0.1:8000/docs` once the service runs.

---

## `GET /health`

Liveness and readiness. Returns 200 even when the model fails to load, with
`status` set to `not_ready` and the cause attached.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_store_loaded": true,
  "model_name": "logistic_regression+isotonic",
  "model_version": "1.0.0",
  "users_indexed": 25000,
  "videos_indexed": 35000,
  "uptime_seconds": 128.4,
  "error": null
}
```

---

## `POST /predict`

**Request**

```json
{"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 45.0, "hour_of_day": 14}
```

| Field | Type | Constraint | Used as a feature? |
|---|---|---|---|
| `user_id` | string | matches `user_[A-Za-z0-9_-]+`, length 6–80 | no — resolves the video/user and produces 404 when unknown |
| `video_id` | string | matches `video_[A-Za-z0-9_-]+`, length 7–80 | no — supplies the duration behind `watch_ratio` |
| `watch_time` | float | 0 ≤ x ≤ 3600 seconds | **yes** |
| `hour_of_day` | int, optional | 0–23; defaults to the current UTC hour and is echoed | no — see ADR-002 |

Unknown fields are rejected (`extra="forbid"`) so a typo fails loudly instead of
being silently ignored.

**Response 200**

```json
{
  "user_id": "user_000001",
  "video_id": "video_0000001",
  "watch_time": 45.0,
  "hour_of_day": 14,
  "probability": 0.381201,
  "confidence": "low",
  "predicted_engaged": true,
  "threshold": 0.381201,
  "model_name": "logistic_regression+isotonic",
  "model_version": "1.0.0",
  "response_time_ms": 1.83,
  "timestamp": "2026-08-19T22:41:03.117+00:00"
}
```

`threshold` is returned explicitly. `predicted_engaged` is the probability
compared against it — **not** against 0.5. The model's maximum output on this
dataset is 0.389, so a 0.5 cut would classify every request as "not engaged".

`confidence` measures distance from a coin flip, not correctness. At this signal
level most predictions legitimately read `low`, and that is the honest answer.

**Errors**

| Code | Condition |
|---|---|
| 404 | Identifier is well formed but absent from the store |
| 422 | Missing field, wrong type, out of range, malformed identifier, unexpected field |
| 503 | Model not loaded; `/health` reports the cause |

The distinction between 404 and 422 is deliberate: *malformed* is the caller's
formatting problem, *unknown* is a data problem, and they need different fixes.

---

## `GET /model/info`

The active model, the selection evidence and the feature list.

```json
{
  "model_name": "logistic_regression+isotonic",
  "model_version": "1.0.0",
  "trained_at": "2026-08-19T22:38:05+00:00",
  "split_strategy": "chronological",
  "selection_metric": "validation roc_auc with a paired-bootstrap tie test, then cost",
  "recommended_threshold": 0.381201,
  "features": ["watch_time_seconds", "watch_ratio"],
  "metrics": {"logistic_regression": {"roc_auc": 0.5711, "...": "..."}},
  "test_metrics": {"roc_auc": 0.579572, "brier": 0.196939, "ece": 0.003533}
}
```

`features` lists two entries while the request carries four fields. That gap is
intentional and explained in ADR-002.

---

## Examples

```bash
# valid
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_000001","video_id":"video_0000001","watch_time":45,"hour_of_day":14}'

# unknown identifier -> 404
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_999999","video_id":"video_0000001","watch_time":45}'

# malformed identifier -> 422
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"oops","video_id":"video_0000001","watch_time":45}'

# boundary values -> 200
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_000001","video_id":"video_0000001","watch_time":0,"hour_of_day":0}'
```
