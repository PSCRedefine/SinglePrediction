# Deployment

Three entry points from one image: a one-shot **trainer**, the **API** on 8000
and the **Streamlit console** on 8501. The console holds no model — it talks to
the API over HTTP — so the two long-running services scale and fail
independently.

> **Verification status.** The local path below was run end to end on the
> development machine: both services started, a prediction was scored, and the
> screenshots in `image/` were captured from that running stack. The Docker
> path was **not** executed, because Docker is not installed on that machine.
> The compose file parses and its health-check command was run against the live
> service and returned 0, but the image build itself is unverified. Treat the
> first `docker compose up` as the real test.

---

## What this project needs that a typical service does not

Two things are deliberately **not** in the image, and both shape the
deployment:

- **`data/interactions.csv` (139 MB)** is above GitHub's file limit and is not
  in the repository. Training needs it; serving does not.
- **`models/best_model.joblib`** is gitignored. A fresh clone has **no trained
  model**, so the API starts, reports `degraded`, and returns 503 from
  `/predict` until one exists.

So the deployment has a step that a stateless web service usually skips: train
first, or bring a model with you. The lookup tables (`users.csv`, `videos.csv`,
9 MB, tracked) *are* baked into the image, because identifier resolution needs
them at request time.

## 1. Local, no containers

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Place `interactions.csv`, `users.csv` and `videos.csv` in `data/`, then build
the table and the model:

```bash
python -m single_prediction.prepare_data     # ~1 min
python -m single_prediction.train            # ~1 min, writes models/ reports/ image/
```

Serve, in two terminals:

```bash
uvicorn single_prediction.api:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run app.py
```

Console at http://localhost:8501, API docs at http://localhost:8000/docs.

Two shareable demo links skip the form: **`?demo=1`** fills a valid request and
predicts it, **`?demo=invalid`** fills a malformed identifier so the validation
path renders. The screenshots in `image/` are those two URLs, captured
unedited.

## 2. Docker Compose

### Train once

```bash
docker compose --profile train run --rm train
```

This mounts `./data`, `./models`, `./reports` and `./image` into the container,
runs `prepare_data` then `train`, and writes the model back to your working
copy. It needs `data/interactions.csv` to be present. It is a profile, so a
plain `docker compose up` never runs it by accident.

### Then serve

```bash
docker compose up --build
```

| Service | Port | Role |
|---|---|---|
| `train` | — | One-shot, profile-gated. Writes `models/`, `reports/`, `image/` |
| `api` | 8000 | `uvicorn ... --workers 2`, mounts `./models` read-only |
| `console` | 8501 | Streamlit, waits for the API to report healthy |

Stop with `docker compose down`. Rebuild after a dependency change with
`docker compose build --no-cache`.

### If you already have a model

Skip training and drop `best_model.joblib` and `model_metadata.json` into
`./models`. The API mounts that directory, so nothing needs rebuilding. The
decision threshold travels in the metadata rather than in code, which is what
makes this safe — this model's useful threshold is 0.381, not 0.5, and a
deployment that hard-coded 0.5 would flag nothing at all.

## 3. Configuration

Every setting is an environment variable with a working default.

| Variable | Default | Purpose |
|---|---|---|
| `SINGLE_PREDICTION_API_URL` | `http://127.0.0.1:8000` | Where the console sends requests. Compose sets `http://api:8000` |
| `SINGLE_PREDICTION_MODEL_PATH` | `models/best_model.joblib` | Serve a different model without rebuilding |
| `SINGLE_PREDICTION_METADATA_PATH` | `models/model_metadata.json` | Threshold and model name are read from here |
| `SINGLE_PREDICTION_USERS_PATH` | `data/users.csv` | Identifier resolution |
| `SINGLE_PREDICTION_VIDEOS_PATH` | `data/videos.csv` | Durations for `watch_ratio` |
| `SINGLE_PREDICTION_INTERACTIONS_PATH` | `data/interactions.csv` | Training input only |
| `SINGLE_PREDICTION_PROCESSED_PATH` | `data/processed_interactions.csv` | Training input only |

There are no secrets, no database and no outbound calls.

## 4. Health and readiness

```bash
curl -s localhost:8000/health
```

```json
{"status":"healthy","model_loaded":true,"feature_store_loaded":true,
 "model_name":"logistic_regression+isotonic","model_version":"1.0.0",
 "users_indexed":25000,"videos_indexed":35000,"uptime_seconds":5.3}
```

A load failure does **not** kill the process. `/health` keeps serving and
reports the reason, while `/predict` returns 503. This distinguishes "the
process is up but cannot score" from "the process is down" — two conditions
with different fixes, and on this project the first one is common, because it
is what an untrained clone looks like.

The compose health check therefore tests `status == "healthy"` rather than the
port. Without a model in `./models`, `api` starts but never becomes healthy,
and `console` correctly refuses to start against it.

For an orchestrator: `/health` gated on `model_loaded` is the **readiness**
probe; a plain TCP check is the **liveness** probe. Gating liveness on the
model would restart a container that is up and honestly reporting a problem a
restart cannot fix.

## 5. Scaling

- **Workers.** Each worker holds its own copy of the model (3 KB) and the
  lookup tables (~60 MB resident once parsed). The tables, not the model, are
  the memory cost — this model is a calibrated logistic regression over two
  features, chosen partly for exactly that reason (see
  [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md), ADR-004).
- **Throughput.** Scoring is a dot product; the measured response time is
  dominated by identifier resolution and response assembly.
- **State.** The API keeps none, so replicas behind a load balancer scale
  cleanly. The console keeps per-session state, so multiple console replicas
  need sticky sessions.
- **Training does not belong in the serving path.** It is a separate profile
  precisely so a restart never retrains.

## 6. Monitoring

The image includes `monitoring/check_drift.py`, which compares a live sample
against the training distribution using PSI:

```bash
docker compose run --rm api python monitoring/check_drift.py --self-test
```

Thresholds live in `monitoring/monitoring.yaml` and were derived from the
training run, not chosen by feel.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `api` never becomes healthy | No trained model in `./models` | Run the `train` profile, or drop a model in |
| Console shows `API Offline` | Wrong URL | Under compose it must be `http://api:8000`, not `localhost` |
| `train` fails immediately | `data/interactions.csv` missing | It is gitignored and 139 MB; supply it separately |
| Predictions all say "not engaged" | Expected | The model's maximum output is 0.389 and the threshold is 0.381; see ADR-005 |
| 404 on a prediction | Identifier not in the lookup tables | The shipped tables cover `user_000001`–`user_025000` |
| 422 on a prediction | Malformed identifier | Must match `user_…` / `video_…` |
