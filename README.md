# Cognitive Shorts: Single Prediction

An end-to-end engagement prediction system for short-video interactions. Raw
interaction logs are processed offline into a leakage-safe feature table, four
candidate models are compared against a cost-aware statistical decision rule, and
the selected model is served through a FastAPI service with a Streamlit front
end.

The headline result is a negative one, and it is the most useful thing in the
repository: **on this dataset only watch behaviour predicts engagement.**
Twenty-three of twenty-five candidate features are statistically indistinguishable
from noise, and any honest model is bounded at roughly 0.58–0.60 ROC-AUC. The
project is built to demonstrate that claim rather than assert it.

---

## Contents

- [Results](#results)
- [Pipeline](#pipeline)
- [Feature selection](#feature-selection)
- [Model selection](#model-selection)
- [The operating point](#the-operating-point)
- [Data and label definition](#data-and-label-definition)
- [Interface](#interface)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Deployment](#deployment)
- [Verification](#verification)
- [Limitations](#limitations)
- [References](#references)

---

## Results

Measured on the latest 15% of the log by time, held out until the final report.

| Metric | Value | Note |
|---|---:|---|
| ROC-AUC | **0.5796** | against a ceiling of roughly 0.60 for this data |
| PR-AUC | 0.3267 | base rate 0.2807 |
| Brier | 0.1969 | after isotonic calibration |
| ECE | 0.0035 | calibration error, from 0.0273 before calibration |
| Features | **2** | selected from 25 candidates by measurement |
| Artefact | 0.002 MB | 1,958x smaller than the largest candidate |
| Recommended threshold | 0.3812 | flags 23.9% of traffic at **1.40x lift** |

The model is small, calibrated, and honest about a weak signal. That combination
is the deliverable.

---

## Pipeline

```text
interactions.csv (500k rows, 43 columns)      users.csv      videos.csv
             │                                     │              │
             ▼                                     ▼              ▼
   prepare_data.py     chunked streaming, Pandera contract check
             ▼
   processed_interactions.csv     500k rows x 2 features + label + split keys
                                  (8 columns; preview in docs/DATA_PREVIEW.md)
             ▼
   train.py            chronological split -> 4 candidates
                       -> paired-bootstrap tie test -> cost-based selection
                       -> calibration gate -> operating-point analysis
             ▼
   best_model.joblib · model_metadata.json · training_report.json · 6 figures
             ▼
   api.py              POST /predict · GET /health · GET /model/info
             ▼
   app.py              Streamlit: input validation, gauge, model info
```

---

## Feature selection

Full derivation and every measurement: **[docs/FEATURE_SELECTION.md](docs/FEATURE_SELECTION.md)**.
Reproduce with `python scripts/feature_selection.py`.

Each candidate is screened univariately with a bootstrap interval and
multivariately with permutation importance. A feature survives if either test
clears its threshold.

| Feature | ROC-AUC | 95% interval | Verdict |
|---|---:|---|---|
| `watch_time_seconds` | 0.5641 | [0.5623, 0.5660] | **keep** |
| `watch_ratio` | 0.5569 | [0.5551, 0.5588] | **keep** |
| `video_category` | 0.5065 | [0.5038, 0.5093] | drop |
| `user_country` | 0.5060 | [0.5033, 0.5085] | drop |
| `hour_of_day` | 0.5010 | [0.5000, 0.5034] | drop |
| *…20 more* | ≤0.5063 | all reaching 0.500 | drop |

Pruning costs nothing: the 25-feature model scores 0.5739 and the 2-feature model
0.5721, a gap of 0.0018 against a standard error of about 0.0045. The two are
indistinguishable, so the smaller one wins.

**Why the other features are noise.** `users.csv` and `videos.csv` share
identifier spaces with the interaction log but not attribute values. For the same
`user_id`, the age on the interaction row and the age in `users.csv` disagree on
**97.1%** of rows with a correlation of **0.0014**. Every shared field behaves
the same way. Joining those files attaches values unrelated to the interaction
being predicted — which is why a thirty-two-feature version of this project could
train without any metric revealing the problem.

**Leakage control.** `engagement_score` alone scores ROC-AUC **0.836**. It is
computed from the outcome and does not exist when a request arrives. It, the five
action columns, and `skipped_quickly` are excluded by an explicit blocklist in
`features.py`, and a test asserts the blocklist never intersects the feature set.

---

## Model selection

Four candidates across three model families, on the validation split:

| Model | ROC-AUC | 95% interval | Gap to leader | Latency | Artefact |
|---|---:|---|---|---:|---:|
| gradient_boosting | 0.5742 | [0.5696, 0.5786] | — | 1.23 ms | 0.196 MB |
| random_forest | 0.5736 | [0.5688, 0.5779] | [−0.0011, +0.0023] | 34.69 ms | 3.916 MB |
| lightgbm | 0.5732 | [0.5684, 0.5775] | [−0.0009, +0.0029] | 1.74 ms | 0.692 MB |
| **logistic_regression** | 0.5711 | [0.5667, 0.5757] | [−0.0002, +0.0060] | **1.03 ms** | **0.002 MB** |

![Candidate scorecard](image/model_comparison.png)

Every gap interval contains zero: the four models are **statistically tied**. The
per-model standard error (~0.0045) exceeds the entire spread between them
(0.0031), so ranking by point estimate would ranks sampling noise and would flip
on a different seed.

The rule encoded in `train.py`:

```
1. Cost veto      latency > 50 ms or artefact > 100 MB -> disqualified
2. Tie test       paired bootstrap; gap interval containing 0 -> tied
3. Cheapest wins  among the tied, smallest artefact then lowest latency
```

Winner: **logistic regression**, 1,958x smaller and 34x faster than the random
forest with no measurable loss. This is the one-standard-error rule from CART
(Breiman et al., 1984) applied to operational cost.

The tie is itself the finding: four model families converging within noise means
the ceiling is set by the available signal, not by model capacity.

### Calibration

![Calibration](image/calibration.png)

| Metric | Raw | Calibrated |
|---|---:|---:|
| Brier | 0.19840 | **0.19694** |
| ECE | 0.02725 | **0.00353** |

Isotonic calibration is fitted on validation and shipped only because Brier — a
strictly proper scoring rule — improved. ECE alone is not a sufficient gate: it
can be driven toward zero by collapsing every prediction to the base rate.

One consequence is documented rather than hidden: isotonic regression is a step
function, and at this signal level it collapses to **eight distinct output
probabilities**. Ranking is unaffected; threshold choice is quantised.

---

## The operating point

![Operating points](image/operating_points.png)

The calibrated model's maximum output is **0.389**. No prediction reaches 0.5, so
at the default threshold every candidate reports F1 = 0.0 — which is exactly the
degenerate result a previous version of this project shipped without noticing.
That was never a broken model; it was a broken threshold.

F1-optimisation does not rescue it. Across the sweep F1 peaks at threshold 0.01,
flagging 100% of traffic: on a weak-signal problem F1 collapses into "treat
everybody", which is true and useless as guidance.

The operating point is therefore chosen by budget:

| Traffic flagged | Threshold | Precision | Recall | Lift |
|---:|---:|---:|---:|---:|
| 7.3% | 0.3890 | 0.396 | 0.102 | 1.41x |
| **23.9%** | **0.3812** | **0.393** | **0.334** | **1.40x** |
| 100% | 0.2366 | 0.281 | 1.000 | 1.00x |

**Recommendation:** threshold 0.3812. Reaching a quarter of traffic selects a
group 1.40x richer in engaged users than random, capturing a third of all
engagement. `/predict` returns this threshold explicitly, and `predicted_engaged`
is computed against it rather than against 0.5.

![Decile lift](image/lift_deciles.png)

The decile table shows where the signal lives: the top three deciles run at
1.33–1.41x lift and hold 41% of engaged users, while deciles four through ten are
flat at about 0.85x. The model separates a top ~30% and is uninformative below
that — which is worth knowing before promising more.

---

## Data and label definition

| File | Rows | Content |
|---|---:|---|
| `data/interactions.csv` | 500,000 | View log with timestamps, sessions and outcomes |
| `data/users.csv` | 25,000 | User profiles |
| `data/videos.csv` | 35,000 | Video metadata |
| `data/processed_interactions.csv` | 500,000 | **Generated** feature table — the only input `train.py` reads |

> **Data preview:** [docs/DATA_PREVIEW.md](docs/DATA_PREVIEW.md) — schema, first rows,
> distributions and null counts for the generated table, plus a 20-row
> [sample CSV](docs/processed_interactions_sample.csv) you can open directly. The full
> file is gitignored; rebuild it with `python -m single_prediction.prepare_data`.

**Label.** `target_engaged = liked OR shared OR commented OR followed_creator OR
replayed`. Positive rate **27.79%**.

**Split.** Chronological: earliest 70% train, next 15% validation, latest 15%
test. The log spans 2025-07-14 to 2025-08-14, so this matches production, where
the past predicts the future.

The training script also fits the winner on a random split and reports both.
Chronological test ROC-AUC is 0.5796 against 0.5770 for the random split — a
difference of −0.0026. **The expected optimism did not materialise**: this data
has no material temporal drift. The chronological split is kept because it is the
one that matches production and will show drift the day it appears, but the
honest report is that it changed nothing here.

`interactions.csv` is 139 MB, above GitHub's 100 MB single-file limit, and is
excluded by `.gitignore`. It must be supplied locally.

---

## Interface

![Single Prediction page](image/ui_single_prediction.png)

Client-side format validation (specification §7.1) rejects a malformed identifier
before any network call, so a typo produces an immediate message rather than a
round trip and a 422. The server repeats the check — that copy is the one that
protects the service.

---

## Repository layout

```text
app.py                            Streamlit front end
Dockerfile, docker-compose.yml    one image: train, api, console
src/single_prediction/
  config.py                      paths and thresholds, all env-overridable
  features.py                    the feature contract and blocklists
  prepare_data.py                chunked offline feature table construction
  train.py                       candidate comparison, tie test, calibration, operating points
  metrics.py                     discrimination, calibration, operating-point analysis
  api.py                         /predict · /health · /model/info
  plots.py                       six figures, regenerated by train.py
scripts/
  feature_selection.py           the measurement behind the two-feature model
  prepare_data.py / train_models.py   CLI entry points
monitoring/
  monitoring.yaml                thresholds derived from the training run
  check_drift.py                 PSI input drift, output drift, with a self-test
docs/
  DEPLOYMENT.md                  running it, locally and in containers
  DATA_PREVIEW.md                schema and sample of the generated table
  processed_interactions_sample.csv   first 20 rows, openable in any viewer
  FEATURE_SELECTION.md           why two features
  DESIGN_DECISIONS.md            seven ADRs
  API.md                         endpoint contract
tests/                           43 tests covering specification §9.2 in full
image/                           figures
reports/                         feature_selection.json, training_report.json
```

---

## Quick start

Python 3.10–3.13.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Place `interactions.csv`, `users.csv` and `videos.csv` in `data/`, then:

```bash
python scripts/feature_selection.py          # ~4 min, writes the evidence
python -m single_prediction.prepare_data     # ~1 min
python -m single_prediction.train            # ~1 min, writes models/ reports/ image/
pytest -q                                    # 43 passed
python monitoring/check_drift.py --self-test
```

Services, in two terminals:

```bash
uvicorn single_prediction.api:app --reload --host 127.0.0.1 --port 8000
streamlit run app.py
```

Console at http://localhost:8501. Two links skip the form:
**`?demo=1`** fills a valid request and predicts it, **`?demo=invalid`** fills a
malformed identifier so the validation path renders. The two screenshots above
are those URLs, captured unedited.

---

## Deployment

One image, three entry points — a one-shot trainer, the API on 8000, the
console on 8501:

```bash
docker compose --profile train run --rm train   # writes models/, needs data/interactions.csv
docker compose up --build
```

The trainer is behind a profile so `up` never retrains by accident, and
`models/` is a bind mount rather than an image layer because it is gitignored —
a fresh clone has no model, and the API will honestly report `degraded` until
one exists. The console waits on a health check that tests `status == healthy`
rather than the port, so it never starts against a service that cannot score.

Full guide — configuration, probes, scaling, drift checks, troubleshooting:
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. The local path there was run end
to end; the Docker build was not, because Docker is not installed on the
development machine, and the document says so.

---

## Verification

| # | Step | Expected |
|---|---|---|
| 1 | `python -m single_prediction.prepare_data` | 500,000 rows, positive rate 27.79% |
| 2 | `python scripts/feature_selection.py` | keeps 2 of 25; pruned and full within ±0.002 |
| 3 | `python -m single_prediction.train` | four candidates reported tied; winner logistic regression; 6 figures |
| 4 | `pytest -q` | 43 passed |
| 5 | `GET /health` | `status: healthy`, 25,000 users and 35,000 videos indexed |
| 6 | `POST /predict` valid | probability in [0, 1] with confidence and threshold |
| 7 | `POST /predict` malformed | 422 |
| 8 | `POST /predict` unknown id | 404 |
| 9 | Boundaries: `watch_time` 0 and 3600, `hour_of_day` 0 and 23 | 200 |
| 10 | Move the model file, restart | `/health` reports `not_ready` with the cause; `/predict` returns 503 |
| 11 | `python monitoring/check_drift.py --self-test` | quiet traffic stays quiet, shifted traffic alerts |

---

## Limitations

| Limitation | Consequence | Remediation |
|---|---|---|
| Only watch behaviour carries signal | ROC-AUC is bounded near 0.60 | New data: real user history, video embeddings, sequence context |
| `videos.csv` duration disagrees with the log on 90% of rows | Serving computes `watch_ratio` from a different duration than training | A point-in-time feature store returning values as of a timestamp |
| Snapshot files are uncorrelated with the log | User and video attributes cannot be used at all | Regenerate the dataset so the three files describe one universe |
| Isotonic output is quantised to 8 values | Threshold choice is coarse | Sigmoid calibration, or accept the quantisation |
| No online monitoring wired up | Config and checker exist but nothing runs them | Schedule `check_drift.py`; add the labelled calibration job |
| No A/B framework | Offline lift is not shown to move business metrics | Test the 23.9% treatment group against a holdout |

---

## References

- Breiman, L. et al. (1984). *Classification and Regression Trees.* — the one-standard-error rule.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* — the paired bootstrap.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning.* ICML.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets.* PLOS ONE 10(3).
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
- Zinkevich, M. *Rules of Machine Learning.* https://developers.google.com/machine-learning/guides/rules-of-ml
- scikit-learn, *Probability calibration.* https://scikit-learn.org/stable/modules/calibration.html

---

## Related

- Batch prediction: the follow-on project, adding `/predict/batch` with per-row fault tolerance.
