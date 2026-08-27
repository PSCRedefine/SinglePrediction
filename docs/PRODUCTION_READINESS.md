# Production readiness

What this service does not have yet, and what it would need before it carries
real traffic. Ordered by the risk of shipping without it.

Almost nothing here is a defect against the specification. The specification
stops where it stops; this document is where it would have to resume. It is
written to be argued with — each item states the risk, the remedy and the cost,
so that "accept this one" is a decision somebody can make on the record rather
than an omission nobody noticed.

## Blocking

### 1. The serving feature is not the training feature

`watch_ratio` is derived at training time from the duration recorded on the
interaction row, and at serving time from `videos.csv`. Those two disagree on
about 90% of rows. The model is therefore scoring a quantity it was not fitted
on, every request, silently.

This is the only item in this document that makes predictions *wrong* rather
than *unmonitored*, which is why it is first.

**Remedy.** One source of truth for duration, resolved as of the interaction's
timestamp — a point-in-time feature store, or at minimum a serving path that
reads the same column the training table was built from. Until then, a test
that scores a sample of held-out rows through the serving path and asserts the
features match the training table row for row would at least keep the size of
the gap visible.

**Cost.** Days, not hours. It is a data-architecture change, not a code change.

### 2. The model artefact is loaded on trust

`joblib.load` is `pickle`: loading an artefact executes whatever is inside it.
The artefact arrives on a bind mount with no checksum, no signature and no
record of where it came from. Nothing at load time can distinguish the model
that was trained and reviewed from a file with the same name.

**Remedy.** Publish artefacts to an immutable store, record a content hash in
the release manifest, and verify that hash before `joblib.load` rather than
after. Record provenance alongside it — training run identifier, data snapshot,
code revision and library versions — and refuse to load an artefact whose
provenance block is missing.

**Cost.** Low. A hash check and a metadata block are an afternoon; the
immutable store is a dependency on whatever the platform already provides.

### 3. The build is not reproducible, and the artefact is already drifting

`requirements.txt` pins ranges (`scikit-learn>=1.5,<2.0`), so the interpreter
that trained a model and the container that loads it can resolve different
libraries. Loading the shipped artefact under NumPy 2.4 already emits
deprecation warnings from the unpickling path — the artefact still loads, but it
is on a clock.

**Remedy.** A lockfile for the image build, with the ranges kept only for
development. Record the exact `scikit-learn`, `numpy` and `joblib` versions in
the model metadata at training time, and compare them at load time — warn on a
minor mismatch, refuse on a major one.

**Cost.** Low, but it comes with an obligation: a lockfile nobody regenerates is
worse than a range, because it looks maintained. Only adopt it with a scheduled
refresh.

### 4. `/predict` is unauthenticated and unmetered

Any caller can send unlimited requests. There is no identity on a request, so
there is also no way to attribute load, revoke a caller, or bill one.

**Remedy.** Authentication at the edge, a per-caller quota, and a concurrency
limit on the process. The limit matters independently of the quota: the model
call holds a worker thread, and enough concurrent requests will exhaust the pool
regardless of how polite each individual caller is.

**Cost.** Low if the platform has an API gateway. Do not build it here.

### 5. The identifier snapshot never refreshes

`users.csv` and `videos.csv` are read once at start-up. A user created after
that snapshot resolves to a 404 for the lifetime of the process. In real traffic
that is a growing fraction of requests failing, and on the dashboard it looks
like client-side error noise rather than staleness.

**Remedy.** A lookup service, or a periodic reload with a freshness metric
exposed on `/health`. Separately, an identifier that is *unknown* and an
identifier that is *malformed* should not both surface as client errors — they
have different causes and different owners.

**Cost.** Medium. It is the point at which the service acquires a runtime
dependency it does not currently have.

### 6. `/health` answers two different questions at once

An orchestrator asks two things: is this process alive, and can it serve? The
correct response to the first is a restart; to the second, removal from the load
balancer. One endpoint reporting `not_ready` cannot drive both, and a restart
loop against a missing artefact is the wrong reaction.

**Remedy.** Split liveness from readiness. Liveness stays trivially true while
the process is running; readiness reflects the model and the store.

**Cost.** An hour.

---

## Required within the first quarter

### 7. A prediction cannot be traced

There is no request identifier, no trace context, and the logs are formatted for
a human reader. Given a probability that looks wrong, there is no path from it
back to the call that produced it, the features that went in, or the artefact
that was loaded at the time.

**Remedy.** A request identifier accepted or generated at the edge and echoed on
every response and log line; structured (JSON) logs; trace context propagated to
anything downstream. Log the feature vector for a sampled fraction of requests —
without it, debugging a bad prediction is guesswork.

### 8. Calibration drift is currently unobservable

The headline property of this model is that its output is a rate, not a rank.
Nothing checks whether that stays true. Input drift and output drift are both
detectable; the failure that actually costs money — the model still reporting
0.38 while the real rate has moved to 0.25 — is not, because no outcome ever
comes back.

**Remedy.** A delayed-label job: join predictions to the engagement outcomes
that arrive hours or days later, recompute Brier and ECE on that window, and
alert on the calibration curve moving rather than on the score distribution
moving. This is the single most valuable thing missing from the system.

**Cost.** Medium, and it is the piece that requires a decision about how long to
wait before a non-engagement counts as a negative.

### 9. There is no way to roll a model back

Promotion is a file copy and rollback is the same file copy in reverse. There is
no registry, no canary, no shadow traffic, and no record of which artefact
served which request.

**Remedy.** A registry with immutable versions; `model_version` recorded on
every prediction and every log line; promotion behind a canary; rollback as a
pointer change rather than a redeploy.

### 10. Retraining has no cadence and no gate

Retraining is a manual command. Nothing decides when it should happen, and
nothing prevents a worse model from being promoted.

**Remedy.** A schedule, plus an automated gate: a candidate is promoted only if
it beats the incumbent on a chronological hold-out by more than the bootstrap
interval that `train.py` already computes. The selection machinery for this
exists; it just is not wired to the release.

### 11. The latency numbers are not load numbers

The recorded percentiles come from sequential calls on a development machine.
They say nothing about behaviour under concurrency, and there is no stated
objective to hold the service to.

**Remedy.** An availability and latency objective with an error budget, and a
load test that establishes the knee. Publish the concurrency at which the
objective stops holding.

### 12. The operating point has no owner

The threshold is a business decision — how much traffic to pay for — that
currently lives as a number in a JSON file. When the cost of the intervention
changes, the correct threshold changes, and nothing prompts anyone to revisit
it.

**Remedy.** Name an owner, a review cadence, and a change log. Keep the value in
metadata; move the decision somewhere it can be argued about.

---

## Accepted, with reasons

| Item | Why it is acceptable |
|---|---|
| ROC-AUC near 0.58 | The ceiling was measured, not assumed. Better features are a data problem, not a modelling one. |
| Isotonic output quantised to eight values | Ranking is unaffected; only threshold granularity suffers, and the operating point does not fall near a step. |
| Two features | Measured to be indistinguishable from twenty-five. Re-measure when new data arrives, not before. |

---

## What is already in place

Listed so this document reads as a review and not a confession.

| Concern | Where it is handled |
|---|---|
| Load failure does not kill the process | `/health` stays serviceable and reports the cause |
| Non-root container, dependency layer cached separately | `Dockerfile` |
| The console starts only behind a real health check | `docker-compose.yml` |
| Configuration is environment-backed, not hard-coded | `config.py` |
| The suite runs on every supported interpreter | `.github/workflows/tests.yml` |
