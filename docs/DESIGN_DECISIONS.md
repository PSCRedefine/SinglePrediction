# Architecture Decision Records

Each record states the context, the options, the decision and its cost. The cost
line matters most: a decision with no cost usually means one that was not
examined.

---

## ADR-001 · Features are chosen by measurement, not by availability

**Context.** Twenty-five candidate features can be assembled from the request and
the snapshot stores. The previous version used thirty-two of them.

**Decision.** Keep only features whose contribution is distinguishable from
noise. On this dataset that is two: `watch_time_seconds` and `watch_ratio`.

**Cost.** The model looks thin next to a thirty-two-feature table, and a reader
expecting a rich feature set has to be shown the measurement before the choice
makes sense — hence `docs/FEATURE_SELECTION.md`.

**Benefit.** Identical performance (±0.002 ROC-AUC, inside the standard error)
from a model with 23 fewer inputs to serve, drift-check and explain. Twenty-three
noise features are not free: each one is a column that can go missing, change
type, or drift, and each one dilutes any real feature-importance analysis.

---

## ADR-002 · The API contract and the feature set are separate

**Context.** The specification defines `hour_of_day` as a request field.
Measurement puts its univariate ROC-AUC at 0.5010, with an interval reaching
0.5000.

**Decision.** Accept it, validate it, echo it in the response — and do not train
on it.

**Rationale.** A field being in the contract is a statement about the interface;
a field being in the model is a statement about evidence. Conflating them is
precisely how noise enters models. Keeping the contract intact also means the
field is already there if a future dataset gives it signal.

**Cost.** A reader comparing the request schema against `features` in
`/model/info` will notice they differ, so both places say why.

---

## ADR-003 · The split is chronological

**Context.** The log spans one month and carries a real timestamp and a session
identifier. The previous version used a random row split.

**Decision.** Train on the earliest 70%, validate on the next 15%, report on the
latest 15%. The training script additionally fits the winner on a random split
and reports both numbers.

**Result.** Chronological test ROC-AUC 0.5796 against 0.5770 for a random split —
a difference of −0.0026. **The hypothesis was not confirmed**: this dataset has
no material temporal drift, so the random split was not inflating the previous
result.

**Why keep it anyway.** The chronological split is the one that matches
production, where the past predicts the future. It costs nothing here, and the
day the data does drift it will show the drift instead of hiding it. Reporting
both numbers is what turns an assumption into a measurement.

**Cost.** Slightly more code, and a negative result to explain rather than a
dramatic finding to claim.

---

## ADR-004 · Model choice is a tie test followed by a cost comparison

**Context.** Four candidates span 0.5711 to 0.5742 validation ROC-AUC — a range
of 0.0031, against a per-model standard error of about 0.0045.

**Decision.** Three rules in order: veto on latency and artefact size; establish
which models are statistically tied via a paired bootstrap; take the cheapest of
the tied set.

**Result.** All four are tied — every gap interval contains zero. The winner is
logistic regression: **1,958x smaller** than the random forest (0.002 MB against
3.916 MB) and 34x faster per prediction, with no measurable loss.

**Cost.** "We picked the highest AUC" is a shorter sentence. This requires
explaining what a tie test is.

**Benefit.** The decision survives a rerun. Ranking by point estimate would flip
between gradient boosting and random forest on a different random seed.

---

## ADR-005 · The operating point is chosen by budget, not by F1

**Context.** The calibrated model's maximum output is 0.389. No prediction ever
reaches 0.5, so at the default threshold every model reports F1 = 0.0 — including
the one the previous version shipped. F1 across the sweep is maximised at
threshold 0.01, which flags 100% of traffic.

**Decision.** Report operating points indexed by the share of traffic flagged,
and recommend the highest-lift point that still flags at least 10%.

**Result.** Threshold 0.3812: flags 23.9% of traffic at precision 0.393 against a
base rate of 0.281, so **1.40x lift**, capturing 33.4% of engaged users.

**Rationale.** F1 assumes precision and recall are equally valuable. They rarely
are, and on a weak-signal problem F1 optimisation collapses into "treat
everybody", which is a true statement about F1 and useless as guidance. A
product team can act on "if you can afford to reach a quarter of your traffic,
this picks a group 1.4x richer in engagers than random".

**Cost.** The recommendation depends on a volume floor, which is a judgement
call. It is a parameter of `recommend_threshold`, not a constant buried in the
code.

---

## ADR-006 · Calibration is applied because it was measured to help

**Context.** Isotonic calibration is fitted on the validation split, which the
model never trained on.

**Decision.** Ship it only if the Brier score improves. Here Brier moves 0.19840
to 0.19694 and ECE 0.02725 to 0.00353, so the calibrated model ships.

**Why Brier is the gate.** Brier and log loss are strictly proper scoring rules.
ECE alone is not: collapsing every prediction to the base rate drives ECE toward
zero while destroying discrimination.

**Known consequence.** Isotonic regression produces a step function. With this
much signal it collapses to **eight distinct output probabilities**. Ranking is
unaffected (the mapping is monotone), but threshold choices are quantised, which
is why several traffic budgets resolve to the same cut in the operating-point
table. This is documented rather than smoothed over, and the count is monitored
as a change-detection signal — a sudden change means the artefact was swapped.

---

## ADR-007 · A load failure leaves `/health` serving

**Context.** The specification asks for a clear error when the model file is
missing.

**Decision.** The process starts regardless. `/health` returns 200 with
`status: not_ready` and the underlying exception text; `/predict` returns 503.

**Rationale.** An orchestrator needs an endpoint that answers so it can withdraw
the instance from rotation. A container that exits produces a restart loop, and
the actual cause scrolls past in the logs.

**Cost.** A running process that cannot serve predictions is arguably worse than
one that fails loudly — so the readiness probe, not liveness, is what must be
wired to `status`.
