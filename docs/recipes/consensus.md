# Consensus gating

The practical use: run two models, trust the units they agree on, and route the
rest to a human. That only saves review effort if the agreed-on units are
**actually more reliable**, so clinval reports accuracy separately for agreed
and contested units. If they are the same, the gate is buying nothing, and the
report says so rather than leaving it to be inferred.

```bash
uv run clinval consensus \
  --systems sysA.jsonl --systems sysB.jsonl \
  --gold gold.jsonl --config eval.yaml --index index.csv \
  --rule majority --out consensus/
```

```python
cons = cv.consensus_gate(
    {"sysA": preds_a, "sysB": preds_b},
    golds, rule="majority", config=config, index=index,
)
print(cons.to_markdown())
cons.to_review_queue().to_csv("review_queue.csv")
```

## The rules

| Rule | Positive when | Undecided when |
| --- | --- | --- |
| `majority` | more than half of systems say yes | an even split |
| `unanimous` | every system says yes | never |
| `any` | at least one system says yes | never |

`needs_review` is true whenever the systems were **not unanimous**, regardless
of whether the rule reached a verdict. A 2–1 majority still has a dissenter
worth a human's time.

### The two-system trap

With exactly two systems, `majority` requires both to be positive — and a 1–1
split is **undecided**, not negative.

!!! warning "Undecided units are excluded from the score"
    Scoring a 1–1 tie as a negative would be the single easiest way to make a
    gate look good. clinval does not do that. But the consequence is that
    `majority` on two systems is scored only on the units it decided, so a more
    deferring rule can post a better number while answering fewer questions.

On the demo data:

| consensus | κ | decided |
| --- | --- | --- |
| majority | 1.000 | 70.5% |
| unanimous | 0.872 | 100% |
| any | 0.596 | 100% |

`majority` looks perfect because it declined to answer 29.5% of the questions.
`ConsensusResult` always reports `n_review` and `review_fraction` so this is
visible, and tuning `consensus` emits an explicit warning. See
[decisions](../decisions.md) D-007.

## Does the gate buy anything?

The number that justifies or kills it:

```python
avd = cons.agreed_vs_disagreed
avd["agreed"]["accuracy"]      # on unanimous units
avd["disagreed"]["accuracy"]   # on contested units the rule decided
avd["accuracy_lift"]           # the difference
avd["disagreed"]["undecided_excluded"]   # how many it declined to decide
```

A lift near zero means the gate is **not separating reliable from unreliable
units** — you are paying for human review without buying accuracy. The
Markdown output states this in words:

> Accuracy on agreed units exceeds disagreed units by 0.214. A lift near zero
> means the gate is not separating reliable from unreliable units.

On agreed units the systems share one value, so that value is the prediction.
On contested units there is no single system call, so accuracy is measured
against the rule's verdict, and units the rule could not decide are excluded
and counted.

## The review queue

```python
queue = cons.to_review_queue()      # most-contested first
```

Sorted by `split_margin`, the absolute difference between positive and negative
votes, so the most evenly split units come first. With two systems every
contested unit is a 1–1 tie; with three or more, 2–1 splits sort behind 1–1–1
ones.

## Between-system agreement, independent of gold

```python
cons.pairwise["sysA|sysB"]
# {'n_units': 72, 'cohen_kappa': 0.58, 'pabak': 0.61,
#  'observed_agreement': 0.806, 'confusion': {...}}
```

This measures whether two systems **say the same thing**, which is a different
question from whether either is right. Two systems can agree perfectly with
each other while both being wrong — and clinval will show κ 1.0 between them
alongside a confusion table with zero true positives.

High inter-system agreement with low accuracy usually means they share a
failure mode: the same prompt, the same base model, or the same blind spot
about negation.

## Tuning the rule

```yaml
freeze:
  objective: cohen_kappa
  grid:
    consensus: [majority, unanimous, any]
```

This needs predictions from at least two systems; clinval raises with that
message if you supply one. The tuning table gains a `decided` column, and the
deferral warning fires when the winner answers materially fewer questions than
an alternative.

!!! note "Do not tune `consensus` and `min_overlap` in the same grid"
    `consensus` governs the consensus gate; `min_overlap` governs the
    per-system evaluation. Tuning both in one table produces a frozen config
    whose `consensus` value has no effect on what `clinval evaluate` does. The
    bundled walkthrough config tunes `min_overlap` only, for exactly this
    reason.

## Three or more systems

`majority` becomes genuinely decisive: with three systems every unit resolves,
and `undecided_excluded` is zero.

```python
cons = cv.consensus_gate(
    {"sysA": a, "sysB": b, "sysC": c}, golds, rule="majority", config=config, index=index
)
cons.df["n_positive"]    # 0..3
cons.df["consensus"]     # never null with an odd system count
```

## Accepting pre-computed frames

To avoid re-aligning, pass `AlignedFrame` or `WindowedFrame` objects instead of
raw predictions:

```python
frames = {}
for name, preds in systems.items():
    aligned = cv.align_predictions(preds, golds, config)
    frames[name] = cv.aggregate_to_windows(aligned, config.window, index=index)

cons = cv.consensus_gate(frames, golds, rule="unanimous", config=config)
```
