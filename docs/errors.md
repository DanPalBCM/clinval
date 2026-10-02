# Error mechanism taxonomy

Scoring tells you *how much* a system is wrong. This is about *why*, and the
distinction drives different fixes:

- a **window-placement** error is a pipeline bug
- a **misreading** is a prompt problem
- an **annotation-threshold** disagreement may mean the annotation guideline is
  underspecified rather than that anybody erred

## The seven mechanisms

| Mechanism | Meaning |
| --- | --- |
| `annotation_threshold` | both sides point at the finding, but the quoted extents disagree below the overlap threshold |
| `window_placement` | the evidence exists and was read correctly, but attributed to the wrong time window |
| `misreading` | the system misread the note — a negated mention taken as asserted, or an answer reached from unrelated text |
| `missing_evidence` | a positive asserted with no supporting span, or gold positive with no span to check against |
| `ontology_mismatch` | the right concept, the wrong code |
| `temporal_error` | the mention refers to another time or another person — history, family history, a plan |
| `other` | explicit fallback: no rule matched |

A mechanism showing **zero** was checked and not found. That is deliberately
distinguishable from *not checked* — the report prints every tag including the
zeros.

A large `other` count means the taxonomy is missing a mechanism for your data,
not that the errors have no cause.

## Error types

| Type | Condition |
| --- | --- |
| `FP` | predicted positive, gold negative |
| `FN` | predicted negative, gold positive |
| `MATCH` | both agree on the boolean, but span or code matching failed |

`MATCH` is a clinval addition to the spec's `FP | FN`. Those two cannot express
a real and separately actionable failure: the label is right but the evidence
or the code is wrong. A system that correctly says "this patient had diarrhea"
while quoting the wrong sentence has made a mistake neither FP nor FN
describes. See [decisions](decisions.md) D-005. Consumers that only handle two
types can filter `error_type != "MATCH"`.

## Rule order

The first matching rule wins, and the chosen tag always carries a
`mechanism_reason` naming the evidence that produced it. **A tag you cannot
audit is worse than no tag.**

### False positives

1. **`missing_evidence`** — no predicted span at all.
2. **`misreading`** — a negation cue in the quoted span (`denies`, `no
   evidence of`, `ruled out`, `negative for`, `resolved`, …).
3. **`temporal_error`** — a temporal or other-subject cue (`history of`,
   `h/o`, `family history`, `years ago`, `plans to`, …).
4. **`ontology_mismatch`** — the predicted code disagrees with gold's, or
   denotes a different concept that *is* gold-positive for this patient under
   another label.
5. **`window_placement`** — gold is positive for this patient and label in a
   different window.
6. **`annotation_threshold`** — the predicted span overlaps an annotated span
   but below the threshold.
7. **`other`**.

### False negatives

1. **the evidence policy** — under `require_span`, a positive that lost credit
   scores as a negative; the mechanism is *why it lost credit* (a boundary, a
   missing span, a wrong code), not a misreading.
2. **`window_placement`** — the system is positive for this patient and label
   in a different window, or the note could not be placed in a window at all.
3. **`temporal_error`** — the annotated span carries a temporal cue the system
   may reasonably have excluded.
4. **`misreading`** — the annotator quoted supporting evidence in this note and
   the system still said no.
5. **`missing_evidence`** — gold is positive but carries no span.
6. **`other`**.

### Right label, wrong evidence (`MATCH`)

1. **`ontology_mismatch`** — codes differ.
2. **`annotation_threshold`** — both spans present, overlapping, below the
   threshold.
3. **`missing_evidence`** — one side carries no span.
4. **`misreading`** — spans present but not overlapping at all, so the system
   reached the right answer from different text.
5. **`other`**.

## Cues are explanations, never scores

`NEGATION_CUES` and `TEMPORAL_CUES` are deliberately conservative surface
patterns. They are used to **explain a disagreement after the fact** and never
change a value. A negated mention does not become a negative prediction because
clinval spotted the word "denies"; the system's own output stands, and the cue
only labels the mistake.

## `window_placement` needs the window grid

That mechanism is about a note landing in the wrong window, so it requires the
aggregated frame:

```python
aligned = cv.align_predictions(preds, golds, config)
windowed = cv.aggregate_to_windows(aligned, config.window, index=index)
errors = cv.tag_errors(aligned, config, windowed=windowed)   # window context available
```

Called without `windowed`, the tag is simply never assigned — honest
degradation rather than a guess. `run_eval` always passes it.

Every note in a multi-note unit gets window context, not just the unit's
representative note; see [decisions](decisions.md) D-017.

## The export

`errors.csv`, one row per error:

```
error_id, system, label, unit_key, error_type, patient_id, window,
mechanism, span_text, note_id, mechanism_reason, gold_span_text,
span_jaccard, pred_code, gold_code, code_how, confidence
```

The first ten columns are the spec's §6.4 schema; the rest are clinval's audit
trail. `error_id` values are stable across runs on the same input.

## Working with the table

```python
errors = report.errors

errors.by_mechanism()   # every tag, including zeros
errors.by_label()       # label × mechanism
errors.by_window()      # window × mechanism
errors.by_system()      # system × type × mechanism

errors.sample("misreading", n=5)   # eyeball a few
errors.to_csv("errors.csv")
```

## Every mechanism has a triggering fixture

`tests/test_errors.py::TestEveryMechanismHasAFixture` asserts that all seven
tags are reachable, and fails if a change makes one unreachable. That is the
spec's §11 requirement written as a test rather than as a claim.
