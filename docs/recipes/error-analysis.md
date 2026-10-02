# Error analysis workflow

A metric tells you whether to ship. An error analysis tells you what to fix.
This is the loop clinval is built around.

## 1. Start from the mechanism table, not the metrics

```python
report = cv.run_eval(preds, golds, config, index=index)
print(report.errors.by_mechanism())
```

```
            mechanism   n
0  annotation_threshold   0
1      window_placement   1
2            misreading  23
3      missing_evidence   0
4     ontology_mismatch  12
5        temporal_error   7
6                 other   0
```

Read the shape before the counts. Each mechanism points at a different fix:

| Dominant mechanism | What to change |
| --- | --- |
| `misreading` | the prompt — usually negation or hedging |
| `temporal_error` | the prompt — add explicit temporal scoping |
| `ontology_mismatch` | the terminology mapping, not the model |
| `window_placement` | your pipeline's date handling |
| `annotation_threshold` | the annotation guideline, possibly nobody erred |
| `missing_evidence` | require evidence, or accept unsupported assertions |
| `other` (large) | the taxonomy is missing a mechanism for your data |

A zero means **checked and not found**, which is different from not checked.

## 2. Read actual examples

```python
report.errors.sample("misreading", n=5)[
    ["error_id", "system", "label", "error_type", "span_text", "mechanism_reason"]
]
```

```
error_id  system          label         error_type  span_text                              mechanism_reason
E000002   careful_reader  constipation  FP          no hard stools since the last visit    negation cue in quoted span: 'no'
E000014   negation_blind  diarrhea      FP          denies diarrhea or loose stools        negation cue in quoted span: 'denies'
```

Every tag carries the evidence that produced it. Do not trust a mechanism
summary you cannot drill into — read ten examples before concluding anything
about a category.

## 3. Find out which labels and windows carry the problem

```python
report.errors.by_label()      # label x mechanism
report.errors.by_window()     # window x mechanism
report.errors.by_system()     # system x error_type x mechanism
```

Errors concentrated in one label mean a label-specific prompt problem. Errors
concentrated in window 1 but not window 0 often mean follow-up notes are
written differently from diagnosis notes — a real finding about your corpus,
not about your model.

## 4. Separate the two directions

```python
errors = report.errors.df
fp = errors[errors["error_type"] == "FP"]     # over-calling
fn = errors[errors["error_type"] == "FN"]     # under-calling
match = errors[errors["error_type"] == "MATCH"]  # right label, wrong evidence
```

A system with many FPs and few FNs is over-eager — raise its threshold or fix
its negation handling. The reverse is conservative. A pile of `MATCH` rows
means the labels are right and the *evidence* is not, which matters if you plan
to show clinicians the quotes.

## 5. Check the confusion export by hand

```python
import pandas as pd
conf = pd.read_csv("report/confusion.csv")
conf[conf["cell"] == "FP"].head()
```

One row per `(system, unit_key, label)` with the 2×2 cell named, plus
`n_notes`. This is the file a reviewer asks for when they want to verify the
metrics themselves. A unit built from one positive note out of eight reads
differently from eight out of eight.

## 6. Compare systems on the units they both saw

```python
for row in report.comparisons:
    if row["metric"] == "cohen_kappa":
        print(row["system_a"], "-", row["system_b"],
              row["difference"], row["ci_low"], row["ci_high"], row["p_value"])
```

Paired on `unit_key`, so unit difficulty cancels. Do **not** compare two
independent intervals by eye — overlapping intervals do not imply no
difference, and non-overlapping ones overstate it.

If the paired interval straddles zero, the honest conclusion is that your
sample does not separate the systems. That is a finding about study size, not a
tie.

## 7. Put the numbers against the human ceiling

```python
report.iaa["pairs"]["rater_a|rater_b"]["cohen_kappa"]   # 0.813
report.metrics["careful_reader"].metrics["cohen_kappa"].point   # 0.912
```

A system at κ 0.70 is a different story when clinicians managed 0.72 with each
other than when they managed 0.95. If your system scores *above* the ceiling,
interrogate that — it often means the gold was adjudicated using a rule the
system also follows, so the comparison is not independent.

## 8. Re-tune only if you change the harness, not the model

If you change a prompt, you have a new system: re-run the whole protocol from
`tune`. If you change only a clinval threshold, you are tuning the harness, and
that still belongs on the dev split.

Either way, do not look at the test set between iterations. The frozen hash
cannot detect it.

## What the mechanism tags will not do

Negation and temporal cues are **conservative surface patterns**, used to
explain a disagreement after the fact. They never change a score, and they will
miss constructions they do not have patterns for — a clause-scoped negation, an
unusual hedge, a non-English note.

So a `misreading` count is a lower bound on misreadings, and `other` is where
the ones it could not explain end up. A rising `other` count is the signal to
extend the taxonomy for your corpus rather than to trust the breakdown.
