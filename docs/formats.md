# Data formats

Record files are **JSONL**: one JSON object per line, no wrapping array. Errors
report the 1-based line number, because the whole point of a validation harness
is that you can find the bad row.

Blank lines and lines starting `//` are skipped.

## Prediction

One record per (note, label). Unknown fields are **rejected**, so a typo in a
field name is caught rather than silently ignored.

```jsonc
{
  "system": "my-pipeline",          // required
  "note_id": "N1",                  // required
  "label": "diarrhea",              // required; opaque to clinval
  "value": true,                    // required; boolean in v1

  "patient_id": "P1",               // required for unit=patient_window
  "note_datetime": "2022-02-04T10:15:00",  // required for window assignment

  "span": {"start": 62, "end": 97, "text": "reports loose stools three times"},
  "code": {"system": "HPO", "code": "HP:0002014", "display": "Diarrhea"},
  "confidence": 0.91,

  "run": {
    "model": "gpt-x",
    "prompt_version": "v1",
    "tokens_in": 742,
    "tokens_out": 58,
    "latency_s": 1.84,
    "call_id": "N1"
  }
}
```

Uniqueness key: `(system, note_id, label)`. A repeat raises
`DuplicateRecordError` rather than a silent last-one-wins, because which copy
survived would change the metrics. The same (note, label) from *different*
systems is fine and expected.

### `label` is opaque

clinval never splits or interprets `label`. It may carry structure clinval does
not know about — `fhirground` emits `Condition:<code or text>` — so colons and
spaces are permitted and survive a round trip intact.

### `span`

Python slice semantics: `text[start:end]`. `start == end` is a zero-length span,
allowed but never matching another span under `exact` or `overlap`. An inverted
span (`end < start`) is rejected.

### `run.call_id`

One model call usually produces **several** records — a single pass over a note
typically answers every label at once. Repeating that call's token count on
each record and summing them multiplies the true cost by the number of labels.

Set `call_id` to something stable per model call (the note id is the usual
choice) and clinval counts each call once for both cost and latency. Records
without a `call_id` are charged individually. See
[decisions](decisions.md) D-004.

## Annotation

Identical to Prediction except:

- `system` → **`annotator`**
- `confidence` and `run` are **absent** (a human annotator has no token count,
  and the schema enforces it)
- **`adjudicated: boolean`** is added

```jsonc
{"annotator": "adjudicated", "note_id": "N1", "label": "diarrhea",
 "value": true, "patient_id": "P1",
 "span": {"start": 62, "end": 97, "text": "reports loose stools three times"},
 "note_datetime": "2022-02-04T10:15:00", "adjudicated": true}
```

Uniqueness key: `(annotator, note_id, label)`. Two annotators per pair is legal
and is the input to inter-rater agreement — but evaluating against an
unadjudicated two-rater file **raises**. Run `clinval adjudicate` first; see
[decisions](decisions.md) D-008.

## Index dates (CSV)

One row per patient. Several common column names are accepted so an index file
exported from an OMOP or MEDS pipeline works without renaming:

| Role | Accepted names |
| --- | --- |
| patient | `patient_id`, `subject_id`, `person_id`, `patient` |
| date | `index_date`, `index_datetime`, `anchor_date`, `date` |

```csv
patient_id,index_date
P00001,2022-01-15
P00002,2021-06-01
```

Timezones are stripped. More than one row per patient raises — window
assignment needs a single anchor.

## Notes (JSONL)

Only needed by `clinval negatives`, to define the note universe:

```jsonc
{"note_id": "N1", "patient_id": "P1", "note_datetime": "2022-02-04T10:15:00",
 "text": "Routine follow-up visit. Patient reports loose stools..."}
```

## Config (YAML)

```yaml
split: dev                  # dev | test
unit: patient_window        # note | patient | patient_window
labels: [diarrhea, constipation]   # empty means every label in gold

matching:
  span:
    mode: overlap           # exact | overlap | fuzzy
    min_overlap: 0.5
    fuzzy_threshold: 0.85
    require_span: true
  ontology:
    hpo: {fuzzy: false}
    icd-10: {prefix_depth: 3}

window:
  size: 6M                  # 30D | 2W | 6M | 1Y
  anchor: index_date
  n_windows: 2
  tie_break: earliest       # earliest | latest | highest_confidence
  include_prior: false

metrics: [cohen_kappa, pabak, sensitivity, specificity, ppv, npv, accuracy, f1, prevalence]

bootstrap:
  n: 1000
  cluster_by: patient_id
  seed: 0
  alpha: 0.05

cost:                       # USD per 1,000 tokens
  input_per_1k: 0.003
  output_per_1k: 0.015

consensus: majority         # majority | unanimous | any

freeze:
  objective: cohen_kappa
  grid:
    min_overlap: [0.2, 0.35, 0.5, 0.65, 0.8]
```

Unknown keys and unknown metric names are rejected at load time. The Python API
also accepts a flat shorthand:

```python
cv.Matching(span="overlap", min_overlap=0.5)
```

## Freeze manifest (JSON)

Written by `clinval freeze`, verified by `clinval evaluate`.

```jsonc
{
  "config": { /* the full EvalConfig */ },
  "config_hash": "05100d07...",     // SHA-256 of the canonical config body
  "frozen_at": "2026-10-02T16:21:43.918Z",
  "git_commit": "a1b2c3d...",       // null outside a repository
  "clinval_version": "1.0.0",
  "tuning_summary": { /* every candidate and its score */ }
}
```

## Eval report (JSON)

```jsonc
{
  "config_hash": "...",
  "frozen": {"config_hash": "...", "frozen_at": "...", "git_commit": "...", "verified": true},
  "split": "test",
  "unit": "patient_window",
  "units": 72,
  "systems": ["careful_reader", "negation_blind"],
  "metrics": { /* first system, for the §6.3 shape */ },
  "metrics_by_system": { /* all of them */ },
  "cis": { /* per system, per metric */ },
  "per_label": [ /* one row per system x label */ ],
  "error_counts": {"misreading": 23, "ontology_mismatch": 12},
  "error_counts_full": [ /* every mechanism, including zeros */ ],
  "cost": { /* rates, per-system totals, any_unknown */ },
  "comparisons": [ /* paired bootstrap per system pair x metric */ ],
  "iaa": { /* pairwise annotator agreement */ },
  "split_check": { /* dev/test disjointness */ },
  "coverage": {"careful_reader": 1.0},
  "warnings": [ ... ]
}
```

An undefined number is `null`, never `0`.

## Error export (CSV)

```
error_id, system, label, unit_key, error_type, patient_id, window,
mechanism, span_text, note_id, mechanism_reason, gold_span_text,
span_jaccard, pred_code, gold_code, code_how, confidence
```

The first ten are the spec's §6.4 schema; the rest are the audit trail. See
[the error taxonomy](errors.md).
