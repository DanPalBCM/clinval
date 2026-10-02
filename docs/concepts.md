# Units, windows, and alignment

## The pipeline

```
predictions.jsonl + annotations.jsonl + eval.yaml
        │
        ▼
   align_predictions ──► one row per (system, note_id, label)
        │
        ▼
  aggregate_to_windows ──► one row per (system, unit_key, label)
        │
        ├──► compute_agreement  → metrics with clustered CIs
        ├──► tag_errors         → mechanism per disagreement
        └──► summarize_cost     → tokens, latency, USD
        │
        ▼
   report.html · metrics.json · errors.csv
```

## Units

`EvalConfig.unit` picks the row metrics are computed over.

| Unit | One row per | Needs |
| --- | --- | --- |
| `note` | `(system, note_id, label)` | nothing extra |
| `patient` | `(system, patient_id, label)` | `patient_id` on records |
| `patient_window` | `(system, patient_id, window, label)` | `patient_id`, `note_datetime`, an index file |

`patient_window` is the default, because it is the unit a chart-review study
actually reports: *did this patient have diarrhea in the first six months after
diagnosis?*

Scoring at the note level answers a different and usually less interesting
question, and it inflates the apparent sample size — fifty notes from ten
patients are not fifty independent observations.

## The any-positive rule

A patient-window is positive for a label if **any** note in it is positive.
That matches how a reviewer answers the question: one documented episode is
enough.

The window also records `n_notes`, `n_pred_positive`, and `n_gold_positive`, so
a unit built from one positive note out of eight is distinguishable from one
built from eight out of eight.

## Windows

```yaml
window:
  size: 6M          # 30D, 2W, 6M, 1Y
  anchor: index_date
  n_windows: 2
  tie_break: earliest
  include_prior: false
```

**Intervals are half-open `[start, end)`.** A note timestamped exactly on a
boundary falls in the **later** window, deterministically and exactly once. So
with `size: 6M` anchored at 2022-01-01:

| Note timestamp | Window |
| --- | --- |
| 2021-12-31 | outside all windows |
| 2022-01-01 | 0 (the anchor is inside window 0) |
| 2022-06-30 | 0 |
| 2022-07-01 | **1** — boundaries belong to the later window |
| 2023-01-01 | outside all windows |

**Months and years are calendar-based**, not fixed multiples of 30 or 365 days.
One month after 2022-01-31 is 2022-02-28, because "six calendar months" is what
a protocol means. Days and weeks are exact.

`include_prior: true` adds one window immediately *before* the anchor, indexed
`-1`, for look-back questions.

`tie_break` picks which note represents a positive window — `earliest`,
`latest`, or `highest_confidence`. It affects the `representative_note_id`
reported in the error table, not the boolean value.

### Notes that do not land in a window

Kept, not dropped, with a non-empty `window_miss_reason`:

| Reason | Meaning |
| --- | --- |
| `outside_all_windows` | timestamped, but outside every window |
| `no_note_datetime` | no timestamp to place it with |
| `patient_not_in_index` | the patient has no index date |
| `no_patient_id` | the record carries no patient |

They are reported as unscored and the runner warns about them. Keeping them is
what makes `window_placement` errors detectable.

## Alignment

`align_predictions` joins predictions to gold on `(note_id, label)` over the
**union** of both sides, so nothing is silently dropped:

| Situation | `pred_present` | `gold_present` | Scored? |
| --- | --- | --- | --- |
| both sides have the pair | ✓ | ✓ | yes |
| prediction, no gold | ✓ | ✗ | no — nothing to be right about |
| gold, no prediction | ✗ | ✓ | no — **unscored, not a negative** |

A property test asserts that the row count equals the size of that union and
that every input record appears.

### Unscored is not negative

clinval cannot distinguish "the system processed this note and said no" from
"the system never saw this note". So a gold pair with no prediction is
**excluded** from the confusion table.

Because ignoring that silently flatters recall, `AlignedFrame.coverage()`
reports the share of gold pairs each system predicted on, and the runner emits
a warning naming the system and the percentage. Use
[`clinval negatives`](getting-started.md) to turn silence into explicit
negatives once you know the note universe.

### Gold resolution

One gold record per pair is required. Resolution order:

1. A record with `adjudicated: true` wins. Two of them for one pair raises.
2. Otherwise, if exactly one annotator covers the pair, that record wins.
3. Otherwise it raises — run `clinval adjudicate` first.

Step 3 is deliberate. Averaging two raters, or quietly picking one, would hide
the disagreement the study exists to measure. See
[decisions](decisions.md) D-008.

### Credited versus raw predictions

When `matching.span.require_span` is set, a positive prediction whose evidence
fails the span and code policy is **not credited** and scores as a negative.
The aligned frame keeps all three columns so the downgrade is auditable:

| Column | Meaning |
| --- | --- |
| `pred_raw` | what the system said |
| `credited` | whether its evidence satisfied the policy |
| `pred` | the effective value used for scoring |

With `require_span` off — the default — `pred` always equals `pred_raw`. This
is also what gives `min_overlap` an effect on the boolean metrics; see
[decisions](decisions.md) D-006.

## Idempotence

`aggregate_to_windows` detects already-aggregated input and returns it
unchanged, so running it twice is a genuine no-op rather than a second collapse
that would lose the original note ids. A property test covers this for every
unit.
