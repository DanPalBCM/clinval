# The runner

`run_eval` executes the whole pipeline in order and returns an `EvalReport`.
Every stage is also callable on its own; the runner exists so the common case
is one line and so the stages always run in the same order.

```python
import clinval as cv

report = cv.run_eval(
    "test_pred.jsonl",          # records or a path
    "test_gold.jsonl",
    config,                     # EvalConfig, FrozenConfig, path, or dict
    index="test_index.csv",     # required for unit=patient_window
    frozen=manifest,            # records the verified hash in the report
    annotator_records="two_raters.jsonl",   # for the human ceiling
    cost="cost.json",           # or a CostModel, or taken from the config
)
```

## Stage order

1. **align** — join predictions to gold on `(note_id, label)`
2. **aggregate** — collapse to the evaluation unit
3. **score** — metrics with intervals, per system, per label, per window
4. **tag errors** — a mechanism per disagreement
5. **account for cost** — tokens, latency, USD
6. **extras** — paired system comparisons, inter-annotator agreement
7. **report** — HTML, Markdown, JSON, CSVs

## What comes back

```python
report.metrics                 # MetricsTable: per-system AgreementReports
report.metrics.systems()       # ['careful_reader', 'negation_blind']
report.metrics["careful_reader"].metrics["cohen_kappa"]   # an Interval
report.metrics.to_markdown()
report.metrics.to_dataframe()  # long form: system, metric, point, ci_low, ci_high

report.errors                  # ErrorTable
report.cost                    # CostReport
report.aligned                 # note-level rows
report.windowed                # unit-level rows
report.comparisons             # paired bootstrap, every system pair x metric
report.iaa                     # inter-annotator agreement
report.warnings                # things to know before trusting the numbers
report.config_hash
```

## Writing artifacts

```python
report.write_all("report/")     # everything below
report.write_html("report/")
report.write_markdown("report.md")
report.write_json("metrics.json")
report.errors.to_csv("errors.csv")
```

`write_all` produces:

| File | Contents |
| --- | --- |
| `report.html` | static, self-contained, no JavaScript |
| `report.md` | the same report as Markdown |
| `metrics.json` | the §6.3 report schema plus clinval's audit fields |
| `metrics.csv` | long-form metrics for a spreadsheet |
| `errors.csv` | one row per tagged disagreement |
| `confusion.csv` | per-unit rows with the 2×2 cell named |
| `units.csv` | the full unit-level frame |
| `figures/*.png` | four matplotlib figures |

## Warnings are part of the output

The runner collects things a reader needs to know *before* reading the numbers,
and the report puts them above the metrics rather than in a footnote:

- a system whose coverage is below 100% — the missing pairs are unscored, not
  negatives
- units that could not be placed in a window, with the reasons
- a single annotator supplied where inter-rater agreement was requested

## The report is ordered caveats-first

The provenance block and any warnings come before the metrics, because an
unfrozen config or a coverage gap changes how every number below should be
read. An unfrozen run gets a prominent banner. A verified frozen run gets a
green confirmation with the hash and the commit.

Figures are static PNGs and the HTML contains no JavaScript, so a report can be
attached to a submission, opened from a file share, or archived without a
server.

## Guards

`unit=patient_window` without an index file raises `ConfigError` with the fix
in the message rather than failing deeper in aggregation:

```
unit='patient_window' needs per-patient index dates. Pass index=...
(a CSV with patient_id and index_date), or set unit: note in the config.
```

Multi-annotator gold passed as `annotations` raises — it must be adjudicated
first. Pass the unadjudicated records as `annotator_records` instead, which is
what the human-ceiling calculation wants.
