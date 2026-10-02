# clinval

**The missing evaluation layer for clinical LLM extraction.**

[![tests](https://github.com/DanPalBCM/clinval/actions/workflows/ci.yml/badge.svg)](https://github.com/DanPalBCM/clinval/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

LLM extraction is easy to demo and hard to trust. Given note-level predictions
from any extractor, clinval answers the questions a reviewer will actually ask:
which patient-level phenotypes are supported, how well does the system agree
with clinician annotators, and *where do the residual errors come from*.

It is deliberately not an eval framework and not an extraction library:

| | What it scores | What it knows about |
| --- | --- | --- |
| **LLM eval frameworks** (promptfoo, DeepEval, Inspect) | prompts and responses | nothing about patients, time windows, or inter-rater agreement |
| **Extraction libraries** (LangExtract, custom pipelines) | nothing — they stop at extraction | spans and schemas |
| **clinval** | patient-by-time-window phenotypes | agreement with CIs, error mechanisms, the dev/test protocol |

**clinval never calls an LLM.** It consumes an extractor's output, which is what
makes it extractor-agnostic and what makes its results reproducible.

---

## The frozen dev protocol

Threshold tuning is the easiest way to overfit a validation study without
noticing. clinval makes the protocol mechanical rather than a matter of
discipline:

```
          dev split                                  test split
         ┌───────────┐                              ┌───────────┐
         │ dev_pred  │                              │ test_pred │
         │ dev_gold  │                              │ test_gold │
         └─────┬─────┘                              └─────┬─────┘
               │                                          │
               ▼                                          │
     ┌───────────────────┐                                │
     │  clinval tune     │  grid-search the declared       │
     │                   │  parameters; refuses a config   │
     │                   │  marked `split: test`           │
     └─────────┬─────────┘                                │
               ▼                                          │
     ┌───────────────────┐                                │
     │  clinval freeze   │  SHA-256 over the config body,  │
     │                   │  + timestamp + git commit       │
     └─────────┬─────────┘                                │
               │  frozen.json                             │
               ▼                                          ▼
     ┌──────────────────────────────────────────────────────────┐
     │  clinval evaluate   recomputes the hash and REFUSES to   │
     │                     run if the config was edited after   │
     │                     freezing                             │
     └──────────────────────────┬───────────────────────────────┘
                                ▼
                 report.html · metrics.json · errors.csv
```

Three guards, all enforced in code:

1. `tune` refuses a config whose `split` is `test`.
2. `freeze` records a hash of the config body.
3. `evaluate` verifies that hash, and refuses a config marked `test` that was
   never frozen. `clinval run` will do an unfrozen run only behind
   `--i-understand-this-is-dev`, because an unfrozen number is not a validation
   result.

---

## Quickstart

```bash
git clone https://github.com/DanPalBCM/clinval
cd clinval
uv sync
uv run python demo.py
```

The demo runs the whole protocol on the toy data committed in
`examples/data/`, then opens with a flourish: it tampers with the frozen config
and shows clinval refusing to run. Output lands in
`demo_output/report/report.html`.

Or run the protocol by hand:

```bash
# generate a synthetic dataset, split by patient
uv run clinval synth --out demo/ --n-patients 200 --labels diarrhea,constipation --seed 0

# tune on dev, freeze, evaluate out of sample
uv run clinval tune     --config demo/eval.yaml --pred demo/dev_pred.jsonl  --gold demo/dev_gold.jsonl  --index demo/dev_index.csv  --out demo/tuned.yaml
uv run clinval freeze   --config demo/tuned.yaml --out demo/frozen.json
uv run clinval evaluate --frozen demo/frozen.json --pred demo/test_pred.jsonl --gold demo/test_gold.jsonl --index demo/test_index.csv --dev-gold demo/dev_gold.jsonl --out demo/report/
```

### What the demo reports

Real output from `demo.py` on the committed toy data — 40 synthetic patients
split 20/20 by patient, two mock systems with deliberately different
pathologies:

```
  system            units                 kappa          sensitivity          specificity
  ---------------- ------  -------------------- -------------------- --------------------
  careful_reader       72   0.912 (0.767-1.000)  0.956 (0.852-0.988)  0.963 (0.817-0.993)
  negation_blind       72   0.397 (0.165-0.585)  0.889 (0.765-0.952)  0.481 (0.307-0.660)

  paired bootstrap on 72 shared units: difference 0.515 (95% CI 0.268 to 0.776, p = 0.000)
  human ceiling — annotators rater_a|rater_b agreed at kappa 0.813 over 142 units
```

Note what that last line does. `negation_blind` has a **sensitivity of 0.889** —
it finds almost everything. Read sensitivity alone and it looks usable. Its
specificity is 0.481, because it fires on any mention including negated ones,
and its kappa is 0.397. And `careful_reader`'s 0.912 sits *above* the 0.813 the
two human annotators managed with each other, which is the number you should be
comparing against rather than 1.0.

Then the part that makes it actionable:

```
  Why the disagreements happened
    annotation_threshold    none found
    window_placement                 1  █
    misreading                      23  █████████████████████
    missing_evidence        none found
    ontology_mismatch               12  ███████████
    temporal_error                   7  ███████
    other                   none found

  One tagged error, in full
    id        E000002  (FP, misreading)
    system    careful_reader  label constipation
    evidence  'no hard stools since the last visit'  (system quote)
    why       negation cue in quoted span: 'no'
```

Every tag carries the evidence that produced it. A mechanism showing
`none found` was checked and not found — distinguishable from not checked.

---

## Worked example (Python API)

```python
import clinval as cv

preds = cv.read_predictions("pred.jsonl")
golds = cv.read_annotations("gold.jsonl")
index = cv.read_index_dates("index.csv")

config = cv.EvalConfig(
    unit="patient_window",
    window=cv.Window(size="6M", anchor="index_date"),
    matching=cv.Matching(span="overlap", min_overlap=0.5),
)

report = cv.run_eval(preds, golds, config, index=index, cost="cost.json")
print(report.metrics.to_markdown())
report.errors.to_csv("errors.csv")
report.write_html("report/")
```

And a consensus comparison between two systems:

```python
cons = cv.consensus_gate(
    {"sysA": preds_a, "sysB": preds_b}, golds, rule="majority", index=index
)
print(cons.to_markdown())
cons.to_review_queue().to_csv("review_queue.csv")   # most-contested first
```

---

## Design principles

**Extractor-agnostic.** Anything that emits clinval's JSONL format can be
evaluated. The bundled LangExtract adapter is a thin translator that lives
outside the core; it is the only code in the package that touches a model.

**The patient is the unit.** Metrics are computed on patient-by-time-window
rows, with bootstrap intervals **clustered by patient**. Two windows from the
same patient are not independent observations, and resampling rows instead of
patients produces intervals that are too narrow. Window intervals are half-open
`[start, end)`, so a note on a boundary is counted once, in the later window.

**Unscored is not negative.** If a system emitted no record for a gold pair,
clinval cannot tell "it considered this note and said no" from "it never saw
this note". Those rows are excluded and counted in `coverage`, and the report
warns about them. `clinval negatives` is the opt-in for callers who know the
note universe. Reading silence as a negative would silently flatter recall.

**Kappa, PABAK, and prevalence travel together.** A rare phenotype can show 97%
accuracy and a kappa near zero. Reporting only the first is the standard way
clinical extraction results get oversold, so clinval computes all three and
prints prevalence beside them.

**Undefined is `n/a`, not zero.** A specificity with no gold negatives is
undefined. Printing `0.000` would read as "the system got every negative
wrong", which is a different and false claim.

**Errors are first-class.** Every disagreement gets a mechanism from a fixed
taxonomy — `annotation_threshold`, `window_placement`, `misreading`,
`missing_evidence`, `ontology_mismatch`, `temporal_error`, `other` — with a
`mechanism_reason` naming the evidence. A tag you cannot audit is worse than no
tag.

**Interval methods are named per row.** Wilson for metrics that are proportions
of a fixed denominator, clustered percentile bootstrap for kappa, PABAK, and
F1. A reader comparing two intervals should know they came from different
machinery.

---

## Limitations

Stated plainly, because a validation harness that oversells itself is
self-defeating.

- **clinval measures agreement. It does not certify clinical validity.** No
  number it prints licenses a claim about patient benefit.
- **Boolean labels only.** v1 scores a boolean per (note, label), optionally
  with a span and a code. No categorical values, no relation extraction, no
  multi-label structured output.
- **It cannot verify your splits.** `check_split_disjoint` *reports* whether dev
  and test share patients, but clinval only sees the files you hand it. If you
  tuned on a patient you later tested on, the frozen hash will still verify.
- **It cannot verify your gold.** Adjudication quality, annotation guidelines,
  and whether the gold standard means what you think are outside its reach.
- **The bundled ontology map is demo scope** — six concepts, for the tests and
  the walkthrough. It is not a terminology service. Supply your own crosswalk
  with `load_ontology_map`.
- **Error mechanism tags are heuristics.** Negation and temporal cues are
  surface patterns used to *explain* a disagreement after the fact. They never
  change a score. A large `other` count means the taxonomy is missing a
  mechanism for your data.
- **Cost depends on what your extractor reports.** langextract exposes no token
  counts at all, so its cost is reported as `unknown` rather than zero (see
  [`docs/decisions.md`](docs/decisions.md) D-003). Set `RunMeta.call_id` if one
  model call produces several records, or the cost is multiplied by the number
  of labels.
- **Bootstrap p-values are approximations**, not exact tests.
- **One anchor per patient**, and `index_date` is the only window anchor in v1.

---

## Data formats

Predictions and annotations are JSONL, one object per line. Errors report the
1-based line number, because the point of a validation harness is that you can
find the bad row.

```jsonc
// prediction
{"system": "sysA", "note_id": "N1", "label": "diarrhea", "value": true,
 "patient_id": "P1", "note_datetime": "2022-02-04T10:15:00",
 "span": {"start": 62, "end": 97, "text": "reports loose stools three times"},
 "code": {"system": "HPO", "code": "HP:0002014", "display": "Diarrhea"},
 "confidence": 0.91,
 "run": {"model": "gpt-x", "prompt_version": "v1", "tokens_in": 742,
         "tokens_out": 58, "latency_s": 1.84, "call_id": "N1"}}

// annotation: `annotator` replaces `system`, `adjudicated` is added,
// `confidence` and `run` are absent
{"annotator": "adjudicated", "note_id": "N1", "label": "diarrhea",
 "value": true, "patient_id": "P1", "adjudicated": true}
```

`label` is an **opaque string**. clinval never splits or interprets it, so
`fhirground`'s `Condition:K59.1` works unchanged — see
`tests/test_integration.py::TestFhirgroundInterop`.

Full schema reference: [`docs/formats.md`](docs/formats.md).

---

## Interoperability

clinval is one of three projects that share a data contract:

- **[fhirground](https://github.com/DanPalBCM/fhirground)** — grounded FHIR R4
  extraction from clinical notes. Exports `clinval_predictions.jsonl`, so
  clinval evaluates its output with no translation step.
- **clinval** (this project) — the validation harness.
- **[ehrlint](https://github.com/DanPalBCM/ehrlint)** — leakage auditor for EHR
  prediction tasks.

A LangExtract adapter ships in `clinval.plugins.langextract`, verified against
`langextract` 1.7.0.

---

## No real data, anywhere

Every dataset in this repository is **generated**. `clinval.synth` assembles
notes from a fixed sentence bank with sequential identifiers (`P00001`,
`N0000001`) and dates derived from a constant epoch. There is no PHI to leak
because there was never any source to leak from, and a test asserts that every
sentence in a generated note came from that bank.

No test makes a network call. The one test that would reach a model is marked
`live_llm` and skipped unless you pass `--run-live-llm`.

---

## Development

```bash
uv sync                  # install, including dev dependencies
uv run pytest            # 632 tests, ~96% coverage
uv run pytest --cov=clinval --cov-report=term-missing
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Agreement numbers are checked against reference implementations:
Cohen's kappa against `sklearn.metrics.cohen_kappa_score` over 200 randomized
tables, Wilson intervals against `statsmodels.stats.proportion.proportion_confint`.
Property tests (hypothesis) cover alignment completeness, aggregation
idempotence, interval coverage, and freeze-hash sensitivity.

Every decision that departs from the specification, resolves an ambiguity, or
rests on something verified about an external library is recorded in
[`docs/decisions.md`](docs/decisions.md).

---

## Citing

See [`CITATION.cff`](CITATION.cff).

```bibtex
@software{clinval,
  author  = {Palacios, Daniel},
  title   = {clinval: a validation harness for clinical LLM extraction},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/DanPalBCM/clinval}
}
```

## License

[Apache-2.0](LICENSE). Chosen for the patent grant and its acceptability in
health-enterprise settings, consistent with the sibling projects.
