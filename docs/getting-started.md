# Getting started

## Install

clinval needs Python 3.11 or newer.

```bash
git clone https://github.com/DanPalBCM/clinval
cd clinval
uv sync
```

The optional LangExtract adapter pulls its own dependency:

```bash
uv sync --all-extras          # or: pip install 'clinval[langextract]'
```

## Run the demo

```bash
uv run python demo.py
```

This runs the whole protocol on the toy data committed in `examples/data/` and
writes `demo_output/report/report.html`. It finishes by tampering with the
frozen config and showing clinval refuse to run, because that refusal is the
package's central guarantee.

Expect roughly this, on 40 synthetic patients split 20/20:

```
  system            units                 kappa          sensitivity          specificity
  ---------------- ------  -------------------- -------------------- --------------------
  careful_reader       72   0.912 (0.767-1.000)  0.956 (0.852-0.988)  0.963 (0.817-0.993)
  negation_blind       72   0.397 (0.165-0.585)  0.889 (0.765-0.952)  0.481 (0.307-0.660)
```

`negation_blind` has a sensitivity of 0.889 and a kappa of 0.397. It fires on
any mention of a concept, including negated ones. Read sensitivity alone and it
looks usable — which is the point of including it.

## Your own data

### 1. Get your predictions into clinval's format

One JSONL record per (note, label). See [Data formats](formats.md).

```jsonc
{"system": "my-pipeline", "note_id": "N1", "label": "diarrhea", "value": true,
 "patient_id": "P1", "note_datetime": "2022-02-04T10:15:00",
 "span": {"start": 62, "end": 97, "text": "reports loose stools three times"}}
```

Emit **explicit negatives**. clinval reads silence as unscored, not as a
negative — it cannot tell a considered negative from an unprocessed note. If
your extractor only emits positives:

```bash
uv run clinval negatives --from pred.jsonl --notes notes.jsonl \
  --labels diarrhea,constipation --out pred_filled.jsonl
```

### 2. Check the files parse before running anything

```bash
uv run clinval validate --config eval.yaml --pred pred.jsonl \
  --gold gold.jsonl --index index.csv
```

Errors name the file and the 1-based line number.

### 3. Adjudicate multi-annotator gold

clinval refuses to evaluate against an unadjudicated two-rater file, because
averaging raters hides the disagreement a validation study exists to measure.

```bash
uv run clinval adjudicate --gold two_raters.jsonl --out gold.jsonl --rule agreement
```

The default `agreement` rule drops pairs the raters disagreed on and tells you
how many.

### 4. Split by patient, then run the protocol

Split **by patient**, never by note — two notes from one patient on opposite
sides of the split means you tuned on a patient you later tested on.

```bash
uv run clinval tune     --config eval.yaml --pred dev_pred.jsonl  --gold dev_gold.jsonl  --index dev_index.csv  --out tuned.yaml
uv run clinval freeze   --config tuned.yaml --out frozen.json
uv run clinval evaluate --frozen frozen.json --pred test_pred.jsonl --gold test_gold.jsonl --index test_index.csv --dev-gold dev_gold.jsonl --out report/
```

Passing `--dev-gold` makes `evaluate` check that the splits are disjoint and
exit 2 if they are not.

## Commands

| Command | What it does |
| --- | --- |
| `clinval synth` | generate a deterministic synthetic dataset |
| `clinval validate` | parse and sanity-check inputs, then stop |
| `clinval negatives` | materialize explicit negatives for unmentioned pairs |
| `clinval adjudicate` | collapse multi-annotator gold to a single standard |
| `clinval tune` | grid-search declared parameters on dev |
| `clinval freeze` | lock a config and record its hash |
| `clinval evaluate` | run a frozen config on test data |
| `clinval run` | single-shot unfrozen run (needs `--i-understand-this-is-dev`) |
| `clinval consensus` | gate two or more systems and route disagreements |

## Limitations

- **clinval measures agreement. It does not certify clinical validity.**
- **Boolean labels only** in v1, optionally with a span and a code. No
  categorical values, no relation extraction.
- **It cannot verify your splits.** `check_split_disjoint` reports overlap, but
  clinval only sees the files you hand it. A frozen hash verifies the config,
  not your study design.
- **It cannot verify your gold.** Annotation quality is outside its reach.
- **The bundled ontology map is demo scope** — six concepts. Supply your own
  with `load_ontology_map`.
- **Error mechanism tags are heuristics.** Negation and temporal cues are
  surface patterns used to explain a disagreement after the fact; they never
  change a score.
- **Cost depends on what your extractor reports.** See
  [decisions](decisions.md) D-003 and D-004.
- **Bootstrap p-values are approximations**, not exact tests.
