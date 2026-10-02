# Tune, freeze, evaluate

The full protocol on the toy data in this repository. Everything below is
reproducible with `uv run python demo.py`.

## Generate a dataset

```bash
uv run clinval synth --out demo/ --n-patients 200 --labels diarrhea,constipation --seed 0
```

This writes dev and test halves **split by patient**, plus an `eval.yaml` to
start from:

```
demo/dev_pred.jsonl      demo/test_pred.jsonl
demo/dev_gold.jsonl      demo/test_gold.jsonl
demo/dev_index.csv       demo/test_index.csv
demo/dev_notes.jsonl     demo/test_notes.jsonl
demo/dev_gold_two_annotators.jsonl
demo/eval.yaml
```

## 1. Check the inputs parse

```bash
uv run clinval validate --config demo/eval.yaml --pred demo/dev_pred.jsonl \
  --gold demo/dev_gold.jsonl --index demo/dev_index.csv
```

```
ok config: unit=patient_window, split=dev, 9 metric(s), hash 5f2a9c81b4e3…
ok predictions: 356 records, 2 system(s) (careful_reader, negation_blind), 2 label(s)
ok annotations: 178 records, 1 annotator(s) (adjudicated), 178 adjudicated
ok index dates: 20 patient(s), 2021-01-13 to 2023-06-21
```

## 2. Tune on dev

```bash
uv run clinval tune --config demo/eval.yaml \
  --pred demo/dev_pred.jsonl --gold demo/dev_gold.jsonl \
  --index demo/dev_index.csv --out demo/tuned.yaml
```

```
Objective: cohen_kappa, maximized over 5 candidates.

| min_overlap | cohen_kappa | chosen |
| --- | --- | --- |
| 0.2  | 0.7018 | yes |
| 0.35 | 0.7018 |     |
| 0.5  | 0.6609 |     |
| 0.65 | 0.6609 |     |
| 0.8  | 0.5970 |     |

chosen {'min_overlap': 0.2} → cohen_kappa=0.7018
```

Two things to read here. The grid **discriminates** — 0.70 down to 0.60 — so
tuning bought something. And 0.2 and 0.35 tie, which is worth knowing: the
choice between them is arbitrary on this dev set even though the grid overall
is not flat.

!!! note "If the whole grid is flat"
    clinval says so, and you should report the value as a default rather than
    as a tuned finding. The usual cause is that `require_span` is off, in which
    case span thresholds have no effect on the boolean metrics at all. See
    [decisions](../decisions.md) D-006.

## 3. Freeze

```bash
uv run clinval freeze --config demo/tuned.yaml --out demo/frozen.json
```

```
frozen → demo/frozen.json
  config hash: 05100d073c1cf06904676ca5bf6858b998af191c0149341913644b76f4ff2ec4
  frozen at:   2026-10-02T16:21:43.918204+00:00
  git commit:  a1b2c3d4e5f6
```

The manifest carries the config, the hash, the timestamp, the clinval version,
the commit, and the full tuning record — every candidate, not just the winner.

## 4. Evaluate out of sample

```bash
uv run clinval evaluate --frozen demo/frozen.json \
  --pred demo/test_pred.jsonl --gold demo/test_gold.jsonl \
  --index demo/test_index.csv \
  --dev-gold demo/dev_gold.jsonl \
  --raters demo/test_gold_two_annotators.jsonl \
  --out demo/report/
```

`--dev-gold` checks the splits are disjoint and exits 2 if they are not.
`--raters` adds the human ceiling.

```
  system            units                 kappa          sensitivity          specificity
  ---------------- ------  -------------------- -------------------- --------------------
  careful_reader       72   0.912 (0.767-1.000)  0.956 (0.852-0.988)  0.963 (0.817-0.993)
  negation_blind       72   0.397 (0.165-0.585)  0.889 (0.765-0.952)  0.481 (0.307-0.660)
```

Open `demo/report/report.html`.

## What to actually look at

**Sensitivity in isolation is misleading.** `negation_blind` finds 88.9% of
true positives. Its specificity is 0.481 and its κ is 0.397, because it fires
on any mention including negated ones.

**Compare against the human ceiling, not against 1.0.** The two annotators
agreed at κ 0.813. `careful_reader` scores 0.912 — *above* the ceiling, which
is itself worth interrogating rather than celebrating.

**Use the paired comparison, not two separate intervals.**

```
paired bootstrap on 72 shared units: difference 0.515 (95% CI 0.268 to 0.776, p = 0.000)
```

Both systems are scored on the identical units in every resample, so unit
difficulty cancels.

**Read the mechanism table before the metrics.**

```
  annotation_threshold    none found
  window_placement                 1
  misreading                      23
  missing_evidence        none found
  ontology_mismatch               12
  temporal_error                   7
  other                   none found
```

23 misreadings and 12 code mismatches point at two different fixes: the first
is a prompt problem, the second a terminology-mapping problem. `none found`
means checked and absent — not unchecked.

## 5. Confirm the guarantee holds

Edit the frozen config and try again:

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("demo/frozen.json")
raw = json.loads(p.read_text())
raw["config"]["matching"]["span"]["min_overlap"] = 0.01
p.write_text(json.dumps(raw))
PY

uv run clinval evaluate --frozen demo/frozen.json --pred demo/test_pred.jsonl \
  --gold demo/test_gold.jsonl --index demo/test_index.csv --out demo/report2/
```

```
error: frozen config failed verification: the config body was modified after freezing.
  recorded hash: 05100d073c1cf06904676ca5bf6858b998af191c0149341913644b76f4ff2ec4
  actual hash:   eaed509277aa24797c41cc77439cf358010734dde67454709e36693353557ca7
Re-run `clinval freeze` if the change was intentional.
```

Exit code 1, no report written.

## In Python

```python
import clinval as cv

data = cv.synth.generate(n_patients=200, labels=("diarrhea", "constipation"), seed=0)
dev, test = data.split_by_patient(0.5)

config = cv.EvalConfig.from_yaml("eval.yaml")

tuning = cv.tune(config, dev.all_predictions, dev.annotations, index=dev.index)
print(tuning.to_markdown())

frozen = cv.freeze(tuning.config, tuning=tuning)
cv.write_frozen(frozen, "frozen.json")

manifest = cv.load_frozen("frozen.json")        # verifies the hash
report = cv.run_eval(
    test.all_predictions, test.annotations, manifest.config,
    index=test.index, frozen=manifest,
    annotator_records=[*test.second_annotator, *test.annotations],
)
report.split_check = cv.check_split_disjoint(dev.annotations, test.annotations)
report.write_all("report/")
```
