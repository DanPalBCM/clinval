# Evaluating fhirground output

[fhirground](https://github.com/DanPalBCM/fhirground) extracts grounded,
validated FHIR R4 resources from clinical notes and exports
`clinval_predictions.jsonl` directly. There is **no translation step**.

```bash
fhirground extract --notes notes.jsonl --out out/ --format bundle,ndjson,clinval,debug

uv run clinval validate --pred out/clinval_predictions.jsonl --gold gold.jsonl --index index.csv

uv run clinval tune     --config eval.yaml --pred dev/clinval_predictions.jsonl  --gold dev_gold.jsonl  --index dev_index.csv  --out tuned.yaml
uv run clinval freeze   --config tuned.yaml --out frozen.json
uv run clinval evaluate --frozen frozen.json --pred test/clinval_predictions.jsonl --gold test_gold.jsonl --index test_index.csv --out report/
```

## The contract

| clinval field | fhirground value |
| --- | --- |
| `system` | the fhirground config name |
| `label` | `<ResourceType>:<code or text>`, e.g. `Condition:K59.1` |
| `span` | from the mention's evidence quote |
| `code` | resolved by the terminology layer, never emitted by the LLM |
| `confidence` | the per-resource confidence score |
| `run` | model, prompt version, token counts, latency |

## Resource-qualified labels work unchanged

`label` is an **opaque string** in clinval. It is never split, parsed, or
interpreted, so `Condition:K59.1` survives intact. This is the part most likely
to break a naive consumer, and a test asserts it:

```python
preds = cv.read_predictions("clinval_predictions.jsonl")
{p.label for p in preds}      # {'Condition:K59.1', 'Condition:K59.0'}
```

One consequence worth planning for: because the label embeds the code, the
label space is **high-cardinality**. A cohort with two hundred distinct
conditions produces two hundred labels, most with a handful of positives each.
Per-label metrics on five positives are not informative — set `labels:` in the
config to the conditions your study is actually about, and let clinval ignore
the rest.

## Codes come from the resolver, not the model

fhirground's hard rule is that the LLM never emits a medical code: the model
extracts plain-language mentions with evidence quotes, and codes come only from
the terminology resolver.

That changes how to read an `ontology_mismatch` tag. The model did not
hallucinate a code — it cannot. A mismatch means the **resolver** mapped a
mention to a different concept than the annotator did, which is a
terminology-mapping problem rather than a prompting one. Use ICD-10 prefix
matching when category-level agreement is what you care about:

```yaml
matching:
  ontology:
    icd-10:
      prefix_depth: 3     # E11.9 and E11.65 both reduce to E11
```

## Cost is complete here

Unlike langextract, fhirground reports token counts, so clinval produces a real
cost rather than `unknown`:

```
| system            | records | token coverage | tokens in | tokens out | cost (USD) |
| fhirground-demo   |       6 |           100% |      4258 |        290 |     0.0171 |
```

If fhirground emits several labels from one model call, make sure the records
share a `run.call_id` or the cost is multiplied by the number of labels. See
[decisions](../decisions.md) D-004.

## Patient-window evaluation

fhirground passes `patient_id` and `note_datetime` through from the source
notes, so `unit: patient_window` works as long as you supply index dates:

```yaml
unit: patient_window
window:
  size: 6M
  anchor: index_date
  n_windows: 2
```

If the notes lack a patient id, clinval raises with the fix in the message
rather than failing deeper in aggregation. Fall back to `unit: note` when you
are evaluating extraction quality rather than a patient-level phenotype.

## The fixture

`tests/fixtures/fhirground_clinval_predictions.jsonl` is built to this contract
and is what `tests/test_integration.py::TestFhirgroundInterop` runs against.
Until fhirground ships, it documents exactly what clinval expects. When
fhirground exists, point that test at its real output; see
[decisions](../decisions.md) D-012.
