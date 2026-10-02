# Evaluating a LangExtract pipeline

clinval ships a reference adapter for [LangExtract](https://github.com/google/langextract),
verified against **1.7.0**. It is an optional extra:

```bash
uv sync --all-extras        # or: pip install 'clinval[langextract]'
```

## What you need to know first

Four properties of langextract shape this adapter, and none of them is hidden
from you.

### 1. It reports no token counts, so cost is `unknown`

langextract exposes **no** usage metadata anywhere. `ScoredOutput` is
`(score, output)`; nothing in the package records prompt or completion tokens.

So the adapter leaves `tokens_in` and `tokens_out` unset, and clinval reports
cost as *unknown* rather than zero — a zero would make the least-instrumented
system look cheapest, which is exactly backwards. `latency_s` **is** populated,
measured by clinval with a wall clock.

```
| system      | records | token coverage | tokens in | cost (USD) | mean latency (s) |
| langextract |      40 |            n/a |         0 |    unknown |            2.411 |
```

A guard test fails if a future langextract release adds usage metadata, which
is the signal to start populating those fields. See
[decisions](../decisions.md) D-003.

### 2. Extractions are spans, not booleans

A `data.Extraction` has an `extraction_class` and quoted text; there is no truth
value. The adapter applies an explicit **presence rule** — an extraction of a
label's class means `value=True` — and materializes explicit `False` records
for every target label a note did not produce. That matters because clinval
reads silence as *unscored*, not as a negative.

### 3. Negation is not handled by the library

`"denies diarrhea"` extracts as class `diarrhea` unless your prompt asks for
polarity. The adapter reads a configurable attribute and **never guesses from
the text** — guessing silently would turn an extraction bug into a clinval
scoring bug.

Ask for polarity in the prompt:

```python
prompt_description = (
    "Extract symptom mentions. For each, add a `negated` attribute that is "
    "true when the note denies or excludes the symptom."
)
```

Without it, the adapter warns and counts every extraction as asserted:

> 12 extraction(s) carried no readable 'negated' attribute, so they were
> counted as asserted. langextract does not detect negation on its own.

### 4. Offsets may be approximate

`CharInterval` is `start_pos`/`end_pos`, either of which may be `None`, and
`alignment_status` can be `MATCH_FUZZY`, `MATCH_GREATER`, or `MATCH_LESSER`.
Evaluating those with `mode: exact` measures the aligner rather than the model,
so the adapter warns:

> langextract reported non-exact span alignment for at least one extraction.
> Evaluating these with span matching mode 'exact' measures the aligner rather
> than the model; use mode 'overlap' or 'fuzzy'.

Use `overlap` or `fuzzy`. The status is recorded on each prediction's `run`
metadata so you can audit it.

## Running it

```python
import langextract as lx
import clinval as cv
from clinval.plugins.base import Note
from clinval.plugins.langextract import LangExtractConfig, LangExtractPlugin

notes = [
    Note(note_id="N1", text=note_text, patient_id="P1", note_datetime=dt),
    # ...
]

plugin = LangExtractPlugin(
    LangExtractConfig(
        labels=("diarrhea", "constipation"),   # explicit negatives for these
        negation_attribute="negated",
        system="langextract-v1",
        prompt_version="symptoms-v1",
    ),
    prompt_description=(
        "Extract symptom mentions. For each, add a `negated` attribute that is "
        "true when the note denies or excludes the symptom."
    ),
    examples=[
        lx.data.ExampleData(
            text="Patient reports loose stools.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="diarrhea",
                    extraction_text="loose stools",
                    char_interval=lx.data.CharInterval(start_pos=16, end_pos=28),
                    attributes={"negated": "false"},
                )
            ],
        )
    ],
    model_id="gemini-3.5-flash",
)

preds = list(plugin.predict(notes))     # reaches the network; needs an API key
cv.write_predictions(preds, "pred.jsonl")
```

Then evaluate exactly as you would any other system:

```bash
uv run clinval tune     --config eval.yaml --pred dev_pred.jsonl --gold dev_gold.jsonl --index dev_index.csv --out tuned.yaml
uv run clinval freeze   --config tuned.yaml --out frozen.json
uv run clinval evaluate --frozen frozen.json --pred test_pred.jsonl --gold test_gold.jsonl --index test_index.csv --out report/
```

## Configuration

| Field | Default | Purpose |
| --- | --- | --- |
| `labels` | required | target label set; explicit negatives are emitted for all of them |
| `class_to_label` | `{}` | map `extraction_class` to a clinval label when the names differ |
| `negation_attribute` | `"negated"` | attribute carrying polarity; `None` disables handling |
| `negated_means` | `True` | set `False` when the attribute is named like `present` |
| `system` | `"langextract"` | the `system` value stamped on predictions |
| `prompt_version` | `"v1"` | recorded for the audit trail |
| `require_offsets` | `False` | raise instead of yielding a span-less prediction |

## Testing without an API key

The adapter splits into a network half and a pure half.
`predictions_from_documents` takes `AnnotatedDocument` objects and returns
predictions, so you can test the mapping with no network at all — which is how
clinval's own 46 adapter tests work:

```python
from clinval.plugins.langextract import predictions_from_documents

doc = lx.data.AnnotatedDocument(
    extractions=[
        lx.data.Extraction(
            extraction_class="diarrhea",
            extraction_text="loose stools",
            char_interval=lx.data.CharInterval(start_pos=8, end_pos=20),
            attributes={"negated": "false"},
        )
    ],
    text=note_text,
)
preds = predictions_from_documents([doc], notes[:1], config)
```

## A note on langextract's own plugin system

langextract's plugin registry registers **LLM backends** — entry-point group
`langextract.providers`, each plugin a `BaseLanguageModel` subclass with an
`infer` method. That plugs models *into* langextract, which is the opposite
direction from what clinval needs. clinval's `ExtractorPlugin` protocol is its
own; see [decisions](../decisions.md) D-002.
