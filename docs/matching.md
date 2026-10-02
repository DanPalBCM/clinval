# Span and ontology matching

Span and code agreement are reported **separately** from label agreement. A
system can be right about the phenotype and wrong about which sentence proves
it, and the [error taxonomy](errors.md) needs to tell those apart — that
distinction is what separates a boundary disagreement from a misreading.

## Span modes

```yaml
matching:
  span:
    mode: overlap        # exact | overlap | fuzzy
    min_overlap: 0.5
    fuzzy_threshold: 0.85
    require_span: false
```

### `exact`

Offsets must match exactly. Use it only when both sides came from the same
tokenizer. Note that `langextract` reports an `alignment_status` that can be
`MATCH_FUZZY`, `MATCH_GREATER`, or `MATCH_LESSER` — comparing those offsets
exactly measures the aligner, not the model, and the adapter warns when you do.

### `overlap` (default)

Character-level Jaccard index must reach `min_overlap`:

$$
J(a,b) = \frac{|a \cap b|}{|a \cup b|}
$$

| Spans | Jaccard |
| --- | --- |
| `(0,10)` vs `(0,10)` | 1.00 |
| `(0,10)` vs `(5,15)` | 0.33 |
| `(0,20)` vs `(5,10)` | 0.25 |
| `(0,10)` vs `(10,20)` | 0.00 — adjacent is disjoint |

Two zero-length spans report 0.0 rather than raising: a zero-length span
carries no extent to agree about.

`overlap_fraction` is also available when you want the overlap relative to one
span's length rather than the union.

### `fuzzy`

Compares the **quoted text** instead of the offsets, for extractors that
re-tokenize and lose exact positions. Uses token-set ratio, so word order and
duplicates are ignored: `"loose stools, watery"` and `"watery loose stools"`
score 1.0.

Text is **case-folded** first. rapidfuzz is case-sensitive by default, and
leaving it that way penalizes a sentence-initial capital — an extractor quoting
`"Diarrhea since Monday"` against an annotator's `"diarrhea since Monday"` is
agreement, not a near miss. Accents are **not** folded, because case carries no
clinical meaning while accent folding can merge genuinely distinct terms. See
[decisions](decisions.md) D-013.

When either side carries no text, `fuzzy` falls back to `overlap` and says so
in the result's `reason`.

## Missing spans

A missing span is **not** a failure by default. A system may assert a label
without quoting evidence, and whether that is acceptable is a policy question:

```yaml
matching:
  span:
    require_span: true
```

With it set, an unsupported positive is **not credited** and scores as a
negative. This is also what gives `min_overlap` an effect on the boolean
metrics — see [decisions](decisions.md) D-006 and
[Concepts](concepts.md#credited-versus-raw-predictions).

The result's `reason` always names which side was missing:
`pred_span_absent`, `gold_span_absent`, or `both_spans_absent`.

## Ontology matching

```yaml
matching:
  ontology:
    hpo:
      fuzzy: true
      threshold: 0.85
    icd-10:
      prefix_depth: 3
```

Keys are terminology names, upper-cased on load. Two codes can agree in three
ways, tried in this order:

**1. Identical after normalization.** HPO identifiers become zero-padded
`HP:0002014`, so `hp:2014`, `HP_0002014`, and `HP:0002014` are the same code.
ICD-10 becomes uppercase with the decimal point **retained** — the point is the
level separator, and dropping it would make `E1`+`19` indistinguishable from
`E11`+`9`.

**2. Hierarchically or by crosswalk.** With `prefix_depth`, codes in the same
terminology are compared at that depth: depth 3 of `E11.9` is `E11`, the
category. The bundled map also crosswalks between HPO and ICD-10 in both
directions, so `K59.1` matches `HP:0002014`.

**3. By display name, fuzzily.** Off by default, because it is the weakest of
the three. Display names are case-folded before comparison.

The result's `how` field records which route succeeded — `identical`,
`crosswalk`, `prefix`, `fuzzy_display`, `absent`, `disabled`, or `mismatch` —
and the error taxonomy uses it to decide whether a disagreement is an
`ontology_mismatch`.

### A missing code is not a failure

v1 treats codes as optional enrichment of a boolean label, so an absent code on
either side reports `how="absent"` and counts as a match. Requiring codes is
the caller's choice, expressed by filtering on `how`.

## The bundled map is demo scope

`clinval.match.ontology.BUNDLED_MAP` holds **six** concepts: the
gastrointestinal and endocrine terms used by the synthetic generator, the
tests, and this documentation.

!!! warning "This is not a terminology service"
    It must not be mistaken for one. A real study supplies its own crosswalk:

    ```python
    onto = cv.load_ontology_map("my_crosswalk.json")
    report = cv.run_eval(preds, golds, config, index=index, ontology=onto)
    ```

    Expected shape:

    ```json
    {"HP:0002014": {"display": "Diarrhea",
                    "synonyms": ["loose stools", "watery stools"],
                    "icd10": ["K59.1", "R19.7"]}}
    ```

A test asserts the bundled map stays under twenty entries, as a guard against
it quietly growing into a pretend terminology that users might trust.

## Grading text where a code was expected

`OntologyMap.lookup_by_text` finds the best concept for a free-text term, which
is useful when an extractor emits a phrase where you wanted a code:

```python
onto = cv.bundled_map()
onto.lookup_by_text("watery stools")        # ("HP:0002014", 1.0)
onto.lookup_by_text("fractured femur")      # None — below threshold
```
