# Decisions and deviations

A record of every decision that departs from the specification, resolves an
ambiguity in it, or rests on something verified about an external library. Each
entry says what was checked, when, and how to re-check it.

---

## D-001 — The name `clinval` is available

**Status:** resolved, no approval needed.
**Checked:** 2 October 2026.

- **PyPI:** `clinval` returns 404. Eight candidate alternatives were also free,
  so no fallback was needed.
- **GitHub:** four repositories match `clinval` in the name
  (`Thesaleem/clinval`, `clinverse/clinValidations`, `vbabenko97/clinval-validator`,
  `OjasRamwala/ClinValAI`), all unrelated and at zero stars. GitHub namespaces
  are per-owner, so `DanPalBCM/clinval` is unaffected.

**Re-check with:**

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/clinval/json
gh api 'search/repositories?q=clinval+in:name' --jq '.total_count'
```

**Note on confusability:** `clinval` is one letter from **ClinVar**, the NCBI
variant database. The two are in different subfields and the spelling differs,
but it is worth knowing if the project is ever presented to a genomics
audience.

---

## D-002 — LangExtract's plugin API is a *provider* registry, not an extractor interface

**Status:** resolved; the spec's §4.1 layout and §4.3 protocol stand unchanged.
**Checked:** 2 October 2026 against `langextract` **1.7.0**, installed and
introspected rather than read about.

What `langextract` actually exposes:

- Its plugin system registers **LLM backends**: entry-point group
  `langextract.providers`, each plugin a subclass of
  `langextract.core.base_model.BaseLanguageModel` with one abstract method,
  `infer`. Built-ins are `gemini`, `ollama`, and `openai`. That plugs models
  *into* langextract — the opposite direction from what clinval needs.
- The surface clinval consumes is
  `lx.extract(...) -> AnnotatedDocument | list[AnnotatedDocument]`, with
  `data.Extraction(extraction_class, extraction_text, char_interval,
  alignment_status, extraction_index, group_index, description, attributes)`.

**Conclusion:** clinval's `ExtractorPlugin` protocol (SPEC §4.3) is clinval's
own and is unaffected; `plugins/langextract.py` is a thin adapter that calls
`lx.extract` and maps `Extraction` to `Prediction`. No change to §4.1 was
required.

`tests/test_plugin_langextract.py::TestLibraryShape` pins the field names this
adapter depends on, so a renaming upstream fails loudly rather than silently
mis-mapping.

---

## D-003 — langextract reports no token usage, so cost is *unknown*, not zero

**Status:** resolved; approved by the project owner on 2 October 2026
("build it, document the gaps").
**Checked:** 2 October 2026 against `langextract` 1.7.0.

`langextract` exposes **no** token accounting anywhere. `ScoredOutput` is
`(score, output)`; a grep across the installed package for
`tokens_in|token_count|usage_metadata|prompt_tokens|total_tokens` returns only
a log message about tokenizer length. `AnnotatedDocument` has no usage
attribute.

Consequences, all of them visible to the caller rather than papered over:

| Field | Behaviour | Why |
| --- | --- | --- |
| `run.tokens_in`, `run.tokens_out` | left unset | the library reports nothing to put there |
| `run.latency_s` | measured by clinval with a wall clock | clinval can observe this itself |
| `cost` | `None`, with `cost_status="unknown"` | a zero would make the least-instrumented system look cheapest |

SPEC §3.7 lists "per-system token counts" in scope. It remains implemented and
tested; it is simply **not obtainable from this particular extractor**, and the
report says so in words rather than printing a misleading number. Extractors
that do report tokens — `fhirground`, for one — get a complete cost.

A guard test (`test_library_still_reports_no_token_usage`) fails if a future
langextract release adds usage metadata, which is the signal to start
populating those fields.

---

## D-004 — `call_id`: one model call, charged once

**Status:** addition to SPEC §6.1, not a deviation from it.

One model call usually produces *several* prediction records — a single pass
over a note typically answers every label at once. Stamping that call's token
count on each record and summing them multiplies the true cost by the number of
labels. The demo originally reported \$19 for 150 predictions over 75 notes for
exactly this reason.

`RunMeta.call_id` is an optional, stable identifier for the model call.
`summarize_cost` counts each distinct `call_id` once for both tokens and
latency, and `SystemCost.n_calls` reports how many calls were seen. Records
without a `call_id` are charged individually, which preserves the previous
behaviour for anyone who does not set it.

Both bundled producers set it to the note id: `clinval.synth` and the
langextract adapter each make one call per note.

---

## D-005 — A third error type, `MATCH`

**Status:** extension of SPEC §6.4.

§6.4 defines `error_type` as `FP | FN`. Those two cannot express a real and
separately actionable failure: **the label is right but the evidence or the
code is wrong.** A system that correctly says "this patient had diarrhea" while
quoting the wrong sentence, or attaching constipation's code, has made a
mistake that neither FP nor FN describes.

clinval therefore emits a third value, `MATCH`, for rows where prediction and
gold agree on the boolean but span or code matching failed. Without it,
`ontology_mismatch` and the span-boundary form of `annotation_threshold` would
be unreachable tags, and SPEC §11's requirement that every mechanism have a
triggering fixture could not be met honestly.

`FP` and `FN` keep exactly their §6.4 meanings. Consumers that only understand
two error types can filter `error_type != "MATCH"`.

---

## D-006 — `require_span` is what gives span thresholds an effect on the metrics

**Status:** resolution of an ambiguity in SPEC §5.3.

The spec's example tuning grid varies `min_overlap`. But the evaluation unit is
a **boolean** label, and span agreement is reported separately, so on a first
implementation `min_overlap` had *no effect whatsoever* on kappa — the grid was
perfectly flat and the "tuned" value was arbitrary.

Resolution: `matching.span.require_span` now has teeth. With it set, a positive
prediction whose evidence fails the span and code policy is **not credited**
and scores as a negative. `min_overlap` then moves sensitivity and kappa
directly, and tuning it means something.

With `require_span` off — the default — `pred` equals `pred_raw` and the
boolean metrics are untouched, so nothing changes for existing configs. The
aligned frame carries `pred_raw` and `credited` alongside `pred` so the
downgrade is always auditable, and `TuningResult.is_flat` warns when a grid
turned out not to matter.

The bundled `examples/data/eval.yaml` sets `require_span: true` for this
reason.

---

## D-007 — Tuning a consensus rule is scored on the units it decided

**Status:** hazard identified and surfaced, not silently accepted.

SPEC §5.3's grid includes `consensus: [majority, unanimous]`. Scoring a
consensus candidate exposed a trap: under `majority` with two systems, a 1–1
split is **undecided**, undecided units are excluded from the confusion table,
and the rule therefore posts a near-perfect score by declining to answer the
hard questions. On the demo data, `majority` scored kappa 1.000 while deciding
only 70% of units; `unanimous` scored 0.872 while deciding all of them.

clinval does not "fix" this by inventing a score for undecided units — routing
a case to a human is a legitimate outcome, not an error. Instead:

- every consensus candidate records `decided_fraction`, printed as a column in
  the tuning table;
- `TuningResult.warnings` names the trade-off explicitly when the winner
  decides materially fewer units than an alternative;
- `ConsensusResult.agreed_vs_disagreed` reports accuracy on agreed versus
  contested units, with the lift, so a gate that separates nothing is visible.

The walkthrough config tunes `min_overlap` only, because the consensus rule
governs the consensus gate rather than the per-system evaluation and tuning
both at once in one table would be incoherent. Consensus tuning is covered in
`tests/test_freeze.py`.

---

## D-008 — Multi-annotator gold must be adjudicated, not averaged

**Status:** resolution of an ambiguity in SPEC §6.2.

§6.2 allows two annotators' records to coexist and says adjudication is
resolved by `clinval adjudicate`. It does not say what happens if you evaluate
against an unadjudicated two-rater file.

clinval raises. Averaging or silently picking a rater would hide the
disagreement that a validation study exists to measure. The default
adjudication rule, `agreement`, *drops* pairs the raters disagreed on rather
than guessing, and the CLI reports how many it dropped.

---

## D-009 — A missing prediction is unscored, never a negative

**Status:** implementation of SPEC §6.1, with the consequence made visible.

§6.1 says the absence of a positive prediction is a negative *for a processed
note*, and that systems must emit explicit `false` records or use
`clinval negatives`. The operative qualifier is "processed": clinval cannot
distinguish "the system considered this note and said no" from "the system
never saw it".

So a gold pair with no matching prediction is counted as **unscored** and
excluded from the confusion table, never read as a negative. Because that
silently flatters recall if ignored, `AlignedFrame.coverage()` reports the
share of gold pairs each system actually predicted on, and the runner emits a
warning naming the system and the percentage. `clinval negatives` is the opt-in
for callers who know the note universe.

---

## D-010 — `runner.py` is an added module

**Status:** addition to the SPEC §4.1 layout.

§4.1 lists `freeze.py` and `report/builder.py` but gives `run_eval` (§4.3) no
home. Putting it in `freeze.py` would mix the protocol guards with the
pipeline, and putting it in `report/builder.py` would conflate computing
results with rendering them. It lives in `clinval/runner.py`, which matches
§3.9's description of the runner as its own concern. Everything in §4.1 exists
as specified.

---

## D-011 — Supplementary statistics dependency

**Status:** test-only addition to SPEC §7.

§8.1 requires Wilson intervals to be checked against a `statsmodels`
reference, but §7 does not list `statsmodels`. It is added to the **dev**
dependency group only; clinval's runtime does not import it.

`langextract` is likewise in the dev group so the adapter is tested against the
real library objects rather than a hand-written stub, as SPEC §11 prefers. It
remains an optional extra for users.

---

## D-012 — fhirground interoperability is tested against a fixture

**Status:** resolution of a forward reference in SPEC §8.4.

§8.4 asks for an interop test against `fhirground`'s
`clinval_predictions.jsonl`. That sibling project does not exist yet, so the
test runs against a committed fixture built to the contract its spec states:
`system` is the config name, `label` is `<ResourceType>:<code or text>`, spans
come from the evidence quote, and `run` carries model provenance.

The part most likely to break a consumer is the colon-bearing, high-cardinality
label. clinval treats `label` as an **opaque string** and never splits or
interprets it, and a test asserts that `Condition:K59.1` survives a round trip
intact.

When fhirground exists, point this test at its real output; the fixture
documents what clinval expects in the meantime.

---

## D-013 — Case is folded for fuzzy matching; accents are not

**Status:** behavioural decision.

`rapidfuzz` is case-sensitive by default. Left that way, an extractor quoting
`"Diarrhea since Monday"` against an annotator's `"diarrhea since Monday"`
scored 0.67 and failed an 0.85 threshold — a sentence-initial capital read as a
near miss. Both span text and ontology display names are therefore case-folded
before comparison.

Accents are **not** folded. Case carries no clinical meaning, but accent
folding can merge genuinely distinct terms in some languages, so it stays a
policy choice rather than a silent default.

---

## D-014 — Aggregation is idempotent by detection

**Status:** implementation of the SPEC §8.2 property.

§8.2 requires that aggregating an already-aggregated frame be a no-op. A
property test found this false for `unit=patient`: the aggregated frame has no
`note_id` column, so a second pass raised `KeyError`.

`aggregate_to_windows` now detects aggregated input — `unit_key` together with
`n_notes`, a pair that note-level rows never carry — and returns it unchanged.
That makes the no-op a genuine no-op rather than a second collapse that would
lose the original note ids.

---

## D-015 — The bundled ontology map is demo scope, deliberately

**Status:** scope boundary.

`clinval.match.ontology.BUNDLED_MAP` holds six concepts: the gastrointestinal
and endocrine terms used by the synthetic generator, the tests, and the docs
walkthrough. It is **not** a terminology service and must not be mistaken for
one. Real studies supply their own crosswalk via `load_ontology_map`.

A test asserts the map stays under twenty entries, as a guard against it
quietly growing into a pretend terminology that users might trust.

---

## D-016 — Console error messages are escaped before printing

**Status:** bug found by a test, fixed.

clinval's record errors end with a `[path:line]` location. `rich` parses square
brackets as markup, so printing such a message raised `MarkupError` internally
and the CLI emitted **nothing at all** — the single worst failure mode for a
tool whose job is telling you which row is malformed. `_fail` now escapes the
message. `tests/test_cli.py::test_reports_the_offending_line` covers it.

---

## D-017 — Every note in a unit gets window context

**Status:** bug found by review, fixed.

Error tagging originally took window context from each unit's *representative*
and *gold* note only. In a unit aggregating several notes, the remaining notes
carried no window, which could mistag them as `window_placement` errors.

`aggregate_to_windows` now records full membership in a `note_ids` column, and
`tag_errors` maps every member note to its unit. Covered by
`test_every_note_in_a_multi_note_unit_gets_window_context`.

---

## D-018 — Release scope: GitHub only

**Status:** decided by the project owner, 2 October 2026.

Tagged `v1.0.0` on GitHub with documentation sources committed and buildable
locally. **Not** published to PyPI and no documentation site deployed. SPEC
§11's "published to PyPI, docs deployed" items are therefore deferred rather
than met; nothing about the package blocks either step. A PyPI name and version
are effectively permanent, so that call is left to a deliberate later decision.
