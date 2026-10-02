# Changelog

All notable changes to clinval are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-02

First release.

### Added

**Record formats and IO**

- `Prediction` and `Annotation` records (JSONL), with `Span`, `Code`, and
  `RunMeta`. Unknown fields are rejected so a typo in a field name is caught.
- Readers that report the file and 1-based line number of the first bad row.
- Duplicate detection on `(system, note_id, label)` and
  `(annotator, note_id, label)` — a repeat raises rather than silently
  overwriting.
- Index-date CSV reader accepting OMOP and MEDS column names.
- `materialize_negatives` and `clinval negatives`, for extractors that emit
  only positives.

**Alignment and aggregation**

- `align_predictions` joins predictions to gold over the union of both sides,
  so nothing is dropped. A missing prediction is **unscored, never a negative**.
- `aggregate_to_windows` for `note`, `patient`, and `patient_window` units, with
  calendar-based durations, half-open `[start, end)` intervals, configurable
  tie-breaking, and an optional look-back window.
- `resolve_gold` and `adjudicate`: multi-annotator gold must be adjudicated, not
  averaged.
- Aggregation is idempotent for every unit.

**Matching**

- Span matching: `exact`, `overlap` (character Jaccard), and `fuzzy` (token-set
  ratio on quoted text, case-folded).
- `require_span`, which withholds credit from an unsupported positive and is
  what gives span thresholds an effect on the boolean metrics.
- Ontology matching: code normalization, ICD-10 prefix depth, bidirectional
  HPO/ICD-10 crosswalk, optional fuzzy display matching, and
  `load_ontology_map` for your own crosswalk.

**Statistics**

- Cohen's κ, PABAK, sensitivity, specificity, PPV, NPV, accuracy, F₁,
  prevalence. An undefined metric is `nan`, never `0.0`.
- Wilson intervals for proportions; clustered percentile bootstrap for κ,
  PABAK, and F₁, resampling **patients** rather than rows.
- `paired_bootstrap_difference` for comparing two systems on shared units.
- `inter_annotator_agreement` for the human ceiling.
- The interval method is named in every reported row.

**The freeze protocol**

- `tune` (grid search on dev, refuses test data), `freeze` (SHA-256 over the
  canonical config), `load_frozen` (verifies the hash), `evaluate_guard`.
- Canonical hashing, so key order and duration shorthands do not change the
  hash.
- `TuningResult.is_flat` warns when a grid did not discriminate.
- A deferral warning when a tuned consensus rule wins mainly by declining to
  decide.
- `check_split_disjoint` reports dev/test patient and note overlap.

**Consensus gating**

- `majority`, `unanimous`, and `any` rules. An even split under `majority` is
  **undecided**, not negative.
- `agreed_vs_disagreed` with an accuracy lift, so a gate that separates nothing
  is visible.
- A review queue sorted most-contested first, and gold-free pairwise
  between-system agreement.

**Error analysis**

- The seven-mechanism taxonomy — `annotation_threshold`, `window_placement`,
  `misreading`, `missing_evidence`, `ontology_mismatch`, `temporal_error`,
  `other` — with a `mechanism_reason` on every tag.
- A third error type, `MATCH`, for a right label with wrong evidence or a wrong
  code.
- Breakdowns by mechanism, label, window, and system; CSV export; confusion
  export with the 2×2 cell named.

**Cost and latency**

- Per-system token, latency, and USD accounting. A missing token count is
  reported as **unknown**, never zero.
- `RunMeta.call_id`, so one model call producing several records is charged
  once.

**Reports**

- Static HTML with no JavaScript, Markdown, metrics JSON, four matplotlib
  figures, and CSV exports. Caveats are ordered above the numbers.
- An unfrozen run carries a prominent banner stating the numbers are not
  out-of-sample.

**Plugins**

- `ExtractorPlugin` protocol and `validate_predictions`.
- A LangExtract adapter verified against `langextract` 1.7.0, with the
  library's limitations documented rather than hidden: no token counts, no
  negation handling, approximate offsets.

**Synthetic data**

- `clinval.synth.generate`: deterministic notes, gold, and two mock systems
  with deliberately different pathologies (`careful_reader`,
  `negation_blind`).
- `split_by_patient`, which splits by patient rather than by note.

**CLI**

- `synth`, `validate`, `negatives`, `adjudicate`, `tune`, `freeze`, `evaluate`,
  `run`, `consensus`, `version`.
- `clinval run` requires `--i-understand-this-is-dev` for an unfrozen run.

**Project**

- 632 tests at ~96% coverage. κ checked against scikit-learn over 200
  randomized tables; Wilson intervals against statsmodels. Property tests
  (hypothesis) for alignment completeness, aggregation idempotence, interval
  coverage, and freeze-hash sensitivity.
- `demo.py`, which runs the full protocol on committed toy data and
  demonstrates the tamper check firing.
- CI on Python 3.11, 3.12, and 3.13, including a job that fails if any data
  file appears outside the synthetic directories.

### Known limitations

- Boolean labels only; no categorical values and no relation extraction.
- clinval cannot verify that dev and test hold different patients, only report
  overlap in the files it is given.
- The bundled ontology map covers six concepts and is for the tests and the
  walkthrough. It is not a terminology service.
- Error mechanism tags are conservative surface heuristics used to explain
  disagreements; they never change a score.
- Bootstrap p-values are approximations, not exact tests.
- `index_date` is the only window anchor.
- Not published to PyPI, and no documentation site is deployed; see
  `docs/decisions.md` D-018.

[1.0.0]: https://github.com/DanPalBCM/clinval/releases/tag/v1.0.0
