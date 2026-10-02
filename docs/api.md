# API reference

Generated from docstrings. Everything listed here is importable from the
top-level `clinval` namespace unless noted.

## Runner

::: clinval.runner
    options:
      members:
        - run_eval
        - EvalReport
        - MetricsTable

## Schemas

::: clinval.schemas.prediction
    options:
      members:
        - Prediction
        - Span
        - Code
        - RunMeta

::: clinval.schemas.annotation
    options:
      members:
        - Annotation

::: clinval.schemas.config
    options:
      members:
        - EvalConfig
        - FrozenConfig
        - Matching
        - SpanMatching
        - OntologyMatching
        - Window
        - Duration
        - Bootstrap
        - CostModel
        - FreezeGrid
        - Split
        - Unit
        - SpanMode
        - TieBreak

## IO

::: clinval.io
    options:
      members:
        - read_predictions
        - read_annotations
        - read_index_dates
        - write_predictions
        - write_annotations
        - materialize_negatives

## Alignment and aggregation

::: clinval.aggregate
    options:
      members:
        - align_predictions
        - aggregate_to_windows
        - assign_windows
        - build_windows
        - resolve_gold
        - adjudicate
        - AlignedFrame
        - WindowedFrame

## Matching

::: clinval.match.spans
    options:
      members:
        - match_spans
        - jaccard_overlap
        - overlap_fraction
        - fuzzy_text_score
        - SpanMatchResult

::: clinval.match.ontology
    options:
      members:
        - match_codes
        - normalize_code
        - icd10_prefix
        - bundled_map
        - load_ontology_map
        - OntologyMap
        - ConceptEntry
        - CodeMatchResult

## Agreement

::: clinval.agreement
    options:
      members:
        - compute_agreement
        - compute_metrics
        - inter_annotator_agreement
        - Confusion
        - AgreementReport
        - cohen_kappa
        - pabak
        - sensitivity
        - specificity
        - ppv
        - npv
        - accuracy
        - f1
        - prevalence

## Confidence intervals

::: clinval.intervals
    options:
      members:
        - wilson_ci
        - clustered_bootstrap_ci
        - bootstrap_many
        - paired_bootstrap_difference
        - Interval

## Consensus

::: clinval.consensus
    options:
      members:
        - consensus_gate
        - apply_rule
        - pairwise_agreement
        - ConsensusResult

## Error analysis

::: clinval.errors
    options:
      members:
        - tag_errors
        - confusion_export
        - ErrorTable
        - has_negation_cue
        - has_temporal_cue

## Cost

::: clinval.cost
    options:
      members:
        - summarize_cost
        - token_cost
        - CostReport
        - SystemCost

## The freeze protocol

::: clinval.freeze
    options:
      members:
        - tune
        - freeze
        - write_frozen
        - load_frozen
        - evaluate_guard
        - check_split_disjoint
        - git_commit
        - TuningResult

## Plugins

::: clinval.plugins.base
    options:
      members:
        - ExtractorPlugin
        - Note
        - validate_predictions

::: clinval.plugins.langextract
    options:
      members:
        - LangExtractPlugin
        - LangExtractConfig
        - predictions_from_documents
        - langextract_available

## Synthetic data

::: clinval.synth.generator
    options:
      members:
        - generate
        - SyntheticDataset
        - SUPPORTED_LABELS

## Reports

::: clinval.report.builder
    options:
      members:
        - write_report
        - render_html
        - render_markdown
        - plot_metrics
        - plot_per_label
        - plot_error_mechanisms
        - plot_confusion

## Exceptions

::: clinval.exceptions
    options:
      members:
        - ClinvalError
        - RecordError
        - DuplicateRecordError
        - ConfigError
        - FrozenConfigError
        - SplitGuardError
        - AggregationError
        - PluginError
