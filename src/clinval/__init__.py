"""clinval: a validation harness for clinical LLM extraction.

clinval sits between LLM eval frameworks, which score prompts, and extraction
libraries, which stop at extraction. It takes note-level predictions from any
extractor, aligns them against clinician annotations, aggregates to
patient-by-time-window units, computes agreement with clustering-aware
confidence intervals, and tags every residual disagreement with a mechanism.

Four design commitments, each enforced in code rather than documentation:

* **Extractor-agnostic.** The harness never calls an LLM. Anything that emits
  clinval's JSONL format can be evaluated.
* **Frozen dev protocol.** Thresholds are tuned on dev, locked with a SHA-256
  hash, and verified before a test-set run. See :mod:`clinval.freeze`.
* **The patient is the unit.** Metrics are computed on patient-windows with
  bootstrap intervals clustered by patient.
* **Errors are first-class.** Every disagreement gets a mechanism tag, because
  the point is to learn, not just to score.

Quickstart::

    import clinval as cv

    data = cv.synth.generate(n_patients=100, labels=("diarrhea",), seed=0)
    dev, test = data.split_by_patient(0.5)

    config = cv.EvalConfig(
        unit="patient_window",
        window=cv.Window(size="6M", anchor="index_date"),
        matching=cv.Matching(span="overlap", min_overlap=0.5),
    )
    report = cv.run_eval(test.all_predictions, test.annotations, config, index=test.index)
    print(report.metrics.to_markdown())
"""

from __future__ import annotations

__version__ = "1.0.0"

from clinval import synth
from clinval.aggregate import (
    AlignedFrame,
    WindowedFrame,
    adjudicate,
    aggregate_to_windows,
    align_predictions,
    assign_windows,
    build_windows,
    resolve_gold,
)
from clinval.agreement import (
    AgreementReport,
    Confusion,
    accuracy,
    cohen_kappa,
    compute_agreement,
    compute_metrics,
    f1,
    inter_annotator_agreement,
    npv,
    pabak,
    ppv,
    prevalence,
    sensitivity,
    specificity,
)
from clinval.consensus import ConsensusResult, apply_rule, consensus_gate
from clinval.cost import CostReport, SystemCost, summarize_cost, token_cost
from clinval.errors import ERROR_TYPES, MECHANISMS, ErrorTable, confusion_export, tag_errors
from clinval.exceptions import (
    AggregationError,
    ClinvalError,
    ConfigError,
    DuplicateRecordError,
    FrozenConfigError,
    PluginError,
    RecordError,
    SplitGuardError,
)
from clinval.freeze import (
    TuningResult,
    check_split_disjoint,
    evaluate_guard,
    freeze,
    load_frozen,
    tune,
    write_frozen,
)
from clinval.intervals import (
    Interval,
    bootstrap_many,
    clustered_bootstrap_ci,
    paired_bootstrap_difference,
    wilson_ci,
)
from clinval.io import (
    materialize_negatives,
    read_annotations,
    read_index_dates,
    read_predictions,
    write_annotations,
    write_predictions,
)
from clinval.match import (
    OntologyMap,
    bundled_map,
    jaccard_overlap,
    load_ontology_map,
    match_codes,
    match_spans,
    normalize_code,
)
from clinval.plugins.base import ExtractorPlugin, Note, validate_predictions
from clinval.runner import EvalReport, MetricsTable, run_eval
from clinval.schemas import (
    KNOWN_METRICS,
    Annotation,
    Bootstrap,
    Code,
    CostModel,
    Duration,
    EvalConfig,
    FreezeGrid,
    FrozenConfig,
    Matching,
    OntologyMatching,
    Prediction,
    RunMeta,
    Span,
    SpanMatching,
    SpanMode,
    Split,
    TieBreak,
    Unit,
    Window,
)

__all__ = [
    "ERROR_TYPES",
    "KNOWN_METRICS",
    "MECHANISMS",
    "AggregationError",
    "AgreementReport",
    "AlignedFrame",
    "Annotation",
    "Bootstrap",
    "ClinvalError",
    "Code",
    "ConfigError",
    "Confusion",
    "ConsensusResult",
    "CostModel",
    "CostReport",
    "DuplicateRecordError",
    "Duration",
    "ErrorTable",
    "EvalConfig",
    "EvalReport",
    "ExtractorPlugin",
    "FreezeGrid",
    "FrozenConfig",
    "FrozenConfigError",
    "Interval",
    "Matching",
    "MetricsTable",
    "Note",
    "OntologyMap",
    "OntologyMatching",
    "PluginError",
    "Prediction",
    "RecordError",
    "RunMeta",
    "Span",
    "SpanMatching",
    "SpanMode",
    "Split",
    "SplitGuardError",
    "SystemCost",
    "TieBreak",
    "TuningResult",
    "Unit",
    "Window",
    "WindowedFrame",
    "__version__",
    "accuracy",
    "adjudicate",
    "aggregate_to_windows",
    "align_predictions",
    "apply_rule",
    "assign_windows",
    "bootstrap_many",
    "build_windows",
    "bundled_map",
    "check_split_disjoint",
    "clustered_bootstrap_ci",
    "cohen_kappa",
    "compute_agreement",
    "compute_metrics",
    "confusion_export",
    "consensus_gate",
    "evaluate_guard",
    "f1",
    "freeze",
    "inter_annotator_agreement",
    "jaccard_overlap",
    "load_frozen",
    "load_ontology_map",
    "match_codes",
    "match_spans",
    "materialize_negatives",
    "normalize_code",
    "npv",
    "pabak",
    "paired_bootstrap_difference",
    "ppv",
    "prevalence",
    "read_annotations",
    "read_index_dates",
    "read_predictions",
    "resolve_gold",
    "run_eval",
    "sensitivity",
    "specificity",
    "summarize_cost",
    "synth",
    "tag_errors",
    "token_cost",
    "tune",
    "validate_predictions",
    "wilson_ci",
    "write_annotations",
    "write_frozen",
    "write_predictions",
]
