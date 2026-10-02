"""The runner: predictions plus annotations plus config, in; a report, out.

One function, :func:`run_eval`, executes the whole pipeline in SPEC §4.2 order
and returns an :class:`EvalReport` that can be written as JSON, Markdown, or
HTML. Every stage is also callable on its own; the runner exists so that the
common case is one line and so that the stages always run in the same order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from clinval.aggregate import (
    AlignedFrame,
    WindowedFrame,
    aggregate_to_windows,
    align_predictions,
)
from clinval.agreement import AgreementReport, compute_agreement, inter_annotator_agreement
from clinval.cost import CostReport, summarize_cost
from clinval.errors import ErrorTable, confusion_export, tag_errors
from clinval.exceptions import ConfigError
from clinval.intervals import paired_bootstrap_difference
from clinval.io import (
    read_annotations,
    read_index_dates,
    read_json,
    read_predictions,
    write_json,
)
from clinval.match.ontology import OntologyMap
from clinval.schemas.annotation import Annotation
from clinval.schemas.config import EvalConfig, FrozenConfig, Unit
from clinval.schemas.prediction import Prediction


@dataclass
class MetricsTable:
    """Per-system metrics, renderable as Markdown or a frame."""

    reports: dict[str, AgreementReport] = field(default_factory=dict)

    def __getitem__(self, system: str) -> AgreementReport:
        return self.reports[system]

    def __iter__(self) -> Any:
        return iter(self.reports)

    def __len__(self) -> int:
        return len(self.reports)

    def systems(self) -> list[str]:
        """System names, sorted."""
        return sorted(self.reports)

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form frame: one row per (system, metric) with point and bounds."""
        rows: list[dict[str, Any]] = []
        for system in self.systems():
            rep = self.reports[system]
            for metric, iv in rep.metrics.items():
                rows.append(
                    {
                        "system": system,
                        "metric": metric,
                        "point": iv.point,
                        "ci_low": iv.low,
                        "ci_high": iv.high,
                        "method": iv.method,
                        "n_units": rep.n_units,
                        "n_clusters": rep.n_clusters,
                    }
                )
        return pd.DataFrame(rows)

    def to_markdown(self, digits: int = 3) -> str:
        """One table per system, with the confusion matrix and per-label rows."""
        if not self.reports:
            return "_No systems were scored._"
        return "\n\n".join(self.reports[s].to_markdown(digits=digits) for s in self.systems())

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe nested dict."""
        return {s: r.as_dict() for s, r in sorted(self.reports.items())}


@dataclass
class EvalReport:
    """The complete result of one evaluation run.

    Attributes:
        config: The config that produced it.
        config_hash: SHA-256 of that config.
        metrics: Per-system metrics with intervals.
        errors: Tagged disagreements.
        cost: Cost and latency accounting.
        aligned: Note-level joined rows.
        windowed: Unit-level rows.
        comparisons: Paired bootstrap comparisons between system pairs.
        iaa: Inter-annotator agreement, when a multi-rater file was supplied.
        split_check: Result of the dev/test disjointness check, when run.
        warnings: Things a reader needs to know before trusting the numbers.
    """

    config: EvalConfig
    config_hash: str
    metrics: MetricsTable
    errors: ErrorTable
    cost: CostReport
    aligned: AlignedFrame
    windowed: WindowedFrame
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    iaa: dict[str, Any] = field(default_factory=dict)
    split_check: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    frozen: FrozenConfig | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- summaries -----------------------------------------------------

    @property
    def n_units(self) -> int:
        """Unit-level rows produced."""
        return len(self.windowed)

    def as_dict(self) -> dict[str, Any]:
        """The SPEC §6.3 eval-report JSON, plus clinval's audit fields."""
        first = self.metrics.systems()[0] if len(self.metrics) else None
        per_label: list[dict[str, Any]] = []
        for system in self.metrics.systems():
            rep = self.metrics[system]
            for label, vals in sorted(rep.per_label.items()):
                row: dict[str, Any] = {"system": system, "label": label}
                conf = rep.per_label_confusion.get(label)
                if conf:
                    row.update(conf.as_dict())
                row.update({k: v.as_dict() for k, v in vals.items()})
                per_label.append(row)

        return {
            "config_hash": self.config_hash,
            "clinval_version": self.meta.get("clinval_version"),
            "frozen": (
                {
                    "config_hash": self.frozen.config_hash,
                    "frozen_at": self.frozen.frozen_at.isoformat(),
                    "git_commit": self.frozen.git_commit,
                    "verified": True,
                }
                if self.frozen
                else None
            ),
            "split": self.config.split.value,
            "unit": self.config.unit.value,
            "units": self.n_units,
            "systems": self.metrics.systems(),
            "metrics": self.metrics.as_dict().get(first, {}).get("metrics", {}) if first else {},
            "metrics_by_system": self.metrics.as_dict(),
            "cis": {
                s: {m: v.as_dict() for m, v in self.metrics[s].metrics.items()}
                for s in self.metrics.systems()
            },
            "per_label": per_label,
            "error_counts": {m: int(n) for m, n in sorted(self.errors.counts.items())},
            "error_counts_full": self.errors.by_mechanism().to_dict("records"),
            "cost": self.cost.as_dict(),
            "comparisons": self.comparisons,
            "iaa": self.iaa,
            "split_check": self.split_check,
            "coverage": self.aligned.coverage(),
            "warnings": self.warnings,
        }

    # -- writers -------------------------------------------------------

    def write_json(self, path: str | Path) -> Path:
        """Write the metrics JSON."""
        write_json(self.as_dict(), path)
        return Path(path)

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report."""
        from clinval.report.builder import render_markdown

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_markdown(self), encoding="utf-8")
        return p

    def write_html(self, outdir: str | Path) -> Path:
        """Write the HTML report and its figures into ``outdir``."""
        from clinval.report.builder import write_report

        return write_report(self, outdir)["html"]

    def write_all(self, outdir: str | Path) -> dict[str, Path]:
        """Write every artifact: HTML, Markdown, metrics JSON, and CSV exports."""
        from clinval.report.builder import write_report

        return write_report(self, outdir)

    def to_markdown(self, digits: int = 3) -> str:
        """Render the full Markdown report as a string."""
        from clinval.report.builder import render_markdown

        return render_markdown(self, digits=digits)


def _as_predictions(obj: Any) -> list[Prediction]:
    if isinstance(obj, str | Path):
        return read_predictions(obj)
    return list(obj)


def _as_annotations(obj: Any) -> list[Annotation]:
    if isinstance(obj, str | Path):
        return read_annotations(obj)
    return list(obj)


def _as_config(obj: Any) -> EvalConfig:
    if obj is None:
        return EvalConfig()
    if isinstance(obj, EvalConfig):
        return obj
    if isinstance(obj, FrozenConfig):
        return obj.config
    if isinstance(obj, str | Path):
        return EvalConfig.from_yaml(obj)
    if isinstance(obj, dict):
        return EvalConfig.from_dict(obj)
    raise ConfigError(f"cannot interpret {type(obj).__name__} as an EvalConfig")


def _as_index(obj: Any) -> pd.DataFrame | None:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    return read_index_dates(obj)


def _as_cost_model(obj: Any, config: EvalConfig) -> Any:
    """Resolve a cost model from a path, a dict, or the config."""
    from clinval.schemas.config import CostModel

    if obj is None:
        return config.cost
    if isinstance(obj, CostModel):
        return obj
    if isinstance(obj, dict):
        return CostModel.model_validate(obj)
    raw = read_json(obj)
    if isinstance(raw, dict):
        known = {k: v for k, v in raw.items() if k in CostModel.model_fields}
        return CostModel.model_validate(known)
    raise ConfigError(f"cannot interpret {obj!r} as a cost model")


def run_eval(
    predictions: Sequence[Prediction] | str | Path,
    annotations: Sequence[Annotation] | str | Path,
    config: EvalConfig | FrozenConfig | str | Path | dict[str, Any] | None = None,
    outdir: str | Path | None = None,
    *,
    index: pd.DataFrame | str | Path | None = None,
    cost: Any = None,
    ontology: OntologyMap | None = None,
    frozen: FrozenConfig | None = None,
    annotator_records: Sequence[Annotation] | str | Path | None = None,
    compare_systems: bool = True,
) -> EvalReport:
    """Run the full evaluation pipeline.

    Stage order follows SPEC §4.2: align, aggregate, score with intervals, tag
    errors, account for cost, then build the report.

    Args:
        predictions: Prediction records or a path to a JSONL file.
        annotations: Gold records or a path to a JSONL file.
        config: Config object, path to YAML, dict, or a
            :class:`~clinval.schemas.config.FrozenConfig`. Defaults are used
            when omitted.
        outdir: When given, every artifact is written here.
        index: Per-patient index dates. Required for ``unit=patient_window``.
        cost: A cost model, a dict, or a path to a JSON file carrying
            ``input_per_1k``/``output_per_1k``. Defaults to the config's.
        ontology: Concept map for code matching.
        frozen: A verified frozen config, recorded in the report for the audit
            trail.
        annotator_records: Multi-annotator records for inter-rater agreement.
            Separate from ``annotations``, which must be single-gold.
        compare_systems: Run paired bootstrap comparisons between every pair of
            systems.

    Returns:
        An :class:`EvalReport`.
    """
    from clinval import __version__

    cfg = _as_config(frozen or config)
    preds = _as_predictions(predictions)
    golds = _as_annotations(annotations)
    idx = _as_index(index)
    cost_model = _as_cost_model(cost, cfg)
    warnings: list[str] = []

    if cfg.unit is Unit.PATIENT_WINDOW and idx is None:
        raise ConfigError(
            "unit='patient_window' needs per-patient index dates. Pass index=... "
            "(a CSV with patient_id and index_date), or set unit: note in the config."
        )

    # 1. align
    aligned = align_predictions(preds, golds, cfg, ontology=ontology)

    # 2. aggregate
    windowed = aggregate_to_windows(aligned, cfg.window, index=idx, unit=cfg.unit)

    # coverage warnings: silence is not a negative, so say so loudly
    for system, cov in sorted(aligned.coverage().items()):
        if not math.isnan(cov) and cov < 0.999:
            warnings.append(
                f"System {system!r} emitted predictions for only {cov:.1%} of gold "
                "(note, label) pairs. The missing pairs are counted as unscored, not as "
                "negatives. Run `clinval negatives` to materialize explicit negatives if "
                "the system did process those notes."
            )

    if len(windowed.df) and "window_miss_reason" in windowed.df.columns:
        missed = windowed.df["window_miss_reason"].astype(str).str.len() > 0
        if missed.any():
            reasons = windowed.df.loc[missed, "window_miss_reason"].value_counts().head(5).to_dict()
            warnings.append(
                f"{int(missed.sum())} of {len(windowed.df)} units could not be placed in a "
                f"window: {reasons}. They are reported as unscored and appear in the "
                "error table where relevant."
            )

    # 3. score
    reports: dict[str, AgreementReport] = {}
    for system in windowed.systems():
        rows = windowed.df[windowed.df["system"] == system]
        reports[system] = compute_agreement(
            rows,
            system=system,
            metrics=cfg.metrics,
            bootstrap=cfg.bootstrap,
            per_label=True,
            per_window=cfg.unit is Unit.PATIENT_WINDOW,
        )
    metrics = MetricsTable(reports=reports)

    # 4. tag errors
    error_table = tag_errors(aligned, cfg, windowed=windowed, ontology=ontology)

    # 5. cost
    cost_report = summarize_cost(preds, cost_model)

    # 6. optional extras
    comparisons: list[dict[str, Any]] = []
    if compare_systems and len(reports) > 1:
        comparisons = _compare_systems(windowed, cfg)

    iaa: dict[str, Any] = {}
    if annotator_records is not None:
        recs = _as_annotations(annotator_records)
        frame = pd.DataFrame(
            [
                {
                    "annotator": a.annotator,
                    "note_id": a.note_id,
                    "label": a.label,
                    "value": a.value,
                }
                for a in recs
            ]
        )
        if len(frame) and frame["annotator"].nunique() > 1:
            iaa = inter_annotator_agreement(frame)
        else:
            warnings.append(
                "annotator_records held a single annotator, so no inter-rater agreement "
                "could be computed."
            )

    report = EvalReport(
        config=cfg,
        config_hash=cfg.content_hash(),
        metrics=metrics,
        errors=error_table,
        cost=cost_report,
        aligned=aligned,
        windowed=windowed,
        comparisons=comparisons,
        iaa=iaa,
        warnings=warnings,
        frozen=frozen,
        meta={
            "clinval_version": __version__,
            "n_predictions": len(preds),
            "n_annotations": len(golds),
            "n_notes": int(aligned.df["note_id"].nunique()) if len(aligned.df) else 0,
            "n_patients": int(aligned.df["patient_id"].nunique()) if len(aligned.df) else 0,
            "annotators": list(aligned.annotators),
        },
    )

    if outdir is not None:
        report.write_all(outdir)
    return report


def _metric_statistic(metric: str) -> Any:
    """A confusion-table statistic bound to one metric name."""
    from clinval.agreement import METRIC_FUNCS, Confusion

    return lambda df: METRIC_FUNCS[metric](Confusion.from_frame(df))


def _compare_systems(windowed: WindowedFrame, cfg: EvalConfig) -> list[dict[str, Any]]:
    """Paired cluster bootstrap on every pair of systems, for each metric.

    Paired on ``unit_key``, so both systems are scored on the same units in
    every resample. Only metrics that are defined on a confusion table are
    compared, and an undefined difference is reported as ``None`` rather than 0.
    """

    systems = windowed.systems()
    out: list[dict[str, Any]] = []
    for i, a in enumerate(systems):
        for b in systems[i + 1 :]:
            rows_a = windowed.df[
                (windowed.df["system"] == a)
                & windowed.df["pred"].notna()
                & windowed.df["gold"].notna()
            ]
            rows_b = windowed.df[
                (windowed.df["system"] == b)
                & windowed.df["pred"].notna()
                & windowed.df["gold"].notna()
            ]
            for metric in cfg.metrics:
                statistic = _metric_statistic(metric)
                iv, p = paired_bootstrap_difference(
                    rows_a,
                    rows_b,
                    statistic,
                    cluster_by=cfg.bootstrap.cluster_by,
                    unit_col="unit_key",
                    n_boot=cfg.bootstrap.n,
                    alpha=cfg.bootstrap.alpha,
                    seed=cfg.bootstrap.seed,
                )
                out.append(
                    {
                        "system_a": a,
                        "system_b": b,
                        "metric": metric,
                        "difference": None if math.isnan(iv.point) else iv.point,
                        "ci_low": None if math.isnan(iv.low) else iv.low,
                        "ci_high": None if math.isnan(iv.high) else iv.high,
                        "p_value": None if math.isnan(p) else p,
                        "n_shared_units": iv.n,
                        "n_clusters": iv.n_clusters,
                        "method": iv.method,
                    }
                )
    return out


__all__ = ["EvalReport", "MetricsTable", "confusion_export", "run_eval"]
