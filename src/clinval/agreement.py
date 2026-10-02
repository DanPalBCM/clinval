"""Agreement statistics and the confusion matrix they are computed from.

Three families, and the difference between them is the point:

* **Accuracy-style** metrics (sensitivity, specificity, PPV, NPV) answer
  "how often is the system right", conditioned on truth or on the prediction.
* **Chance-corrected** metrics (Cohen's kappa) answer "how much better than
  guessing at the observed base rate".
* **Prevalence-adjusted** metrics (PABAK) answer the same question with the
  base rate neutralized, which is why kappa and PABAK diverge on rare labels.

A rare phenotype can show 97% accuracy and a kappa near zero. Reporting only
the first is the standard way clinical extraction results get oversold, so
clinval computes all three and the report prints prevalence beside them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from clinval.intervals import Interval, bootstrap_many, wilson_ci
from clinval.schemas.config import KNOWN_METRICS, Bootstrap


@dataclass(frozen=True, slots=True)
class Confusion:
    """A 2x2 contingency table of boolean prediction against boolean gold."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        """Total scored units."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def n_positive_gold(self) -> int:
        """Units the gold standard calls positive."""
        return self.tp + self.fn

    @property
    def n_positive_pred(self) -> int:
        """Units the system calls positive."""
        return self.tp + self.fp

    @classmethod
    def from_frame(cls, df: pd.DataFrame, *, pred: str = "pred", gold: str = "gold") -> Confusion:
        """Build a table from a frame, ignoring rows where either side is null."""
        if len(df) == 0:
            return cls()
        mask = df[pred].notna() & df[gold].notna()
        if not mask.any():
            return cls()
        p = df.loc[mask, pred].astype(bool).to_numpy()
        g = df.loc[mask, gold].astype(bool).to_numpy()
        return cls(
            tp=int(np.sum(p & g)),
            fp=int(np.sum(p & ~g)),
            fn=int(np.sum(~p & g)),
            tn=int(np.sum(~p & ~g)),
        )

    def as_dict(self) -> dict[str, int]:
        """Plain-dict form."""
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, "n": self.n}


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float:
    """Division that reports an undefined metric as ``nan`` rather than 0.

    A specificity with no gold negatives is genuinely undefined; returning 0.0
    would read as "the system never got a negative right", which is a different
    and false claim.
    """
    return float(num) / float(den) if den > 0 else float("nan")


def sensitivity(c: Confusion) -> float:
    """TP / (TP + FN). Undefined without gold positives."""
    return _ratio(c.tp, c.tp + c.fn)


def specificity(c: Confusion) -> float:
    """TN / (TN + FP). Undefined without gold negatives."""
    return _ratio(c.tn, c.tn + c.fp)


def ppv(c: Confusion) -> float:
    """TP / (TP + FP). Undefined when the system predicts no positives."""
    return _ratio(c.tp, c.tp + c.fp)


def npv(c: Confusion) -> float:
    """TN / (TN + FN). Undefined when the system predicts no negatives."""
    return _ratio(c.tn, c.tn + c.fn)


def accuracy(c: Confusion) -> float:
    """(TP + TN) / N."""
    return _ratio(c.tp + c.tn, c.n)


def f1(c: Confusion) -> float:
    """Harmonic mean of PPV and sensitivity."""
    denom = 2 * c.tp + c.fp + c.fn
    return _ratio(2 * c.tp, denom)


def prevalence(c: Confusion) -> float:
    """Gold-positive rate: (TP + FN) / N."""
    return _ratio(c.tp + c.fn, c.n)


def cohen_kappa(c: Confusion) -> float:
    """Cohen's kappa for two binary raters.

    Matches ``sklearn.metrics.cohen_kappa_score`` on the same table. Two edge
    cases are handled explicitly rather than by the formula:

    * **Perfect agreement with one class only** (every unit TN, or every unit
      TP): expected agreement is 1, so the formula divides by zero. The
      convention here is ``1.0`` -- the raters agreed on everything.
    * **Complete disagreement with one class only**: kappa is ``0.0``.
    """
    n = c.n
    if n == 0:
        return float("nan")
    po = (c.tp + c.tn) / n
    p_pred_pos = (c.tp + c.fp) / n
    p_gold_pos = (c.tp + c.fn) / n
    pe = p_pred_pos * p_gold_pos + (1 - p_pred_pos) * (1 - p_gold_pos)
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1 - pe)


def pabak(c: Confusion) -> float:
    """Prevalence-adjusted bias-adjusted kappa: ``2 * p_observed - 1``.

    Depends only on the observed agreement rate, so it is immune to the base
    rate that makes kappa collapse on rare labels. Reported next to kappa so a
    reader can see which effect they are looking at.
    """
    if c.n == 0:
        return float("nan")
    return 2.0 * accuracy(c) - 1.0


#: Metric name to function over a :class:`Confusion`.
METRIC_FUNCS: dict[str, Callable[[Confusion], float]] = {
    "sensitivity": sensitivity,
    "specificity": specificity,
    "ppv": ppv,
    "npv": npv,
    "accuracy": accuracy,
    "f1": f1,
    "prevalence": prevalence,
    "cohen_kappa": cohen_kappa,
    "pabak": pabak,
}

#: Metrics that are plain proportions of a known denominator, so a Wilson
#: interval applies directly. Kappa and F1 are not in this set: they are not
#: single proportions and need the bootstrap.
_WILSON_DENOMINATORS: dict[str, Callable[[Confusion], tuple[int, int]]] = {
    "sensitivity": lambda c: (c.tp, c.tp + c.fn),
    "specificity": lambda c: (c.tn, c.tn + c.fp),
    "ppv": lambda c: (c.tp, c.tp + c.fp),
    "npv": lambda c: (c.tn, c.tn + c.fn),
    "accuracy": lambda c: (c.tp + c.tn, c.n),
    "prevalence": lambda c: (c.tp + c.fn, c.n),
}


def compute_metrics(c: Confusion, metrics: list[str] | None = None) -> dict[str, float]:
    """Point estimates for the requested metrics from one table."""
    names = metrics if metrics is not None else list(KNOWN_METRICS)
    unknown = [m for m in names if m not in METRIC_FUNCS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; known: {sorted(METRIC_FUNCS)}")
    return {m: METRIC_FUNCS[m](c) for m in names}


@dataclass
class AgreementReport:
    """Metrics with intervals for one system, overall and per label."""

    system: str
    confusion: Confusion
    metrics: dict[str, Interval]
    per_label: dict[str, dict[str, Interval]]
    per_label_confusion: dict[str, Confusion]
    n_units: int
    n_clusters: int
    unscored: int = 0
    coverage: float = float("nan")
    per_window: dict[str, dict[str, Interval]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_window is None:
            self.per_window = {}

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe nested dict of every number in the report."""
        return {
            "system": self.system,
            "n_units": self.n_units,
            "n_clusters": self.n_clusters,
            "unscored": self.unscored,
            "coverage": None if math.isnan(self.coverage) else self.coverage,
            "confusion": self.confusion.as_dict(),
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
            "per_label": {
                label: {k: v.as_dict() for k, v in vals.items()}
                for label, vals in self.per_label.items()
            },
            "per_label_confusion": {k: v.as_dict() for k, v in self.per_label_confusion.items()},
            "per_window": {
                w: {k: v.as_dict() for k, v in vals.items()} for w, vals in self.per_window.items()
            },
        }

    def to_markdown(self, digits: int = 3) -> str:
        """Render the overall and per-label tables as GitHub-flavored Markdown."""
        names = list(self.metrics)
        lines = [
            f"### {self.system}",
            "",
            f"Scored units: {self.n_units}  |  clusters: {self.n_clusters}"
            f"  |  unscored: {self.unscored}",
            "",
            "| metric | estimate | 95% CI | method |",
            "| --- | --- | --- | --- |",
        ]
        for m in names:
            iv = self.metrics[m]
            ci = (
                "n/a"
                if math.isnan(iv.low) or math.isnan(iv.high)
                else f"{iv.low:.{digits}f}-{iv.high:.{digits}f}"
            )
            pt = "n/a" if math.isnan(iv.point) else f"{iv.point:.{digits}f}"
            lines += [f"| {m} | {pt} | {ci} | {iv.method} |"]

        c = self.confusion
        lines += [
            "",
            "| | gold + | gold - |",
            "| --- | --- | --- |",
            f"| pred + | {c.tp} | {c.fp} |",
            f"| pred - | {c.fn} | {c.tn} |",
        ]

        if self.per_label:
            lines += ["", "#### Per label", "", "| label | n | " + " | ".join(names) + " |"]
            lines += ["| --- | --- | " + " | ".join("---" for _ in names) + " |"]
            for label in sorted(self.per_label):
                cl = self.per_label_confusion.get(label, Confusion())
                cells = []
                for m in names:
                    cell = self.per_label[label].get(m)
                    cells.append(
                        "n/a"
                        if cell is None or math.isnan(cell.point)
                        else f"{cell.point:.{digits}f}"
                    )
                lines += [f"| {label} | {cl.n} | " + " | ".join(cells) + " |"]
        return "\n".join(lines)


def compute_agreement(
    windowed: pd.DataFrame,
    *,
    system: str = "system",
    metrics: list[str] | None = None,
    bootstrap: Bootstrap | None = None,
    per_label: bool = True,
    per_window: bool = False,
) -> AgreementReport:
    """Compute metrics with intervals for one system's scored rows.

    Interval method is chosen per metric: Wilson for the metrics that are
    proportions of a well-defined denominator, clustered bootstrap for kappa,
    PABAK, and F1. They are reported with the method named, because a reader
    comparing a kappa interval to a sensitivity interval should know they came
    from different machinery.

    Args:
        windowed: Rows at the evaluation unit, with ``pred``, ``gold``, and the
            cluster column. Rows with a null on either side are counted as
            unscored and excluded.
        system: Value to record as the system name.
        metrics: Which metrics to compute. Defaults to all known ones.
        bootstrap: Bootstrap settings. Defaults are used when omitted.
        per_label: Also compute per-label breakdowns.
        per_window: Also compute per-window-index breakdowns.

    Returns:
        An :class:`AgreementReport`.
    """
    boot = bootstrap or Bootstrap()
    names = metrics if metrics is not None else list(KNOWN_METRICS)

    total_rows = len(windowed)
    scored = windowed[windowed["pred"].notna() & windowed["gold"].notna()]
    unscored = total_rows - len(scored)
    overall = Confusion.from_frame(scored)

    n_clusters = (
        int(scored[boot.cluster_by].nunique())
        if boot.cluster_by in scored.columns and len(scored)
        else 0
    )
    coverage = float(len(scored) / total_rows) if total_rows else float("nan")

    intervals = _intervals_for(scored, overall, names, boot)

    per_label_out: dict[str, dict[str, Interval]] = {}
    per_label_conf: dict[str, Confusion] = {}
    if per_label and len(scored):
        for label, chunk in scored.groupby("label", observed=True):
            conf = Confusion.from_frame(chunk)
            per_label_conf[str(label)] = conf
            per_label_out[str(label)] = _intervals_for(chunk, conf, names, boot)

    per_window_out: dict[str, dict[str, Interval]] = {}
    if per_window and len(scored) and "window_index" in scored.columns:
        for widx, chunk in scored.groupby("window_index", observed=True, dropna=False):
            conf = Confusion.from_frame(chunk)
            key = "unassigned" if pd.isna(widx) else f"w{int(widx)}"
            per_window_out[key] = _intervals_for(chunk, conf, names, boot)

    return AgreementReport(
        system=system,
        confusion=overall,
        metrics=intervals,
        per_label=per_label_out,
        per_label_confusion=per_label_conf,
        n_units=len(scored),
        n_clusters=n_clusters,
        unscored=unscored,
        coverage=coverage,
        per_window=per_window_out,
    )


def _intervals_for(
    rows: pd.DataFrame, conf: Confusion, names: list[str], boot: Bootstrap
) -> dict[str, Interval]:
    """Pick Wilson or bootstrap per metric and compute the intervals."""
    out: dict[str, Interval] = {}
    needs_boot = [m for m in names if m not in _WILSON_DENOMINATORS]

    for m in names:
        if m in _WILSON_DENOMINATORS:
            num, den = _WILSON_DENOMINATORS[m](conf)
            out[m] = wilson_ci(num, den, alpha=boot.alpha)

    if needs_boot and len(rows):

        def _metric_of(name: str) -> Callable[[pd.DataFrame], float]:
            """Bind ``name`` so each closure scores its own metric."""
            return lambda df: METRIC_FUNCS[name](Confusion.from_frame(df))

        stats_map: dict[str, Callable[[pd.DataFrame], float]] = {
            m: _metric_of(m) for m in needs_boot
        }
        booted = bootstrap_many(
            rows,
            stats_map,
            cluster_by=boot.cluster_by,
            n_boot=boot.n,
            alpha=boot.alpha,
            seed=boot.seed,
        )
        out.update(booted)
    else:
        for m in needs_boot:
            out[m] = Interval(METRIC_FUNCS[m](conf), method="clustered_bootstrap", alpha=boot.alpha)
    return out


def inter_annotator_agreement(
    annotations: pd.DataFrame,
    *,
    annotator_col: str = "annotator",
    unit_col: str = "note_id",
    label_col: str = "label",
    value_col: str = "value",
) -> dict[str, Any]:
    """Pairwise agreement between annotators on the pairs they both labeled.

    This is the human ceiling. A system scoring a kappa of 0.70 against gold
    looks different when the two clinicians agreed at 0.72 than when they
    agreed at 0.95, so the report prints it alongside system performance.

    Returns:
        ``{"pairs": {"A|B": {...}}, "n_annotators": k}``, each entry carrying
        the confusion table, kappa, PABAK, observed agreement, and the number
        of overlapping units.
    """
    annotators = sorted(annotations[annotator_col].dropna().unique())
    out: dict[str, Any] = {"pairs": {}, "n_annotators": len(annotators)}

    for i, a in enumerate(annotators):
        for b in annotators[i + 1 :]:
            left = annotations[annotations[annotator_col] == a]
            right = annotations[annotations[annotator_col] == b]
            merged = left.merge(
                right,
                on=[unit_col, label_col],
                suffixes=("_a", "_b"),
                how="inner",
            )
            if merged.empty:
                out["pairs"][f"{a}|{b}"] = {"n_units": 0, "note": "no overlapping units"}
                continue
            conf = Confusion.from_frame(merged, pred=f"{value_col}_a", gold=f"{value_col}_b")
            out["pairs"][f"{a}|{b}"] = {
                "n_units": conf.n,
                "confusion": conf.as_dict(),
                "cohen_kappa": _none_if_nan(cohen_kappa(conf)),
                "pabak": _none_if_nan(pabak(conf)),
                "observed_agreement": _none_if_nan(accuracy(conf)),
                "prevalence_a": _none_if_nan(_ratio(conf.tp + conf.fp, conf.n)),
                "prevalence_b": _none_if_nan(prevalence(conf)),
            }
    return out


def _none_if_nan(x: float) -> float | None:
    return None if math.isnan(x) else float(x)
