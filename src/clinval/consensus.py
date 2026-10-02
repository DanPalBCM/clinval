"""Consensus gating across two or more systems.

The practical use: run two models, trust the units they agree on, and route the
rest to a human. That only saves review effort if the agreed-on units are
actually more reliable, so :func:`consensus_gate` reports accuracy *separately*
for agreed and disagreed units. If they are the same, the gate is buying
nothing, and the report says so rather than leaving it to be inferred.

Ties are explicit. Under ``majority`` with two systems, a 1-1 split is not a
negative -- it is ``needs_review``, with no consensus value. Scoring it as a
negative would be the single easiest way to make a gate look good.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from clinval.aggregate import AlignedFrame, WindowedFrame, aggregate_to_windows, align_predictions
from clinval.agreement import AgreementReport, Confusion, cohen_kappa, compute_agreement, pabak
from clinval.exceptions import ConfigError
from clinval.schemas.annotation import Annotation
from clinval.schemas.config import Bootstrap, EvalConfig, Unit
from clinval.schemas.prediction import Prediction

ConsensusRule = Literal["majority", "unanimous", "any"]

#: The rule names accepted by :func:`consensus_gate`, for CLI validation.
CONSENSUS_RULES: tuple[str, ...] = ("majority", "unanimous", "any")

#: Accepted inputs for one system in a consensus comparison.
SystemInput = Sequence[Prediction] | AlignedFrame | WindowedFrame


@dataclass
class ConsensusResult:
    """Per-unit votes, the gate's verdict, and what the gate bought.

    Attributes:
        df: One row per ``(unit_key, label)`` with a column per system, plus
            ``n_systems``, ``n_positive``, ``consensus``, ``needs_review``, and
            ``gold`` when gold was supplied.
        rule: The rule applied.
        systems: System names in sorted order.
        agreement: Metrics for the consensus verdict against gold, when
            available.
        agreed_vs_disagreed: Accuracy on units the systems agreed about versus
            units they did not -- the number that justifies or kills the gate.
        pairwise: Between-system agreement, independent of gold.
    """

    df: pd.DataFrame
    rule: ConsensusRule
    systems: tuple[str, ...]
    agreement: AgreementReport | None = None
    agreed_vs_disagreed: dict[str, Any] = field(default_factory=dict)
    pairwise: dict[str, Any] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        """Number of units voted on."""
        return len(self.df)

    @property
    def n_review(self) -> int:
        """Units routed to human review."""
        return int(self.df["needs_review"].sum()) if len(self.df) else 0

    @property
    def review_fraction(self) -> float:
        """Share of units routed to review."""
        return self.n_review / self.n_units if self.n_units else float("nan")

    def to_review_queue(self) -> pd.DataFrame:
        """The units a human needs to look at, most-contested first."""
        if not len(self.df):
            return self.df
        q = self.df[self.df["needs_review"]].copy()
        q["split_margin"] = (q["n_positive"] - (q["n_systems"] - q["n_positive"])).abs()
        return q.sort_values(["split_margin", "unit_key", "label"])

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "rule": self.rule,
            "systems": list(self.systems),
            "n_units": self.n_units,
            "n_review": self.n_review,
            "review_fraction": None if math.isnan(self.review_fraction) else self.review_fraction,
            "agreement": self.agreement.as_dict() if self.agreement else None,
            "agreed_vs_disagreed": self.agreed_vs_disagreed,
            "pairwise": self.pairwise,
        }

    def to_markdown(self, digits: int = 3) -> str:
        """Render the gate summary as Markdown."""
        lines = [
            f"### Consensus gate (`{self.rule}`, {len(self.systems)} systems)",
            "",
            f"Units: {self.n_units}  |  routed to review: {self.n_review} "
            f"({self.review_fraction:.1%})"
            if self.n_units
            else "No units.",
            "",
        ]
        avd = self.agreed_vs_disagreed
        if avd:
            lines += [
                "| unit group | n | accuracy vs gold |",
                "| --- | --- | --- |",
            ]
            for group in ("agreed", "disagreed"):
                g = avd.get(group, {})
                acc = g.get("accuracy")
                shown = "n/a" if acc is None else f"{acc:.{digits}f}"
                lines += [f"| {group} | {g.get('n', 0)} | {shown} |"]
            lift = avd.get("accuracy_lift")
            if lift is not None:
                lines += [
                    "",
                    f"Accuracy on agreed units exceeds disagreed units by {lift:.{digits}f}. "
                    "A lift near zero means the gate is not separating reliable from "
                    "unreliable units.",
                ]
        if self.pairwise:
            lines += [
                "",
                "| system pair | n | kappa | observed agreement |",
                "| --- | --- | --- | --- |",
            ]
            for pair, vals in sorted(self.pairwise.items()):
                k = vals.get("cohen_kappa")
                o = vals.get("observed_agreement")
                lines += [
                    f"| {pair} | {vals.get('n_units', 0)} | "
                    f"{'n/a' if k is None else f'{k:.{digits}f}'} | "
                    f"{'n/a' if o is None else f'{o:.{digits}f}'} |"
                ]
        if self.agreement:
            lines += ["", self.agreement.to_markdown(digits=digits)]
        return "\n".join(lines)


def apply_rule(n_positive: int, n_systems: int, rule: ConsensusRule) -> tuple[bool | None, bool]:
    """Apply a consensus rule to one unit's votes.

    Returns:
        ``(consensus_value, needs_review)``. ``consensus_value`` is ``None``
        exactly when the rule cannot decide, which under ``majority`` means an
        even split. ``needs_review`` is true when the systems were not
        unanimous, regardless of whether the rule reached a verdict -- a 2-1
        majority still has a dissenter worth a human's time.
    """
    if n_systems <= 0:
        return None, True
    unanimous = n_positive == 0 or n_positive == n_systems

    if rule == "unanimous":
        return (n_positive == n_systems), not unanimous
    if rule == "any":
        return (n_positive > 0), not unanimous
    if rule == "majority":
        if n_positive * 2 == n_systems:
            return None, True
        return (n_positive * 2 > n_systems), not unanimous
    raise ConfigError(f"unknown consensus rule {rule!r}; use 'majority', 'unanimous', or 'any'")


def _to_windowed(
    name: str,
    source: SystemInput,
    golds: Sequence[Annotation] | None,
    config: EvalConfig,
    index: pd.DataFrame | None,
) -> pd.DataFrame:
    """Normalize any accepted system input to unit-level rows."""
    if isinstance(source, WindowedFrame):
        df = source.df.copy()
    elif isinstance(source, AlignedFrame):
        df = aggregate_to_windows(source, config.window, index=index, unit=config.unit).df.copy()
    else:
        if golds is None:
            raise ConfigError(
                f"system {name!r} was given as raw predictions, so gold annotations are "
                "needed to align them; pass annotations= or supply an AlignedFrame"
            )
        aligned = align_predictions(list(source), list(golds), config)
        df = aggregate_to_windows(aligned, config.window, index=index, unit=config.unit).df.copy()
    df["system"] = name
    return df


def consensus_gate(
    systems: Mapping[str, SystemInput],
    annotations: Sequence[Annotation] | None = None,
    *,
    rule: ConsensusRule = "majority",
    config: EvalConfig | None = None,
    index: pd.DataFrame | None = None,
    bootstrap: Bootstrap | None = None,
) -> ConsensusResult:
    """Combine several systems' unit-level calls under a consensus rule.

    Args:
        systems: Mapping of system name to its predictions, aligned frame, or
            windowed frame. At least two are required -- a one-system
            "consensus" is just that system, and silently accepting it would
            produce a meaningless review fraction of zero.
        annotations: Gold annotations. Needed when ``systems`` holds raw
            predictions, and needed for any accuracy reporting.
        rule: ``majority``, ``unanimous``, or ``any``.
        config: Evaluation config, for unit and matching policy.
        index: Per-patient index dates, when the unit is ``patient_window``.
        bootstrap: Bootstrap settings for the consensus metrics.

    Returns:
        A :class:`ConsensusResult`.
    """
    if len(systems) < 2:
        raise ConfigError(
            f"consensus needs at least 2 systems, got {len(systems)}: {sorted(systems)}"
        )
    config = config or EvalConfig()
    boot = bootstrap or config.bootstrap

    frames = {
        name: _to_windowed(name, src, annotations, config, index)
        for name, src in sorted(systems.items())
    }
    names = tuple(sorted(frames))

    key_cols = ["unit_key", "label"]
    carry = ["patient_id", "window_index", "window_label", "gold", "gold_present"]
    base: pd.DataFrame | None = None
    votes: dict[str, pd.Series] = {}

    for name, df in frames.items():
        if not len(df):
            continue
        keep = key_cols + [c for c in carry if c in df.columns]
        side = df[keep].drop_duplicates(subset=key_cols).set_index(key_cols)
        base = side if base is None else base.combine_first(side)
        votes[name] = df.set_index(key_cols)["pred"]

    if base is None:
        empty = pd.DataFrame(
            columns=[*key_cols, "n_systems", "n_positive", "consensus", "needs_review"]
        )
        return ConsensusResult(df=empty, rule=rule, systems=names)

    out = base.copy()
    for name in names:
        out[f"pred_{name}"] = votes.get(name)

    vote_cols = [f"pred_{n}" for n in names]
    present = out[vote_cols].notna()
    out["n_systems"] = present.sum(axis=1).astype(int)
    out["n_positive"] = (
        out[vote_cols].apply(lambda s: s.fillna(False).astype(bool)).sum(axis=1).astype(int)
    )

    verdicts = [
        apply_rule(int(p), int(n), rule)
        for p, n in zip(out["n_positive"], out["n_systems"], strict=True)
    ]
    out["consensus"] = pd.array([v for v, _ in verdicts], dtype="boolean")
    out["needs_review"] = [r for _, r in verdicts]
    out = out.reset_index()

    # Accuracy of the gate, and the comparison that justifies it.
    agreement: AgreementReport | None = None
    avd: dict[str, Any] = {}
    if "gold" in out.columns and out["gold"].notna().any():
        scored = out.copy()
        scored["pred"] = scored["consensus"]
        scored["pred_present"] = scored["consensus"].notna()
        scored["gold_present"] = scored["gold"].notna()
        agreement = compute_agreement(
            scored, system=f"consensus:{rule}", metrics=config.metrics, bootstrap=boot
        )
        avd = _agreed_vs_disagreed(out, names)

    return ConsensusResult(
        df=out,
        rule=rule,
        systems=names,
        agreement=agreement,
        agreed_vs_disagreed=avd,
        pairwise=pairwise_agreement(out, names),
    )


def _agreed_vs_disagreed(out: pd.DataFrame, names: tuple[str, ...]) -> dict[str, Any]:
    """Accuracy on unanimous units versus contested ones.

    On agreed units the systems share one value, so that value is the
    prediction. On disagreed units there is no single system call, so accuracy
    is measured against the rule's verdict, and units the rule could not decide
    are excluded and counted.
    """
    vote_cols = [f"pred_{n}" for n in names]
    unanimous_mask = ~out["needs_review"].astype(bool)
    result: dict[str, Any] = {}

    agreed = out[unanimous_mask & out["gold"].notna()]
    if len(agreed):
        conf = Confusion.from_frame(
            agreed.assign(pred=agreed[vote_cols[0]].fillna(False).astype(bool))
        )
        result["agreed"] = {
            "n": conf.n,
            "accuracy": _nn((conf.tp + conf.tn) / conf.n if conf.n else float("nan")),
            "confusion": conf.as_dict(),
        }
    else:
        result["agreed"] = {"n": 0, "accuracy": None}

    disagreed = out[(~unanimous_mask) & out["gold"].notna()]
    undecided = int(disagreed["consensus"].isna().sum()) if len(disagreed) else 0
    decided = disagreed[disagreed["consensus"].notna()] if len(disagreed) else disagreed
    if len(decided):
        conf = Confusion.from_frame(decided.assign(pred=decided["consensus"]))
        result["disagreed"] = {
            "n": conf.n,
            "accuracy": _nn((conf.tp + conf.tn) / conf.n if conf.n else float("nan")),
            "confusion": conf.as_dict(),
            "undecided_excluded": undecided,
        }
    else:
        result["disagreed"] = {"n": 0, "accuracy": None, "undecided_excluded": undecided}

    a, d = result["agreed"]["accuracy"], result["disagreed"]["accuracy"]
    result["accuracy_lift"] = None if a is None or d is None else a - d
    return result


def pairwise_agreement(out: pd.DataFrame, names: tuple[str, ...]) -> dict[str, Any]:
    """Between-system agreement on the units both systems covered.

    Gold-free: this measures whether two systems say the same thing, which is a
    different question from whether either is right.
    """
    res: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca, cb = f"pred_{a}", f"pred_{b}"
            if ca not in out.columns or cb not in out.columns:
                continue
            both = out[out[ca].notna() & out[cb].notna()]
            if not len(both):
                res[f"{a}|{b}"] = {"n_units": 0, "note": "no shared units"}
                continue
            conf = Confusion.from_frame(both, pred=ca, gold=cb)
            res[f"{a}|{b}"] = {
                "n_units": conf.n,
                "confusion": conf.as_dict(),
                "cohen_kappa": _nn(cohen_kappa(conf)),
                "pabak": _nn(pabak(conf)),
                "observed_agreement": _nn((conf.tp + conf.tn) / conf.n if conf.n else float("nan")),
            }
    return res


def _nn(x: float) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


__all__ = [
    "CONSENSUS_RULES",
    "ConsensusResult",
    "ConsensusRule",
    "Unit",
    "apply_rule",
    "consensus_gate",
    "pairwise_agreement",
]
