"""The frozen-dev protocol: tune on dev, freeze, evaluate on test.

SPEC §4.5. The protocol exists because threshold tuning is the easiest way to
overfit a validation study without noticing. Three mechanical guards, all
enforced in code rather than by convention:

1. :func:`tune` refuses to run on a config whose ``split`` is ``test``.
2. :func:`freeze` records a SHA-256 of the config body.
3. :func:`load_frozen` recomputes that hash and refuses a config that was
   edited after freezing.

What the protocol does *not* claim: it cannot tell whether the dev and test
files really hold different patients. That is the user's responsibility, and
:func:`check_split_disjoint` is provided to make it checkable rather than
assumed.
"""

from __future__ import annotations

import itertools
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from clinval.aggregate import WindowedFrame, aggregate_to_windows, align_predictions
from clinval.agreement import METRIC_FUNCS, Confusion
from clinval.exceptions import ConfigError, FrozenConfigError, SplitGuardError
from clinval.io import read_json, write_json
from clinval.schemas.annotation import Annotation
from clinval.schemas.config import EvalConfig, FrozenConfig, Split
from clinval.schemas.prediction import Prediction


def git_commit(path: str | Path | None = None) -> str | None:
    """Current git commit of ``path``'s repository, or ``None`` outside one.

    Recorded in the freeze manifest so a result can be traced back to the code
    that produced it. Absence is not an error -- clinval is routinely run
    outside a repo.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


@dataclass
class TuningResult:
    """Everything ``tune`` tried and what it chose.

    Keeping every candidate, not just the winner, is what makes the tuning
    auditable: a reviewer can see whether the objective was flat across the
    grid, in which case the chosen value is arbitrary and should be reported as
    such.
    """

    config: EvalConfig
    objective: str
    best_score: float
    best_params: dict[str, Any]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    n_evaluated: int = 0
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        """Whether every candidate scored within 0.005 of the winner.

        A flat grid means the parameter did not matter on dev. Reporting the
        winner as if it were a finding would be overclaiming.
        """
        scores = [c["score"] for c in self.candidates if not math.isnan(c["score"])]
        if len(scores) < 2:
            return False
        return (max(scores) - min(scores)) < 0.005

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary for the freeze manifest."""
        return {
            "objective": self.objective,
            "best_score": None if math.isnan(self.best_score) else self.best_score,
            "best_params": self.best_params,
            "n_evaluated": self.n_evaluated,
            "is_flat": self.is_flat,
            "note": self.note,
            "warnings": self.warnings,
            "candidates": self.candidates,
        }

    def to_markdown(self, digits: int = 4) -> str:
        """Render the grid search as a Markdown table."""
        if not self.candidates:
            return "_No tuning was performed; the config declared no grid._"
        keys = sorted({k for c in self.candidates for k in c.get("params", {})})
        has_decided = any("decided_fraction" in c for c in self.candidates)
        header = "| " + " | ".join(keys) + f" | {self.objective} |"
        divider = "| " + " | ".join("---" for _ in keys) + " | --- |"
        if has_decided:
            header += " decided | "
            divider += " --- |"
        header += " chosen |"
        divider += " --- |"

        lines = [
            f"Objective: **{self.objective}**, maximized over {self.n_evaluated} candidates.",
            "",
            header,
            divider,
        ]
        for c in self.candidates:
            row = [str(c["params"].get(k, "")) for k in keys]
            score = c["score"]
            shown = "n/a" if score is None or math.isnan(score) else f"{score:.{digits}f}"
            cells = "| " + " | ".join(row) + f" | {shown} |"
            if has_decided:
                frac = c.get("decided_fraction")
                cells += (
                    f" {frac:.1%} |"
                    if isinstance(frac, float) and not math.isnan(frac)
                    else " n/a |"
                )
            cells += " yes |" if c["params"] == self.best_params else "  |"
            lines += [cells]

        if self.is_flat:
            lines += [
                "",
                "> The objective varies by less than 0.005 across the whole grid, so the "
                "chosen value is effectively arbitrary on this dev set. Report it as a "
                "default, not as a tuned finding.",
            ]
        for w in self.warnings:
            lines += ["", f"> {w}"]
        return "\n".join(lines)


def _grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid, in deterministic order."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    out: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        out.append(dict(zip(keys, values, strict=True)))
    return out


def _score_candidate(
    config: EvalConfig,
    predictions: Sequence[Prediction],
    annotations: Sequence[Annotation],
    index: pd.DataFrame | None,
    objective: str,
    system: str | None,
    *,
    score_consensus: bool = False,
) -> tuple[float, str, dict[str, Any]]:
    """Score one candidate config on dev data.

    Three scoring targets, chosen by what the grid actually varies:

    * ``score_consensus`` -- the grid tunes the consensus rule, so the
      objective is the *gate's* accuracy against gold, not any one system's.
      This needs two or more systems.
    * ``system`` named -- that system alone.
    * neither -- the mean of the per-system scores. This tunes the harness to
      work across systems rather than to flatter one of them, and the choice
      is recorded in the note so the report can state it.
    """
    aligned = align_predictions(list(predictions), list(annotations), config)
    windowed = aggregate_to_windows(aligned, config.window, index=index, unit=config.unit)
    df = windowed.df
    if not len(df):
        return float("nan"), "no scorable units", {}

    fn = METRIC_FUNCS[objective]
    systems = sorted(df["system"].dropna().unique().tolist())

    if score_consensus:
        if len(systems) < 2:
            raise ConfigError(
                "the tuning grid varies `consensus`, which needs predictions from at "
                f"least two systems; found {systems}. Either supply a second system's "
                "predictions or drop `consensus` from freeze.grid."
            )
        from clinval.consensus import consensus_gate

        frames = {s: WindowedFrame(df=df[df["system"] == s], unit=windowed.unit) for s in systems}
        result = consensus_gate(
            frames,  # type: ignore[arg-type]
            annotations,
            rule=config.consensus,
            config=config,
            index=index,
            # Tuning compares point estimates, so the per-candidate bootstrap
            # is wasted work -- a grid of 6 would otherwise pay for 6 full
            # interval computations that are never read.
            bootstrap=config.bootstrap.model_copy(update={"n": 1}),
        )
        if result.agreement is None:
            return (
                float("nan"),
                f"consensus rule={config.consensus}: no gold to score against",
                {},
            )
        val = fn(result.agreement.confusion)
        # A rule that declines to decide is scored only on what it decided, so
        # a more deferring rule can post a better number while answering fewer
        # questions. Carry the decided fraction so `tune` can say so.
        decided = result.agreement.confusion.n
        total = result.n_units
        return (
            float(val),
            (
                f"consensus rule={config.consensus} over {len(systems)} systems; "
                f"decided {decided} of {total} units, "
                f"{result.n_review} routed to review"
            ),
            {
                "n_decided": decided,
                "n_units": total,
                "decided_fraction": (decided / total) if total else float("nan"),
                "n_review": result.n_review,
            },
        )

    if system is not None:
        if system not in systems:
            raise ConfigError(
                f"system {system!r} is not present in the predictions; found {systems}"
            )
        rows = df[df["system"] == system]
        conf = Confusion.from_frame(rows)
        return float(fn(conf)), f"system={system}", {"n_decided": conf.n, "n_units": len(rows)}

    scores: list[float] = []
    for s in systems:
        rows = df[df["system"] == s]
        val = fn(Confusion.from_frame(rows))
        if not math.isnan(val):
            scores.append(val)
    if not scores:
        return float("nan"), "objective undefined for every system", {}
    if len(scores) == 1:
        return scores[0], f"system={systems[0]}", {}
    return (
        sum(scores) / len(scores),
        f"mean over {len(scores)} systems: {systems}",
        {},
    )


def tune(
    config: EvalConfig,
    predictions: Sequence[Prediction],
    annotations: Sequence[Annotation],
    *,
    index: pd.DataFrame | None = None,
    system: str | None = None,
    allow_test: bool = False,
) -> TuningResult:
    """Grid-search the declared parameters on dev data.

    Args:
        config: Config whose ``freeze.grid`` declares the search space.
        predictions: Dev predictions.
        annotations: Dev gold.
        index: Per-patient index dates, when the unit is ``patient_window``.
        system: Tune against one named system instead of the cross-system mean.
        allow_test: Escape hatch for tuning a config marked ``test``. Off by
            default and never set by the CLI; it exists so a test suite can
            exercise the guard.

    Returns:
        A :class:`TuningResult` whose ``config`` carries the chosen values and
        is still marked ``split: dev``.

    Raises:
        SplitGuardError: If ``config.split`` is ``test`` and ``allow_test`` is False.
    """
    if config.split is Split.TEST and not allow_test:
        raise SplitGuardError(
            "refusing to tune on a config marked `split: test`. Tuning on test data is "
            "the failure this protocol exists to prevent. Set `split: dev` in the config "
            "and point it at your dev files."
        )

    objective = config.freeze.objective
    if objective not in METRIC_FUNCS:
        raise ConfigError(f"unknown tuning objective {objective!r}")

    combos = _grid_combinations(config.freeze.grid)
    score_consensus = "consensus" in config.freeze.grid
    candidates: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    best_config = config
    note = ""

    for params in combos:
        candidate = config.with_overrides(**params) if params else config
        score, how, diag = _score_candidate(
            candidate,
            predictions,
            annotations,
            index,
            objective,
            system,
            score_consensus=score_consensus,
        )
        note = how
        candidates.append({"params": params, "score": score, "scored_over": how, **diag})
        if not math.isnan(score) and score > best_score:
            best_score, best_params, best_config = score, params, candidate

    if not candidates or math.isnan(best_score) or best_score == float("-inf"):
        raise ConfigError(
            f"tuning produced no usable score for objective {objective!r}. "
            "Check that the dev predictions and gold overlap, and that the objective "
            "is defined on the resulting confusion table (specificity needs gold negatives, "
            "sensitivity needs gold positives)."
        )

    warnings = _deferral_warnings(candidates, best_params, objective)
    tuned = best_config.model_copy(update={"split": Split.DEV})
    return TuningResult(
        config=tuned,
        objective=objective,
        best_score=best_score,
        best_params=best_params,
        candidates=candidates,
        n_evaluated=len(candidates),
        note=note,
        warnings=warnings,
    )


def _deferral_warnings(
    candidates: list[dict[str, Any]], best_params: dict[str, Any], objective: str
) -> list[str]:
    """Flag a winner that scored well mainly by answering fewer questions.

    A consensus rule that declines to decide is scored only on the units it
    decided, so deferring more can buy a better number. That is a real
    trade-off, not a free win, and a tuning record that hides it is the kind of
    thing this harness is supposed to prevent.
    """
    out: list[str] = []
    scored = [c for c in candidates if isinstance(c.get("decided_fraction"), float)]
    if len(scored) < 2:
        return out
    best = next((c for c in scored if c["params"] == best_params), None)
    if best is None or math.isnan(best["decided_fraction"]):
        return out

    most_decisive = max(scored, key=lambda c: c["decided_fraction"])
    gap = most_decisive["decided_fraction"] - best["decided_fraction"]
    if gap > 0.02:
        alt = most_decisive
        out.append(
            f"The winning setting {best['params']} decided only "
            f"{best['decided_fraction']:.1%} of units, while {alt['params']} decided "
            f"{alt['decided_fraction']:.1%} at {objective}="
            f"{alt['score']:.4f}. Part of the winner's advantage is that undecided "
            "units are excluded from its score rather than counted against it. Choose "
            "on review burden as well as on the metric."
        )
    return out


def freeze(
    config: EvalConfig,
    *,
    tuning: TuningResult | None = None,
    repo_path: str | Path | None = None,
) -> FrozenConfig:
    """Lock a config and record its hash.

    The frozen config's ``split`` is forced to ``test``: a config is frozen in
    order to be run out of sample, so carrying ``dev`` forward would let
    :func:`evaluate_guard` pass for the wrong reason.
    """
    locked = config.model_copy(update={"split": Split.TEST})
    return FrozenConfig.create(
        locked,
        git_commit=git_commit(repo_path),
        tuning_summary=tuning.as_dict() if tuning else None,
    )


def write_frozen(frozen: FrozenConfig, path: str | Path) -> Path:
    """Write a freeze manifest to JSON."""
    write_json(frozen.model_dump(mode="json"), path)
    return Path(path)


def load_frozen(path: str | Path) -> FrozenConfig:
    """Load a freeze manifest and verify its hash.

    Raises:
        FrozenConfigError: If the file is not a manifest, or the recorded hash
            does not match the config body -- which is what tampering looks
            like.
    """
    raw = read_json(path)
    if not isinstance(raw, dict) or "config_hash" not in raw:
        raise FrozenConfigError(
            f"{path} is not a clinval freeze manifest (no `config_hash` key). "
            "Produce one with `clinval freeze`."
        )
    try:
        frozen = FrozenConfig.model_validate(raw)
    except Exception as exc:
        raise FrozenConfigError(f"malformed freeze manifest {path}: {exc}") from exc
    frozen.verify()
    return frozen


def evaluate_guard(
    config: EvalConfig,
    *,
    frozen: FrozenConfig | None,
    acknowledge_dev: bool = False,
) -> None:
    """Enforce the protocol at evaluation time.

    Three outcomes:

    * A verified frozen config -- always allowed.
    * An unfrozen config marked ``dev`` -- allowed only with
      ``acknowledge_dev``, which the CLI surfaces as
      ``--i-understand-this-is-dev``.
    * An unfrozen config marked ``test`` -- always refused.

    Raises:
        SplitGuardError: When the combination is not permitted.
    """
    if frozen is not None:
        frozen.verify()
        return
    if config.split is Split.TEST:
        raise SplitGuardError(
            "refusing to evaluate `split: test` data without a frozen config.\n"
            "Run the protocol in order:\n"
            "  clinval tune   --config eval.yaml --pred dev_pred.jsonl  --gold dev_gold.jsonl "
            "--out tuned.yaml\n"
            "  clinval freeze --config tuned.yaml --out frozen.json\n"
            "  clinval evaluate --frozen frozen.json --pred test_pred.jsonl "
            "--gold test_gold.jsonl --out report/"
        )
    if not acknowledge_dev:
        raise SplitGuardError(
            "this config is marked `split: dev`, so any result from it is an in-sample "
            "estimate and is not a validation finding. Pass "
            "--i-understand-this-is-dev to run it anyway."
        )


def check_split_disjoint(
    dev_annotations: Sequence[Annotation],
    test_annotations: Sequence[Annotation],
) -> dict[str, Any]:
    """Check that dev and test share no patients and no notes.

    clinval cannot enforce this -- it only sees the files it is handed -- so it
    reports it instead, and the report prints the result. A shared patient
    means the tuned threshold was chosen on a patient it was then tested on,
    and the out-of-sample claim is void.

    Returns:
        ``{"ok": bool, "shared_patients": [...], "shared_notes": [...], ...}``.
    """
    dev_p = {a.patient_id for a in dev_annotations if a.patient_id}
    test_p = {a.patient_id for a in test_annotations if a.patient_id}
    dev_n = {a.note_id for a in dev_annotations}
    test_n = {a.note_id for a in test_annotations}

    shared_p = sorted(dev_p & test_p)
    shared_n = sorted(dev_n & test_n)
    return {
        "ok": not shared_p and not shared_n,
        "n_dev_patients": len(dev_p),
        "n_test_patients": len(test_p),
        "shared_patients": shared_p[:50],
        "n_shared_patients": len(shared_p),
        "shared_notes": shared_n[:50],
        "n_shared_notes": len(shared_n),
        "patient_ids_available": bool(dev_p or test_p),
    }
