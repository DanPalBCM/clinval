"""Token, latency, and cost accounting.

The rule here is one line long and it is the whole module's reason for
existing: **a missing token count is reported as unknown, never as zero.**

Treating absent metadata as zero would make the cheapest-looking system the one
that reports the least, which is exactly backwards. The bundled ``langextract``
adapter is the live example -- that library exposes no token counts at all (see
``docs/decisions.md`` D-003), so its predictions carry ``latency_s`` and
nothing else, and :func:`summarize_cost` says so explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from clinval.schemas.config import CostModel
from clinval.schemas.prediction import Prediction


@dataclass
class SystemCost:
    """Cost and latency for one system.

    Attributes:
        system: System name.
        n_records: Billable units seen -- distinct model calls when the records
            carry a ``call_id``, otherwise prediction records.
        n_calls: Number of distinct ``call_id`` values, or ``None`` when the
            records do not use them.
        n_with_tokens: Billable units carrying both token counts.
        n_with_latency: Billable units carrying a latency.
        tokens_in: Summed input tokens over records that reported them.
        tokens_out: Summed output tokens over records that reported them.
        cost: Estimated USD, or ``None`` when no record reported tokens.
        cost_status: ``complete``, ``partial``, or ``unknown``.
        latency_total_s: Summed wall-clock seconds over reporting records.
    """

    system: str
    n_records: int = 0
    n_calls: int | None = None
    n_with_tokens: int = 0
    n_with_latency: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float | None = None
    cost_status: str = "unknown"
    cost_note: str = ""
    latency_total_s: float = 0.0
    latency_mean_s: float | None = None
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    latency_max_s: float | None = None
    models: tuple[str, ...] = ()
    prompt_versions: tuple[str, ...] = ()

    @property
    def token_coverage(self) -> float:
        """Share of records that reported token counts."""
        return self.n_with_tokens / self.n_records if self.n_records else float("nan")

    @property
    def cost_per_record(self) -> float | None:
        """Estimated USD per *reporting* record, or ``None`` when unknown.

        Divided by reporting records rather than all records, because dividing
        a partial total by the full denominator understates unit cost.
        """
        if self.cost is None or self.n_with_tokens == 0:
            return None
        return self.cost / self.n_with_tokens

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {
            "system": self.system,
            "n_records": self.n_records,
            "n_calls": self.n_calls,
            "n_with_tokens": self.n_with_tokens,
            "n_with_latency": self.n_with_latency,
            "token_coverage": None if math.isnan(self.token_coverage) else self.token_coverage,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "cost_status": self.cost_status,
            "cost_note": self.cost_note,
            "cost_per_record": self.cost_per_record,
            "latency_total_s": self.latency_total_s,
            "latency_mean_s": self.latency_mean_s,
            "latency_p50_s": self.latency_p50_s,
            "latency_p95_s": self.latency_p95_s,
            "latency_max_s": self.latency_max_s,
            "models": list(self.models),
            "prompt_versions": list(self.prompt_versions),
        }


@dataclass
class CostReport:
    """Cost and latency across all systems."""

    per_system: dict[str, SystemCost] = field(default_factory=dict)
    model: CostModel = field(default_factory=CostModel)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {
            "rates": {
                "input_per_1k": self.model.input_per_1k,
                "output_per_1k": self.model.output_per_1k,
                "currency": self.model.currency,
            },
            "per_system": {k: v.as_dict() for k, v in sorted(self.per_system.items())},
            "any_unknown": any(s.cost_status != "complete" for s in self.per_system.values()),
        }

    def to_markdown(self, digits: int = 4) -> str:
        """Render the cost table, naming unknowns rather than printing zeros."""
        if not self.per_system:
            return "_No run metadata was supplied, so no cost or latency is reported._"
        lines = [
            "| system | records | token coverage | tokens in | tokens out | "
            f"cost ({self.model.currency}) | mean latency (s) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, s in sorted(self.per_system.items()):
            cov = "n/a" if math.isnan(s.token_coverage) else f"{s.token_coverage:.0%}"
            if s.cost is None:
                cost_cell = "unknown"
            elif s.cost_status == "partial":
                cost_cell = f"≥{s.cost:.{digits}f} (partial)"
            else:
                cost_cell = f"{s.cost:.{digits}f}"
            lat = "n/a" if s.latency_mean_s is None else f"{s.latency_mean_s:.3f}"
            lines += [
                f"| {name} | {s.n_records} | {cov} | {s.tokens_in} | {s.tokens_out} | "
                f"{cost_cell} | {lat} |"
            ]
        notes = [
            f"- **{n}**: {s.cost_note}" for n, s in sorted(self.per_system.items()) if s.cost_note
        ]
        if notes:
            lines += ["", *notes]
        return "\n".join(lines)


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile without pulling numpy into this module."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def token_cost(tokens_in: int, tokens_out: int, model: CostModel) -> float:
    """USD for a token count at the configured per-1k rates."""
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError(f"token counts must be non-negative, got {tokens_in} / {tokens_out}")
    return (tokens_in / 1000.0) * model.input_per_1k + (tokens_out / 1000.0) * model.output_per_1k


def summarize_cost(
    predictions: Sequence[Prediction],
    model: CostModel | None = None,
) -> CostReport:
    """Aggregate run metadata into per-system cost and latency.

    ``cost_status`` is the field to read:

    * ``complete`` -- every record reported tokens.
    * ``partial`` -- some did. The cost is a lower bound and is printed with a
      ``≥``.
    * ``unknown`` -- none did. ``cost`` is ``None``, and ``cost_note`` names
      the reason.

    Args:
        predictions: Predictions, whose ``run`` metadata is read.
        model: Rates. Defaults to zero rates, in which case any computed cost
            is zero and ``cost_note`` says the rates were unset -- which is a
            different statement from the tokens being unknown.

    Returns:
        A :class:`CostReport`.
    """
    model = model or CostModel()
    by_system: dict[str, list[Prediction]] = {}
    for p in predictions:
        by_system.setdefault(p.system, []).append(p)

    out: dict[str, SystemCost] = {}
    for system, recs in by_system.items():
        tokens_in = tokens_out = 0
        n_tokens = 0
        latencies: list[float] = []
        models: set[str] = set()
        prompts: set[str] = set()
        # Records that name the same model call are charged once. Without this,
        # an extractor that answers five labels in one pass and stamps the
        # call's tokens on all five records would report five times its cost.
        counted_calls: set[str] = set()
        n_calls = 0

        for p in recs:
            run = p.run
            if run is None:
                continue
            if run.model:
                models.add(run.model)
            if run.prompt_version:
                prompts.add(run.prompt_version)

            call_id = run.call_id
            if call_id is not None:
                if call_id in counted_calls:
                    continue
                counted_calls.add(call_id)
            n_calls += 1

            if run.latency_s is not None:
                latencies.append(float(run.latency_s))
            if run.tokens_in is not None and run.tokens_out is not None:
                tokens_in += int(run.tokens_in)
                tokens_out += int(run.tokens_out)
                n_tokens += 1

        # The denominator for coverage is the number of billable calls, not the
        # number of records, whenever call_id is in use.
        n = n_calls if counted_calls else len(recs)
        if n_tokens == 0:
            cost: float | None = None
            status = "unknown"
            note = (
                "no record reported token counts, so cost cannot be estimated. "
                "This is expected for extractors that do not expose usage metadata "
                "(langextract, for one); cost is reported as unknown rather than zero."
            )
        elif n_tokens < n:
            cost = token_cost(tokens_in, tokens_out, model)
            status = "partial"
            note = (
                f"{n_tokens} of {n} records reported token counts, so the cost is a "
                "lower bound over the reporting subset."
            )
        else:
            cost = token_cost(tokens_in, tokens_out, model)
            status = "complete"
            note = ""

        if cost is not None and model.input_per_1k == 0.0 and model.output_per_1k == 0.0:
            note = (
                (note + " ") if note else ""
            ) + "Token rates are unset in the config, so the cost figure is 0 by construction."

        out[system] = SystemCost(
            system=system,
            n_records=n,
            n_calls=len(counted_calls) if counted_calls else None,
            n_with_tokens=n_tokens,
            n_with_latency=len(latencies),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cost_status=status,
            cost_note=note,
            latency_total_s=sum(latencies),
            latency_mean_s=(sum(latencies) / len(latencies)) if latencies else None,
            latency_p50_s=_percentile(latencies, 0.50),
            latency_p95_s=_percentile(latencies, 0.95),
            latency_max_s=max(latencies) if latencies else None,
            models=tuple(sorted(models)),
            prompt_versions=tuple(sorted(prompts)),
        )
    return CostReport(per_system=out, model=model)
