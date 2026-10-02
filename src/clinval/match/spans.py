"""Span matching: exact, overlap, and fuzzy.

Span agreement is reported separately from label agreement. A system can be
right about the phenotype and wrong about which sentence proves it, and the
error taxonomy (:mod:`clinval.errors`) needs to tell those apart -- that
distinction is what separates a boundary disagreement from a misreading.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from clinval.schemas.config import SpanMatching, SpanMode
from clinval.schemas.prediction import Span


@dataclass(frozen=True, slots=True)
class SpanMatchResult:
    """The outcome of comparing one predicted span with one gold span.

    Attributes:
        matched: Whether the spans agree under the configured mode.
        mode: The mode used.
        jaccard: Character-level Jaccard overlap, always computed when both
            spans exist, regardless of mode. ``0.0`` when they are disjoint.
        fuzzy_score: Normalized text similarity in ``[0, 1]``, or ``None`` when
            either span carries no text.
        reason: Short machine-readable explanation, for error tagging.
    """

    matched: bool
    mode: SpanMode
    jaccard: float = 0.0
    fuzzy_score: float | None = None
    reason: str = ""


def jaccard_overlap(a: Span, b: Span) -> float:
    """Character-level Jaccard index of two spans.

    Defined as ``|intersection| / |union|`` over character positions. Two
    zero-length spans have an empty union; that is reported as ``0.0`` rather
    than raising, since a zero-length span carries no extent to agree about.
    """
    inter_start = max(a.start, b.start)
    inter_end = min(a.end, b.end)
    intersection = max(0, inter_end - inter_start)
    union = (a.length + b.length) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def overlap_fraction(a: Span, b: Span, *, relative_to: str = "min") -> float:
    """Overlap as a fraction of one span's length.

    Args:
        a: First span.
        b: Second span.
        relative_to: ``"min"``, ``"max"``, ``"a"``, or ``"b"`` -- which length
            is the denominator.
    """
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    denominators = {
        "min": min(a.length, b.length),
        "max": max(a.length, b.length),
        "a": a.length,
        "b": b.length,
    }
    if relative_to not in denominators:
        raise ValueError(f"relative_to must be one of {sorted(denominators)}, got {relative_to!r}")
    denom = denominators[relative_to]
    return inter / denom if denom > 0 else 0.0


def fuzzy_text_score(a: Span, b: Span) -> float | None:
    """Normalized similarity of two spans' quoted text, or ``None`` if absent.

    Uses token-set ratio, which ignores word order and duplicates, so
    ``"loose stools, watery"`` and ``"watery loose stools"`` score 1.0.

    Text is case-folded first. rapidfuzz is case-sensitive by default, and
    leaving it that way would penalize a sentence-initial capital -- an
    extractor quoting ``"Diarrhea since Monday"`` against an annotator's
    ``"diarrhea since Monday"`` is agreement, not a near miss.

    Accents are **not** folded. Case carries no clinical meaning, but accent
    folding can merge genuinely distinct terms in some languages, so it stays a
    policy decision rather than a silent default.
    """
    if a.text is None or b.text is None:
        return None
    return fuzz.token_set_ratio(a.text.casefold(), b.text.casefold()) / 100.0


def match_spans(
    pred: Span | None,
    gold: Span | None,
    policy: SpanMatching,
) -> SpanMatchResult:
    """Compare a predicted span with a gold span under ``policy``.

    Missing spans are not failures by default. A system may assert a label
    without quoting evidence, and whether that is acceptable is a policy
    question -- set ``policy.require_span`` to make it a failure.

    Returns:
        A :class:`SpanMatchResult`. When either span is absent, ``matched``
        follows ``policy.require_span`` and ``reason`` records which side was
        missing.
    """
    if pred is None and gold is None:
        return SpanMatchResult(
            matched=not policy.require_span, mode=policy.mode, reason="both_spans_absent"
        )
    if pred is None:
        return SpanMatchResult(
            matched=not policy.require_span, mode=policy.mode, reason="pred_span_absent"
        )
    if gold is None:
        return SpanMatchResult(
            matched=not policy.require_span, mode=policy.mode, reason="gold_span_absent"
        )

    jac = jaccard_overlap(pred, gold)
    fuzzy = fuzzy_text_score(pred, gold)

    if policy.mode is SpanMode.EXACT:
        matched = pred.start == gold.start and pred.end == gold.end
        return SpanMatchResult(
            matched=matched,
            mode=policy.mode,
            jaccard=jac,
            fuzzy_score=fuzzy,
            reason="exact_offsets" if matched else "offset_mismatch",
        )

    if policy.mode is SpanMode.OVERLAP:
        matched = jac >= policy.min_overlap
        return SpanMatchResult(
            matched=matched,
            mode=policy.mode,
            jaccard=jac,
            fuzzy_score=fuzzy,
            reason=(
                f"jaccard {jac:.3f} >= {policy.min_overlap:.3f}"
                if matched
                else f"jaccard {jac:.3f} < {policy.min_overlap:.3f}"
            ),
        )

    # SpanMode.FUZZY: offsets may be shifted (an extractor that re-tokenizes
    # loses exact positions), so compare the quoted text and fall back to
    # overlap when text is unavailable on either side.
    if fuzzy is None:
        matched = jac >= policy.min_overlap
        return SpanMatchResult(
            matched=matched,
            mode=policy.mode,
            jaccard=jac,
            fuzzy_score=None,
            reason="fuzzy_requested_but_text_absent_fell_back_to_overlap",
        )
    matched = fuzzy >= policy.fuzzy_threshold
    return SpanMatchResult(
        matched=matched,
        mode=policy.mode,
        jaccard=jac,
        fuzzy_score=fuzzy,
        reason=(
            f"fuzzy {fuzzy:.3f} >= {policy.fuzzy_threshold:.3f}"
            if matched
            else f"fuzzy {fuzzy:.3f} < {policy.fuzzy_threshold:.3f}"
        ),
    )
