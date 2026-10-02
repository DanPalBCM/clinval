"""Error mechanism tagging.

Scoring tells you *how much* a system is wrong. This module is about *why*, and
the distinction drives different fixes: a window-placement error is a pipeline
bug, a misreading is a prompt problem, and an annotation-threshold disagreement
may mean the annotation guideline is underspecified rather than that anybody
erred.

The seven mechanisms are SPEC §4.4. Each is reachable from a distinct
condition, and every tag carries a ``mechanism_reason`` naming the evidence
that produced it -- a tag you cannot audit is worse than no tag.

Rule order matters and is documented per error type below; the first matching
rule wins. ``other`` is the explicit fallback, and a high ``other`` count is a
signal that the taxonomy is missing a mechanism for your data, not that the
errors are inexplicable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from clinval.aggregate import AlignedFrame, WindowedFrame
from clinval.match.ontology import OntologyMap, bundled_map
from clinval.schemas.config import EvalConfig

#: The v1 taxonomy, SPEC §4.4.
MECHANISMS = (
    "annotation_threshold",
    "window_placement",
    "misreading",
    "missing_evidence",
    "ontology_mismatch",
    "temporal_error",
    "other",
)

#: Error types clinval emits. ``FP``/``FN`` are SPEC §6.4. ``MATCH`` is a
#: documented addition (``docs/decisions.md`` D-005): the label is right but
#: the supporting span or code is not, which is a real and separately
#: actionable failure that FP/FN alone cannot express.
ERROR_TYPES = ("FP", "FN", "MATCH")

#: Surface cues that a mention is negated. Deliberately conservative -- these
#: are for *explaining* an error after the fact, never for changing a score.
NEGATION_CUES = (
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bno\s+(?:evidence\s+of\s+|signs?\s+of\s+|complaints?\s+of\s+)?",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bnegative\s+for\b",
    r"\bruled?\s+out\b",
    r"\bfree\s+of\b",
    r"\babsent\b",
    r"\bresolved\b",
    r"\bnone\b",
)

#: Surface cues that a mention refers to another time or another person.
TEMPORAL_CUES = (
    r"\bhistory\s+of\b",
    r"\bh/o\b",
    r"\bpast\s+medical\b",
    r"\bprior\s+to\b",
    r"\bpreviously\b",
    r"\blast\s+(?:year|month|week|visit)\b",
    r"\byears?\s+ago\b",
    r"\bmonths?\s+ago\b",
    r"\bfamily\s+history\b",
    r"\bmother|father|sibling|brother|sister\b",
    r"\bwill\s+(?:likely\s+)?(?:develop|start|begin)\b",
    r"\bplan(?:ned|s)?\s+to\b",
    r"\bif\s+(?:he|she|they|patient)\b",
    r"\brisk\s+of\b",
)

_NEG_RE = re.compile("|".join(NEGATION_CUES), re.IGNORECASE)
_TEMP_RE = re.compile("|".join(TEMPORAL_CUES), re.IGNORECASE)

#: Columns of the error export, SPEC §6.4 first, then clinval's audit columns.
ERROR_COLUMNS = (
    "error_id",
    "system",
    "label",
    "unit_key",
    "error_type",
    "patient_id",
    "window",
    "mechanism",
    "span_text",
    "note_id",
    "mechanism_reason",
    "gold_span_text",
    "span_jaccard",
    "pred_code",
    "gold_code",
    "code_how",
    "confidence",
)


def has_negation_cue(text: str | None) -> bool:
    """Whether a quoted span looks negated."""
    return bool(text) and bool(_NEG_RE.search(text))  # type: ignore[arg-type]


def has_temporal_cue(text: str | None) -> bool:
    """Whether a quoted span refers to another time or another person."""
    return bool(text) and bool(_TEMP_RE.search(text))  # type: ignore[arg-type]


@dataclass
class ErrorTable:
    """Tagged disagreements, one row per error.

    Attributes:
        df: The error rows, columns as in :data:`ERROR_COLUMNS`.
        counts: Mechanism counts, overall.
    """

    df: pd.DataFrame
    counts: dict[str, int]

    def __len__(self) -> int:
        return len(self.df)

    def by_mechanism(self) -> pd.DataFrame:
        """Mechanism counts as a frame, every taxonomy tag present even at zero.

        Printing the zeros matters: a reader can tell "no window-placement
        errors" apart from "window placement was never checked".
        """
        rows = [{"mechanism": m, "n": int(self.counts.get(m, 0))} for m in MECHANISMS]
        return pd.DataFrame(rows)

    def by_label(self) -> pd.DataFrame:
        """Error counts crossed by label and mechanism."""
        if not len(self.df):
            return pd.DataFrame(columns=["label", "mechanism", "n"])
        return (
            self.df.groupby(["label", "mechanism"], observed=True)
            .size()
            .reset_index(name="n")
            .sort_values(["label", "n"], ascending=[True, False])
        )

    def by_window(self) -> pd.DataFrame:
        """Error counts crossed by window and mechanism."""
        if not len(self.df):
            return pd.DataFrame(columns=["window", "mechanism", "n"])
        return (
            self.df.groupby(["window", "mechanism"], observed=True, dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["window", "n"], ascending=[True, False])
        )

    def by_system(self) -> pd.DataFrame:
        """Error counts crossed by system, type, and mechanism."""
        if not len(self.df):
            return pd.DataFrame(columns=["system", "error_type", "mechanism", "n"])
        return (
            self.df.groupby(["system", "error_type", "mechanism"], observed=True)
            .size()
            .reset_index(name="n")
            .sort_values(["system", "error_type", "n"], ascending=[True, True, False])
        )

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """Write the error export to CSV."""
        self.df.to_csv(path, index=False, **kwargs)

    def sample(self, mechanism: str, n: int = 5) -> pd.DataFrame:
        """A few examples of one mechanism, for eyeballing."""
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {mechanism!r}; known: {list(MECHANISMS)}")
        return self.df[self.df["mechanism"] == mechanism].head(n)


def _unit_lookup(windowed: WindowedFrame | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index unit-level rows by (system, note-representative, label) for context."""
    if windowed is None or not len(windowed.df):
        return {}
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in windowed.df.to_dict("records"):
        key = (
            str(row.get("system")),
            str(row.get("representative_note_id")),
            str(row.get("label")),
        )
        out[key] = row
    return out


def _gold_positive_windows(windowed: WindowedFrame | None) -> dict[tuple[Any, str], set[Any]]:
    """Windows where gold is positive, keyed by (patient_id, label)."""
    out: dict[tuple[Any, str], set[Any]] = {}
    if windowed is None or not len(windowed.df):
        return out
    df = windowed.df
    pos = df[df["gold"].fillna(False).astype(bool)]
    for row in pos.itertuples(index=False):
        out.setdefault((row.patient_id, str(row.label)), set()).add(row.window_index)
    return out


def _pred_positive_windows(windowed: WindowedFrame | None) -> dict[tuple[Any, str, Any], set[Any]]:
    """Windows where a system predicted positive, keyed by (patient, label, system)."""
    out: dict[tuple[Any, str, Any], set[Any]] = {}
    if windowed is None or not len(windowed.df):
        return out
    df = windowed.df
    pos = df[df["pred"].fillna(False).astype(bool)]
    for row in pos.itertuples(index=False):
        out.setdefault((row.patient_id, str(row.label), row.system), set()).add(row.window_index)
    return out


def _classify_fp(
    row: dict[str, Any],
    config: EvalConfig,
    gold_pos_windows: dict[tuple[Any, str], set[Any]],
    gold_by_note: dict[tuple[str, str], bool],
    onto: OntologyMap,
) -> tuple[str, str]:
    """Mechanism for a false positive.

    Order: absent evidence, then a negated or out-of-time quote, then a code
    pointing at a different concept, then a window the evidence does not belong
    to, then a boundary disagreement, then ``other``.
    """
    span_text = row.get("pred_span_text")
    policy = config.matching.span

    if pd.isna(row.get("pred_span_start")):
        return "missing_evidence", "positive asserted with no supporting span"

    if has_negation_cue(span_text):
        return "misreading", f"negation cue in quoted span: {_first_cue(_NEG_RE, span_text)!r}"

    if has_temporal_cue(span_text):
        return (
            "temporal_error",
            f"temporal or other-subject cue in quoted span: {_first_cue(_TEMP_RE, span_text)!r}",
        )

    if row.get("code_how") == "mismatch":
        return (
            "ontology_mismatch",
            f"predicted code {row.get('pred_code')} does not match gold {row.get('gold_code')}",
        )

    pred_code = row.get("pred_code")
    if pred_code:
        code = str(pred_code).split("|", 1)[-1]
        concept_labels = {str(onto.display(code) or "").lower()} | {
            s.lower() for s in onto.synonyms(code)
        }
        this_label = str(row.get("label", "")).lower()
        if concept_labels and this_label and this_label not in concept_labels:
            for (pid, lbl), _wins in gold_pos_windows.items():
                if pid == row.get("patient_id") and lbl.lower() in concept_labels:
                    return (
                        "ontology_mismatch",
                        f"code {code} denotes {lbl!r}, which is gold-positive for this "
                        f"patient, but was attached to label {row.get('label')!r}",
                    )

    key = (row.get("patient_id"), str(row.get("label")))
    other_windows = gold_pos_windows.get(key, set())
    this_window = row.get("window_index")
    if other_windows and not _window_in(this_window, other_windows):
        return (
            "window_placement",
            f"gold is positive for this patient and label in window(s) "
            f"{sorted(str(w) for w in other_windows)}, but not in {this_window}",
        )

    jac = row.get("span_jaccard") or 0.0
    if row.get("gold_span_text") is not None and 0.0 < float(jac) < policy.min_overlap:
        return (
            "annotation_threshold",
            f"predicted span overlaps an annotated span at jaccard {float(jac):.3f}, "
            f"below the {policy.min_overlap:.2f} threshold",
        )

    if _is_false(gold_by_note.get((str(row.get("note_id")), str(row.get("label"))))):
        return "other", "annotator reviewed this note-label and called it negative; no cue found"
    return "other", "no mechanism rule matched"


def _classify_fn(
    row: dict[str, Any],
    config: EvalConfig,
    pred_pos_windows: dict[tuple[Any, str, Any], set[Any]],
    onto: OntologyMap,
) -> tuple[str, str]:
    """Mechanism for a false negative.

    Order: an uncredited positive first, then the system found it but filed it
    elsewhere, then an annotated quote the system plausibly excluded as
    historical, then a plain misread of present evidence, then gold with no
    quote at all, then ``other``.
    """
    _ = onto
    # Under `require_span`, a positive whose evidence failed the policy is not
    # credited and scores as a negative. The reason it lost credit -- a
    # boundary, a missing span, a wrong code -- is the real mechanism, so defer
    # to the same rules used for a right-label-wrong-evidence row rather than
    # calling it a misreading.
    if _is_true(row.get("pred_raw")) and _is_false(row.get("credited")):
        mech, reason = _classify_match(row, config)
        return mech, f"positive not credited under require_span; {reason}"
    key = (row.get("patient_id"), str(row.get("label")), row.get("system"))
    pred_windows = pred_pos_windows.get(key, set())
    this_window = row.get("window_index")
    if pred_windows and not _window_in(this_window, pred_windows):
        return (
            "window_placement",
            f"system is positive for this patient and label in window(s) "
            f"{sorted(str(w) for w in pred_windows)}, but not in {this_window}",
        )
    if row.get("window_miss_reason"):
        return (
            "window_placement",
            f"note could not be placed in a window: {row.get('window_miss_reason')}",
        )

    gold_text = row.get("gold_span_text")
    if has_temporal_cue(gold_text):
        return (
            "temporal_error",
            f"annotated span carries a temporal cue the system may have excluded: "
            f"{_first_cue(_TEMP_RE, gold_text)!r}",
        )

    if gold_text is not None or not pd.isna(row.get("gold_span_start")):
        return (
            "misreading",
            "annotator quoted supporting evidence in this note and the system still said no",
        )

    _ = config
    return "missing_evidence", "gold is positive but carries no span to check against"


def _classify_match(row: dict[str, Any], config: EvalConfig) -> tuple[str, str]:
    """Mechanism for a right label with wrong evidence or wrong code."""
    policy = config.matching.span
    if row.get("code_how") == "mismatch":
        return (
            "ontology_mismatch",
            f"label agrees but codes differ: predicted {row.get('pred_code')}, "
            f"gold {row.get('gold_code')}",
        )
    jac = float(row.get("span_jaccard") or 0.0)
    pred_missing = pd.isna(row.get("pred_span_start"))
    gold_missing = pd.isna(row.get("gold_span_start"))
    if pred_missing or gold_missing:
        side = "prediction" if pred_missing else "annotation"
        return "missing_evidence", f"label agrees but the {side} carries no span"
    if 0.0 < jac < policy.min_overlap:
        return (
            "annotation_threshold",
            f"spans overlap at jaccard {jac:.3f}, below the {policy.min_overlap:.2f} threshold",
        )
    if jac == 0.0:
        return (
            "misreading",
            "label agrees but the quoted spans do not overlap at all, so the system "
            "reached the right answer from different text",
        )
    return "other", "label agrees and spans match; no mechanism rule matched"


def _is_true(x: Any) -> bool:
    """Truthiness that survives numpy and pandas scalars.

    ``df.to_dict("records")`` yields ``numpy.bool_``, not ``bool``, so an
    ``is True`` identity check against a frame cell silently never matches.
    """
    return x is not None and not pd.isna(x) and bool(x)


def _is_false(x: Any) -> bool:
    """Whether ``x`` is present and false (as opposed to missing)."""
    return x is not None and not pd.isna(x) and not bool(x)


def _window_in(this: Any, others: set[Any]) -> bool:
    """Whether ``this`` window is among ``others``, treating NA as its own value."""
    if pd.isna(this):
        return any(pd.isna(o) for o in others)
    return any((not pd.isna(o)) and o == this for o in others)


def _first_cue(pattern: re.Pattern[str], text: str | None) -> str:
    m = pattern.search(text or "")
    return m.group(0).strip() if m else ""


def tag_errors(
    aligned: AlignedFrame,
    config: EvalConfig | None = None,
    *,
    windowed: WindowedFrame | None = None,
    ontology: OntologyMap | None = None,
) -> ErrorTable:
    """Tag every disagreement with a mechanism from :data:`MECHANISMS`.

    Args:
        aligned: Note-level aligned rows.
        config: Matching policy, which sets the thresholds the rules reference.
        windowed: Unit-level rows. Supplying them is what makes
            ``window_placement`` detectable, since that mechanism is about a
            note landing in the wrong window; without it the tag is never
            assigned and the reason column says so.
        ontology: Concept map for the ``ontology_mismatch`` rule.

    Returns:
        An :class:`ErrorTable`.
    """
    config = config or EvalConfig()
    onto = ontology or bundled_map()

    gold_pos_windows = _gold_positive_windows(windowed)
    pred_pos_windows = _pred_positive_windows(windowed)

    df = aligned.df
    scorable = df[df["pred_present"] & df["gold_present"]].copy()

    # Window context for every note, keyed by (system, note_id, label).
    # Built from the `note_ids` membership recorded during aggregation so that
    # no note in a multi-note unit is left without a window.
    note_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    if windowed is not None and len(windowed.df):
        for row in windowed.df.to_dict("records"):
            members = row.get("note_ids") or ()
            if isinstance(members, str):
                members = (members,)
            candidates = list(members) or [
                n for n in (row.get("representative_note_id"), row.get("gold_note_id")) if n
            ]
            for nid in candidates:
                note_map.setdefault((str(row["system"]), str(nid), str(row["label"])), row)

    gold_by_note = {
        (str(r["note_id"]), str(r["label"])): bool(r["gold"])
        for r in df[df["gold_present"]].to_dict("records")
        if not pd.isna(r["gold"])
    }

    rows: list[dict[str, Any]] = []
    for rec in scorable.to_dict("records"):
        pred_v = bool(rec["pred"])
        gold_v = bool(rec["gold"])

        ctx = note_map.get((str(rec["system"]), str(rec["note_id"]), str(rec["label"])), {})
        rec_ctx = dict(rec)
        rec_ctx["window_index"] = ctx.get("window_index", pd.NA)
        rec_ctx["window_label"] = ctx.get("window_label")
        rec_ctx["window_miss_reason"] = ctx.get("window_miss_reason", "")
        rec_ctx["unit_key"] = ctx.get("unit_key") or rec["note_id"]

        if pred_v and not gold_v:
            etype = "FP"
            mech, reason = _classify_fp(rec_ctx, config, gold_pos_windows, gold_by_note, onto)
        elif gold_v and not pred_v:
            etype = "FN"
            mech, reason = _classify_fn(rec_ctx, config, pred_pos_windows, onto)
        elif pred_v and gold_v and not (rec["span_matched"] and rec["code_matched"]):
            etype = "MATCH"
            mech, reason = _classify_match(rec_ctx, config)
        else:
            continue

        rows.append(
            {
                "error_id": "",  # assigned after sorting, so ids are stable
                "system": rec["system"],
                "label": rec["label"],
                "unit_key": rec_ctx["unit_key"],
                "error_type": etype,
                "patient_id": rec["patient_id"],
                "window": rec_ctx.get("window_label") or "",
                "mechanism": mech,
                "span_text": rec["pred_span_text"],
                "note_id": rec["note_id"],
                "mechanism_reason": reason,
                "gold_span_text": rec["gold_span_text"],
                "span_jaccard": rec["span_jaccard"],
                "pred_code": rec["pred_code"],
                "gold_code": rec["gold_code"],
                "code_how": rec["code_how"],
                "confidence": rec["confidence"],
            }
        )

    out = pd.DataFrame(rows, columns=list(ERROR_COLUMNS))
    if len(out):
        out = out.sort_values(["system", "label", "note_id", "error_type"]).reset_index(drop=True)
        out["error_id"] = [f"E{i:06d}" for i in range(1, len(out) + 1)]
    counts = out["mechanism"].value_counts().to_dict() if len(out) else {}
    return ErrorTable(df=out, counts={str(k): int(v) for k, v in counts.items()})


def confusion_export(windowed: WindowedFrame) -> pd.DataFrame:
    """Per-unit confusion rows: what each system said, what gold said, the cell.

    This is the file a reviewer asks for when they want to check the metrics by
    hand. One row per ``(system, unit_key, label)`` with the 2x2 cell named.
    """
    if not len(windowed.df):
        return pd.DataFrame(
            columns=["system", "unit_key", "label", "patient_id", "window", "pred", "gold", "cell"]
        )
    df = windowed.df
    out = pd.DataFrame(
        {
            "system": df["system"],
            "unit_key": df["unit_key"],
            "label": df["label"],
            "patient_id": df["patient_id"],
            "window": df.get("window_label"),
            "pred": df["pred"],
            "gold": df["gold"],
            "n_notes": df["n_notes"],
        }
    )

    def _cell(r: Any) -> str:
        if pd.isna(r["pred"]) or pd.isna(r["gold"]):
            return "unscored"
        if r["pred"] and r["gold"]:
            return "TP"
        if r["pred"] and not r["gold"]:
            return "FP"
        if not r["pred"] and r["gold"]:
            return "FN"
        return "TN"

    out["cell"] = out.apply(_cell, axis=1)
    return out
