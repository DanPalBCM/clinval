"""Alignment of predictions to annotations, and aggregation to patient windows.

Two stages live here, in the order the pipeline runs them:

1. :func:`align_predictions` joins note-level predictions to note-level gold on
   ``(note_id, label)``, applying span and code matching. The result is one row
   per ``(system, note_id, label)``.
2. :func:`aggregate_to_windows` collapses those rows to the evaluation unit --
   usually ``(patient_id, window_index, label)``.

Two decisions here are load-bearing and deliberately not silent:

**Missing predictions are unscored, not negatives.** If a system emitted no
record for a (note, label) that gold covers, clinval cannot distinguish "the
system processed the note and said no" from "the system never saw the note".
Those rows are excluded from the confusion matrix and counted in
``coverage``, which the report prints. Call
:func:`clinval.io.materialize_negatives` to turn silence into explicit
negatives once you know the note universe.

**Multi-annotator gold must be adjudicated.** Averaging two raters would hide
the disagreement a validation study exists to measure, so it is an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from clinval.exceptions import AggregationError
from clinval.match.ontology import OntologyMap, bundled_map, match_codes
from clinval.match.spans import match_spans
from clinval.schemas.annotation import Annotation
from clinval.schemas.config import EvalConfig, TieBreak, Unit, Window
from clinval.schemas.prediction import Code, Prediction, Span

#: Columns present on every aligned frame, in order.
#: Rules :func:`adjudicate` accepts.
ADJUDICATION_RULES = ("agreement", "majority", "first")

ALIGNED_COLUMNS = (
    "system",
    "note_id",
    "patient_id",
    "label",
    "note_datetime",
    "pred",
    "pred_raw",
    "credited",
    "gold",
    "pred_present",
    "gold_present",
    "confidence",
    "span_matched",
    "span_jaccard",
    "span_reason",
    "pred_span_text",
    "gold_span_text",
    "pred_span_start",
    "pred_span_end",
    "gold_span_start",
    "gold_span_end",
    "code_matched",
    "code_how",
    "pred_code",
    "gold_code",
)


@dataclass
class AlignedFrame:
    """Note-level predictions joined to gold, one row per (system, note, label).

    Attributes:
        df: The joined rows. ``pred`` and ``gold`` use pandas' nullable
            ``boolean`` dtype, so "absent" is representable and never silently
            becomes ``False``.
        unit: The evaluation unit this frame will aggregate to.
        annotators: Which annotator each gold record came from.
    """

    df: pd.DataFrame
    unit: Unit = Unit.NOTE
    annotators: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def scorable(self) -> pd.DataFrame:
        """Rows where both a prediction and a gold label exist."""
        return self.df[self.df["pred_present"] & self.df["gold_present"]]

    def coverage(self) -> dict[str, float]:
        """Fraction of gold rows each system actually predicted on.

        A system that emits only positives scores a coverage well below 1.0,
        which is the signal that negatives need materializing.
        """
        out: dict[str, float] = {}
        for system, chunk in self.df.groupby("system", observed=True):
            gold_rows = chunk[chunk["gold_present"]]
            if len(gold_rows) == 0:
                out[str(system)] = float("nan")
            else:
                out[str(system)] = float(gold_rows["pred_present"].mean())
        return out

    def systems(self) -> list[str]:
        """System names present, sorted."""
        return sorted(self.df["system"].dropna().unique().tolist())


@dataclass
class WindowedFrame:
    """Predictions and gold aggregated to the evaluation unit.

    Attributes:
        df: One row per ``(system, unit_key, label)``.
        unit: Which unit the rows represent.
        window: The window definition used, when the unit is time-windowed.
    """

    df: pd.DataFrame
    unit: Unit
    window: Window | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def scorable(self) -> pd.DataFrame:
        """Rows with both a prediction and a gold label."""
        return self.df[self.df["pred_present"] & self.df["gold_present"]]

    def systems(self) -> list[str]:
        """System names present, sorted."""
        return sorted(self.df["system"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# gold resolution
# ---------------------------------------------------------------------------


def resolve_gold(
    annotations: Sequence[Annotation],
) -> tuple[dict[tuple[str, str], Annotation], tuple[str, ...]]:
    """Collapse possibly-multi-annotator records to one gold record per pair.

    Resolution order:

    1. If any record for the pair has ``adjudicated=True``, it wins. Two
       adjudicated records for the same pair is a contradiction and raises.
    2. Otherwise, if exactly one annotator covers the pair, that record wins.
    3. Otherwise two or more annotators disagree-or-agree without adjudication,
       which raises -- run ``clinval adjudicate`` first.

    Returns:
        ``({(note_id, label): Annotation}, annotator_names)``.
    """
    by_pair: dict[tuple[str, str], list[Annotation]] = {}
    for a in annotations:
        by_pair.setdefault((a.note_id, a.label), []).append(a)

    resolved: dict[tuple[str, str], Annotation] = {}
    for pair, recs in by_pair.items():
        adjudicated = [r for r in recs if r.adjudicated]
        if len(adjudicated) > 1:
            raise AggregationError(
                f"pair {pair!r} has {len(adjudicated)} records marked adjudicated; "
                "exactly one adjudicated record per pair is allowed"
            )
        if len(adjudicated) == 1:
            resolved[pair] = adjudicated[0]
            continue
        annotators = {r.annotator for r in recs}
        if len(annotators) == 1:
            resolved[pair] = recs[0]
            continue
        values = {r.value for r in recs}
        raise AggregationError(
            f"pair {pair!r} has records from {len(annotators)} annotators "
            f"({sorted(annotators)}) and none is marked adjudicated"
            + (
                " -- they disagree, so clinval will not guess."
                if len(values) > 1
                else " -- they agree, but gold must still be marked adjudicated=true."
            )
            + " Run `clinval adjudicate` to produce a single gold file."
        )
    return resolved, tuple(sorted({a.annotator for a in annotations}))


def adjudicate(
    annotations: Sequence[Annotation],
    *,
    rule: str = "agreement",
    annotator: str = "adjudicated",
) -> list[Annotation]:
    """Produce single-gold records from multi-annotator annotations.

    Args:
        annotations: Records from one or more annotators.
        rule: ``"agreement"`` keeps pairs every annotator agreed on and drops
            the rest (the conservative default -- a disagreement is not gold).
            ``"majority"`` takes the majority value and breaks ties by dropping.
            ``"first"`` takes the first annotator in sorted order, for the
            single-rater case.
        annotator: Name stamped on the output records.

    Returns:
        Adjudicated annotations, each with ``adjudicated=True``.
    """
    if rule not in ADJUDICATION_RULES:
        # Validated up front rather than inside the loop: a file whose records
        # are all already adjudicated would otherwise never reach the branch
        # that checks, and a typo'd rule would pass silently.
        raise AggregationError(
            f"unknown adjudication rule {rule!r}; use one of {', '.join(ADJUDICATION_RULES)}"
        )

    by_pair: dict[tuple[str, str], list[Annotation]] = {}
    for a in annotations:
        by_pair.setdefault((a.note_id, a.label), []).append(a)

    out: list[Annotation] = []
    for _pair, recs in sorted(by_pair.items()):
        existing = [r for r in recs if r.adjudicated]
        if existing:
            out.append(existing[0])
            continue
        values = [r.value for r in recs]
        if rule == "agreement":
            if len(set(values)) > 1:
                continue
            chosen = recs[0]
        elif rule == "majority":
            n_true = sum(values)
            if n_true * 2 == len(values):
                continue  # tie: no gold
            target = n_true * 2 > len(values)
            # `target` is the majority value, so at least one record holds it.
            match = next((r for r in recs if r.value is target), None)
            if match is None:  # pragma: no cover - unreachable by construction
                continue
            chosen = match
        else:  # rule == "first"; validated above
            chosen = sorted(recs, key=lambda r: r.annotator)[0]
        out.append(chosen.model_copy(update={"annotator": annotator, "adjudicated": True}))
    return out


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def _code_str(code: Code | None) -> str | None:
    return f"{code.system}|{code.code}" if code else None


def _span_bounds(span: Span | None) -> tuple[float, float]:
    return (float(span.start), float(span.end)) if span else (np.nan, np.nan)


def align_predictions(
    predictions: Sequence[Prediction],
    annotations: Sequence[Annotation],
    config: EvalConfig | None = None,
    *,
    ontology: OntologyMap | None = None,
) -> AlignedFrame:
    """Join note-level predictions to note-level gold.

    Produces one row per ``(system, note_id, label)`` over the union of
    predicted and gold pairs, so nothing is dropped: a prediction with no gold
    appears with ``gold_present=False``, and gold with no prediction appears
    with ``pred_present=False``.

    Args:
        predictions: Note-level predictions from one or more systems.
        annotations: Note-level gold, adjudicated if multi-annotator.
        config: Matching and label policy. Defaults are used when omitted.
        ontology: Concept map for code matching. Defaults to the bundled
            demo-scope map.

    Returns:
        An :class:`AlignedFrame`.
    """
    config = config or EvalConfig()
    onto = ontology or bundled_map()
    gold_by_pair, annotators = resolve_gold(annotations)

    label_filter = set(config.labels) if config.labels else None
    systems = sorted({p.system for p in predictions})
    if not systems:
        # Still a legal call: gold with no predictions produces all-unscored rows.
        systems = ["<none>"]

    preds_by_system: dict[str, dict[tuple[str, str], Prediction]] = {s: {} for s in systems}
    for pred in predictions:
        preds_by_system[pred.system][(pred.note_id, pred.label)] = pred

    rows: list[dict[str, Any]] = []
    for system in systems:
        pred_map = preds_by_system[system]
        pairs = sorted(set(pred_map) | set(gold_by_pair))
        for pair in pairs:
            note_id, label = pair
            if label_filter is not None and label not in label_filter:
                continue
            p = pred_map.get(pair)
            g = gold_by_pair.get(pair)

            span_res = match_spans(
                p.span if p else None, g.span if g else None, config.matching.span
            )
            code_policy = None
            if p and p.code:
                code_policy = config.matching.ontology.get(p.code.system.upper())
            elif g and g.code:
                code_policy = config.matching.ontology.get(g.code.system.upper())
            code_res = match_codes(p.code if p else None, g.code if g else None, code_policy, onto)

            p_s, p_e = _span_bounds(p.span if p else None)
            g_s, g_e = _span_bounds(g.span if g else None)

            # A positive claim is "credited" only if the evidence policy is
            # satisfied. With require_span off (the default) every positive is
            # credited and `pred` equals `pred_raw`, so the boolean metrics are
            # untouched. With it on, an unsupported positive is not credited
            # and scores as a negative -- which is what gives `min_overlap` a
            # defined effect on sensitivity and kappa.
            raw = p.value if p else None
            credited = True
            if p is not None and p.value and config.matching.span.require_span:
                credited = bool(span_res.matched and code_res.matched)
            effective = (bool(raw) and credited) if p is not None else pd.NA

            rows.append(
                {
                    "system": system,
                    "note_id": note_id,
                    "patient_id": (
                        p.patient_id if p and p.patient_id else (g.patient_id if g else None)
                    ),
                    "label": label,
                    "note_datetime": (
                        p.note_datetime
                        if p and p.note_datetime
                        else (g.note_datetime if g else None)
                    ),
                    "pred": effective,
                    "pred_raw": raw if p else pd.NA,
                    "credited": credited,
                    "gold": g.value if g else pd.NA,
                    "pred_present": p is not None,
                    "gold_present": g is not None,
                    "confidence": p.confidence if p else None,
                    "span_matched": span_res.matched,
                    "span_jaccard": span_res.jaccard,
                    "span_reason": span_res.reason,
                    "pred_span_text": p.span.text if p and p.span else None,
                    "gold_span_text": g.span.text if g and g.span else None,
                    "pred_span_start": p_s,
                    "pred_span_end": p_e,
                    "gold_span_start": g_s,
                    "gold_span_end": g_e,
                    "code_matched": code_res.matched,
                    "code_how": code_res.how,
                    "pred_code": _code_str(p.code if p else None),
                    "gold_code": _code_str(g.code if g else None),
                }
            )

    df = pd.DataFrame(rows, columns=list(ALIGNED_COLUMNS))
    df["pred"] = df["pred"].astype("boolean")
    df["pred_raw"] = df["pred_raw"].astype("boolean")
    df["gold"] = df["gold"].astype("boolean")
    df["note_datetime"] = pd.to_datetime(df["note_datetime"], errors="coerce")
    if len(df) and df["note_datetime"].dt.tz is not None:
        df["note_datetime"] = df["note_datetime"].dt.tz_localize(None)
    return AlignedFrame(df=df, unit=config.unit, annotators=annotators)


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


def build_windows(index: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Lay down the window grid for every patient.

    Intervals are half-open ``[start, end)``, so a note timestamped exactly on
    a boundary falls in the later window and is counted once.

    Args:
        index: Frame with ``patient_id`` and ``index_date``.
        window: Window definition.

    Returns:
        Frame with ``patient_id``, ``window_index``, ``window_start``,
        ``window_end``, ``window_label``.
    """
    required = {"patient_id", "index_date"}
    missing = required - set(index.columns)
    if missing:
        raise AggregationError(f"index frame is missing column(s): {sorted(missing)}")

    offset = window.size.to_offset()
    first = -1 if window.include_prior else 0
    rows: list[dict[str, Any]] = []
    for row in index.itertuples(index=False):
        anchor = pd.Timestamp(row.index_date)
        for i in range(first, window.n_windows):
            start = anchor + offset * i if i >= 0 else anchor - offset
            end = start + offset
            rows.append(
                {
                    "patient_id": str(row.patient_id),
                    "window_index": i,
                    "window_start": start,
                    "window_end": end,
                    "window_label": f"w{i}[{start.date()},{end.date()})",
                }
            )
    return pd.DataFrame(rows)


def assign_windows(
    aligned: AlignedFrame,
    index: pd.DataFrame,
    window: Window,
) -> pd.DataFrame:
    """Attach a window to each aligned note-level row.

    Rows that fall outside every window, that have no timestamp, or whose
    patient is absent from the index file get ``window_index = NA`` and a
    non-empty ``window_miss_reason``. They are kept, not dropped: the error
    taxonomy needs them to recognize a ``window_placement`` mistake.
    """
    grid = build_windows(index, window)
    df = aligned.df.copy()
    df["patient_id"] = df["patient_id"].astype("object")

    known_patients = set(grid["patient_id"])
    out_chunks: list[pd.DataFrame] = []

    for pid, chunk in df.groupby("patient_id", dropna=False, observed=True):
        chunk = chunk.copy()
        if pid is None or (isinstance(pid, float) and np.isnan(pid)) or pd.isna(pid):
            chunk["window_index"] = pd.NA
            chunk["window_start"] = pd.NaT
            chunk["window_end"] = pd.NaT
            chunk["window_label"] = None
            chunk["window_miss_reason"] = "no_patient_id"
            out_chunks.append(chunk)
            continue
        pid_s = str(pid)
        if pid_s not in known_patients:
            chunk["window_index"] = pd.NA
            chunk["window_start"] = pd.NaT
            chunk["window_end"] = pd.NaT
            chunk["window_label"] = None
            chunk["window_miss_reason"] = "patient_not_in_index"
            out_chunks.append(chunk)
            continue

        pgrid = grid[grid["patient_id"] == pid_s]
        idx_vals: list[Any] = []
        starts: list[Any] = []
        ends: list[Any] = []
        labels: list[Any] = []
        reasons: list[str] = []
        for dt in chunk["note_datetime"]:
            if pd.isna(dt):
                idx_vals.append(pd.NA)
                starts.append(pd.NaT)
                ends.append(pd.NaT)
                labels.append(None)
                reasons.append("no_note_datetime")
                continue
            hit = pgrid[(pgrid["window_start"] <= dt) & (dt < pgrid["window_end"])]
            if len(hit) == 0:
                idx_vals.append(pd.NA)
                starts.append(pd.NaT)
                ends.append(pd.NaT)
                labels.append(None)
                reasons.append("outside_all_windows")
            else:
                r = hit.iloc[0]
                idx_vals.append(int(r["window_index"]))
                starts.append(r["window_start"])
                ends.append(r["window_end"])
                labels.append(r["window_label"])
                reasons.append("")
        chunk["window_index"] = pd.array(idx_vals, dtype="Int64")
        chunk["window_start"] = starts
        chunk["window_end"] = ends
        chunk["window_label"] = labels
        chunk["window_miss_reason"] = reasons
        out_chunks.append(chunk)

    if not out_chunks:
        df["window_index"] = pd.array([], dtype="Int64")
        df["window_start"] = pd.NaT
        df["window_end"] = pd.NaT
        df["window_label"] = None
        df["window_miss_reason"] = ""
        return df
    return pd.concat(out_chunks).sort_index()


def _representative(chunk: pd.DataFrame, tie_break: TieBreak, column: str) -> Any:
    """Pick the note that represents a positive aggregate."""
    positives = chunk[chunk[column].fillna(False).astype(bool)]
    pool = positives if len(positives) else chunk
    if tie_break is TieBreak.LATEST:
        pool = pool.sort_values("note_datetime", ascending=False, na_position="last")
    elif tie_break is TieBreak.HIGHEST_CONFIDENCE:
        pool = pool.sort_values("confidence", ascending=False, na_position="last")
    else:
        pool = pool.sort_values("note_datetime", ascending=True, na_position="last")
    return pool.iloc[0]["note_id"] if len(pool) else None


def aggregate_to_windows(
    aligned: AlignedFrame,
    window: Window | None = None,
    *,
    index: pd.DataFrame | None = None,
    unit: Unit | None = None,
) -> WindowedFrame:
    """Collapse note-level rows to the evaluation unit.

    The aggregation rule is **any-positive**: a patient-window is positive for a
    label if any note in it is positive. That matches how a chart reviewer
    answers "did this patient have diarrhea in the first six months" -- one
    documented episode is enough.

    Unit behaviour:

    * ``note`` -- a pass-through that adds a ``unit_key`` of ``note_id``.
      Idempotent, so calling it on an already-aggregated frame is a no-op.
    * ``patient`` -- collapses all notes per patient, ignoring time.
    * ``patient_window`` -- requires ``index``; collapses within each window.

    Args:
        aligned: The note-level aligned frame.
        window: Window definition. Required for ``patient_window``.
        index: Per-patient index dates. Required for ``patient_window``.
        unit: Override the unit on ``aligned``.

    Returns:
        A :class:`WindowedFrame`.
    """
    unit = unit or aligned.unit
    df = aligned.df

    # Idempotence (SPEC §8.2): an already-aggregated frame is returned
    # unchanged. `unit_key` together with `n_notes` is the signature of
    # aggregated output -- note-level rows carry neither -- so re-aggregating
    # is a genuine no-op rather than a second pass that would collapse the
    # representative note and lose the original note ids.
    if {"unit_key", "n_notes"}.issubset(df.columns):
        return WindowedFrame(df=df.copy(), unit=unit, window=window, meta=dict(aligned.meta))

    if unit is Unit.NOTE:
        out = df.copy()
        out["unit_key"] = out["note_id"]
        out["window_index"] = pd.array([pd.NA] * len(out), dtype="Int64")
        out["window_label"] = None
        out["n_notes"] = 1
        out["representative_note_id"] = out["note_id"]
        return WindowedFrame(df=out, unit=unit, window=None)

    if unit is Unit.PATIENT_WINDOW:
        if index is None:
            raise AggregationError(
                "unit='patient_window' requires per-patient index dates; "
                "pass index=read_index_dates('index.csv')"
            )
        window = window or Window()
        work = assign_windows(aligned, index, window)
        group_cols = ["system", "patient_id", "window_index", "label"]
    else:  # Unit.PATIENT
        work = df.copy()
        work["window_index"] = pd.array([pd.NA] * len(work), dtype="Int64")
        work["window_label"] = None
        work["window_miss_reason"] = (
            work["patient_id"].isna().map({True: "no_patient_id", False: ""})
        )
        group_cols = ["system", "patient_id", "label"]
        window = None

    if work["patient_id"].isna().all() and len(work):
        raise AggregationError(
            f"unit={unit.value!r} needs patient_id on the records, but none carry one. "
            "Either set patient_id on predictions and annotations, or evaluate with unit='note'."
        )

    rows: list[dict[str, Any]] = []
    tie_break = window.tie_break if window else TieBreak.EARLIEST
    for keys, chunk in work.groupby(group_cols, dropna=False, observed=True):
        keymap = dict(zip(group_cols, keys, strict=True))
        pid = keymap.get("patient_id")
        widx = keymap.get("window_index", pd.NA)
        label = keymap["label"]
        system = keymap["system"]

        has_pred = bool(chunk["pred_present"].any())
        has_gold = bool(chunk["gold_present"].any())
        pred_vals = chunk.loc[chunk["pred_present"], "pred"]
        gold_vals = chunk.loc[chunk["gold_present"], "gold"]

        wlabels = chunk["window_label"].dropna().unique()
        reasons = sorted({r for r in chunk["window_miss_reason"] if r})
        unit_key = (
            f"{pid}|{wlabels[0]}"
            if len(wlabels)
            else (f"{pid}|unassigned" if pd.isna(widx) else f"{pid}|w{widx}")
        )
        if unit is Unit.PATIENT:
            unit_key = str(pid)

        rows.append(
            {
                "system": system,
                "unit_key": unit_key,
                "patient_id": pid,
                "label": label,
                "window_index": widx,
                "window_label": wlabels[0] if len(wlabels) else None,
                "window_start": chunk["window_start"].min() if "window_start" in chunk else pd.NaT,
                "window_end": chunk["window_end"].max() if "window_end" in chunk else pd.NaT,
                "window_miss_reason": ";".join(reasons),
                "pred": bool(pred_vals.fillna(False).any()) if has_pred else pd.NA,
                "gold": bool(gold_vals.fillna(False).any()) if has_gold else pd.NA,
                "pred_present": has_pred,
                "gold_present": has_gold,
                "n_notes": int(chunk["note_id"].nunique()),
                "n_pred_positive": int(pred_vals.fillna(False).sum()),
                "n_gold_positive": int(gold_vals.fillna(False).sum()),
                "n_uncredited": (
                    int(
                        (
                            chunk["pred_raw"].fillna(False).astype(bool)
                            & ~chunk["credited"].astype(bool)
                        ).sum()
                    )
                    if "credited" in chunk.columns
                    else 0
                ),
                "confidence": float(chunk["confidence"].max())
                if chunk["confidence"].notna().any()
                else None,
                "span_matched": bool(chunk["span_matched"].any()),
                "span_jaccard": float(chunk["span_jaccard"].max()) if len(chunk) else 0.0,
                "code_matched": bool(chunk["code_matched"].all()),
                "code_how": ";".join(sorted(set(chunk["code_how"].dropna()))),
                "representative_note_id": _representative(chunk, tie_break, "pred"),
                "gold_note_id": _representative(chunk, tie_break, "gold"),
                # Full membership, not just the representative. The error
                # taxonomy needs a window for *every* note in the unit:
                # without it, notes that are neither the representative nor
                # the gold note would carry no window context and could be
                # mistagged as window-placement errors.
                "note_ids": tuple(sorted(chunk["note_id"].astype(str).unique())),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out["pred"] = out["pred"].astype("boolean")
        out["gold"] = out["gold"].astype("boolean")
        out["window_index"] = out["window_index"].astype("Int64")
        out = out.sort_values(["system", "patient_id", "window_index", "label"]).reset_index(
            drop=True
        )
    return WindowedFrame(df=out, unit=unit, window=window)
