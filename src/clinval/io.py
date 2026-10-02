"""Readers and writers for clinval's on-disk formats.

All record files are JSONL: one JSON object per line, no wrapping array. Errors
report the 1-based line number, because the whole point of a validation harness
is that you can find the bad row.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from clinval.exceptions import DuplicateRecordError, RecordError
from clinval.schemas.annotation import Annotation
from clinval.schemas.prediction import Prediction

M = TypeVar("M", bound=BaseModel)


class _Keyed(Protocol):
    """A record that exposes a uniqueness key."""

    @property
    def key(self) -> tuple[str, str, str]: ...


#: Column names accepted for the patient identifier in an index-date file.
_INDEX_PATIENT_ALIASES = ("patient_id", "subject_id", "person_id", "patient")
#: Column names accepted for the index date itself.
_INDEX_DATE_ALIASES = ("index_date", "index_datetime", "anchor_date", "date")


def _iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` for each non-blank line of a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise RecordError("file not found", path=str(path))
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RecordError(
                    f"malformed JSON: {exc.msg}", path=str(path), line=lineno
                ) from exc
            if not isinstance(obj, dict):
                raise RecordError(
                    f"each line must be a JSON object, got {type(obj).__name__}",
                    path=str(path),
                    line=lineno,
                )
            yield lineno, obj


def _read_models(path: str | Path, model: type[M]) -> list[M]:
    """Parse every line of ``path`` into ``model``, reporting the first failure."""
    out: list[M] = []
    for lineno, obj in _iter_jsonl(path):
        try:
            out.append(model.model_validate(obj))
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            )
            raise RecordError(
                f"invalid {model.__name__}: {detail}", path=str(path), line=lineno
            ) from exc
    return out


def _check_duplicates(records: Sequence[_Keyed], kind: str) -> None:
    """Raise if any uniqueness key appears twice."""
    seen: dict[tuple[str, str, str], int] = {}
    for idx, rec in enumerate(records, start=1):
        key = rec.key
        if key in seen:
            raise DuplicateRecordError(
                f"duplicate {kind} for {key!r}: records {seen[key]} and {idx}. "
                "clinval refuses to pick one, because which copy survives would change "
                "the metrics. Deduplicate the input first."
            )
        seen[key] = idx


def read_predictions(path: str | Path, *, allow_duplicates: bool = False) -> list[Prediction]:
    """Read a predictions JSONL file.

    Args:
        path: Path to the JSONL file.
        allow_duplicates: When ``False`` (the default), a repeated
            ``(system, note_id, label)`` raises :class:`DuplicateRecordError`.

    Returns:
        Every prediction in file order.
    """
    records = _read_models(path, Prediction)
    if not allow_duplicates:
        _check_duplicates(records, "prediction")
    return records


def read_annotations(path: str | Path, *, allow_duplicates: bool = False) -> list[Annotation]:
    """Read an annotations JSONL file.

    Multiple annotators per (note, label) are expected and legal; the
    uniqueness key includes the annotator.
    """
    records = _read_models(path, Annotation)
    if not allow_duplicates:
        _check_duplicates(records, "annotation")
    return records


def write_predictions(records: Iterable[Prediction], path: str | Path) -> int:
    """Write predictions as JSONL. Returns the number of records written."""
    return _write_models(records, path)


def write_annotations(records: Iterable[Annotation], path: str | Path) -> int:
    """Write annotations as JSONL. Returns the number of records written."""
    return _write_models(records, path)


def _write_models(records: Iterable[BaseModel], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json(exclude_none=True))
            fh.write("\n")
            n += 1
    return n


def read_index_dates(path: str | Path) -> pd.DataFrame:
    """Read per-patient index dates from CSV.

    The file needs a patient column and a date column; several common names are
    accepted (``patient_id``/``subject_id``/``person_id`` and
    ``index_date``/``anchor_date``/``date``) so that an index file exported from
    an OMOP or MEDS pipeline works without renaming.

    Returns:
        A frame with exactly two columns, ``patient_id`` and ``index_date``,
        the latter parsed to datetime and timezone-naive.
    """
    path = Path(path)
    if not path.exists():
        raise RecordError("index-date file not found", path=str(path))
    df = pd.read_csv(path)
    if df.empty:
        raise RecordError("index-date file has no rows", path=str(path))

    lower = {c.lower(): c for c in df.columns}
    pcol = next((lower[a] for a in _INDEX_PATIENT_ALIASES if a in lower), None)
    dcol = next((lower[a] for a in _INDEX_DATE_ALIASES if a in lower), None)
    if pcol is None:
        raise RecordError(
            f"no patient column found; looked for {list(_INDEX_PATIENT_ALIASES)}, "
            f"file has {list(df.columns)}",
            path=str(path),
        )
    if dcol is None:
        raise RecordError(
            f"no index-date column found; looked for {list(_INDEX_DATE_ALIASES)}, "
            f"file has {list(df.columns)}",
            path=str(path),
        )

    out = pd.DataFrame(
        {
            "patient_id": df[pcol].astype(str),
            "index_date": pd.to_datetime(df[dcol], errors="coerce"),
        }
    )
    bad = out["index_date"].isna()
    if bad.any():
        first = int(bad.to_numpy().nonzero()[0][0]) + 2  # +2: header row and 1-based
        raise RecordError(
            f"unparseable index date {df[dcol].iloc[first - 2]!r}", path=str(path), line=first
        )
    if out["index_date"].dt.tz is not None:
        out["index_date"] = out["index_date"].dt.tz_localize(None)

    dupes = out["patient_id"].duplicated()
    if dupes.any():
        raise DuplicateRecordError(
            f"patient {out.loc[dupes, 'patient_id'].iloc[0]!r} has more than one index date; "
            "one anchor per patient is required for window assignment"
        )
    return out


def materialize_negatives(
    predictions: Sequence[Prediction],
    *,
    note_ids: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    notes_path: str | Path | None = None,
) -> list[Prediction]:
    """Add explicit ``value=false`` predictions for unmentioned (note, label) pairs.

    clinval never infers a negative from silence (see
    :mod:`clinval.schemas.prediction`). This is the opt-in that turns silence
    into negatives, for extractors that only emit positives. The caller must
    supply the full note universe, because only the caller knows which notes the
    system actually processed.

    Args:
        predictions: The positive-only predictions.
        note_ids: Every note the systems processed. Required unless ``notes_path``
            is given.
        labels: Every label in the target set. Defaults to the labels observed
            in ``predictions``.
        notes_path: A JSONL notes file with ``note_id`` (and optionally
            ``patient_id`` / ``note_datetime``) per line, used as the universe.

    Returns:
        The input predictions plus one ``value=false`` record per missing
        (system, note, label) triple. Input records are never modified.
    """
    meta: dict[str, dict[str, Any]] = {}
    universe: list[str] = []

    if notes_path is not None:
        for _, obj in _iter_jsonl(notes_path):
            nid = obj.get("note_id")
            if nid is None:
                raise RecordError("notes file row has no note_id", path=str(notes_path))
            nid = str(nid)
            universe.append(nid)
            meta[nid] = {
                "patient_id": obj.get("patient_id"),
                "note_datetime": obj.get("note_datetime"),
            }
    if note_ids is not None:
        universe.extend(str(n) for n in note_ids)
    if not universe:
        raise RecordError("materialize_negatives needs note_ids or notes_path")

    # Fill metadata gaps from the predictions themselves.
    for p in predictions:
        meta.setdefault(p.note_id, {})
        if meta[p.note_id].get("patient_id") is None and p.patient_id is not None:
            meta[p.note_id]["patient_id"] = p.patient_id
        if meta[p.note_id].get("note_datetime") is None and p.note_datetime is not None:
            meta[p.note_id]["note_datetime"] = p.note_datetime

    target_labels = (
        sorted({str(x) for x in labels}) if labels else sorted({p.label for p in predictions})
    )
    systems = sorted({p.system for p in predictions})
    if not systems:
        raise RecordError("materialize_negatives needs at least one prediction to name a system")

    have = {p.key for p in predictions}
    out = list(predictions)
    for system in systems:
        for note_id in dict.fromkeys(universe):
            for label in target_labels:
                if (system, note_id, label) in have:
                    continue
                info = meta.get(note_id, {})
                dt = info.get("note_datetime")
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                out.append(
                    Prediction(
                        system=system,
                        note_id=note_id,
                        label=label,
                        value=False,
                        patient_id=info.get("patient_id"),
                        note_datetime=dt,
                    )
                )
    return out


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    """Write any JSON-serializable object, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, sort_keys=True, default=str))


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    path = Path(path)
    if not path.exists():
        raise RecordError("file not found", path=str(path))
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RecordError(f"malformed JSON: {exc.msg}", path=str(path), line=exc.lineno) from exc
