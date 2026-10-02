"""The extractor plugin interface.

clinval never calls an LLM. This protocol is the boundary: anything that can
turn notes into :class:`~clinval.schemas.prediction.Prediction` records can be
evaluated, and the adapter -- not the core -- is where the model-specific mess
lives.

Implementing a plugin is three methods, and only :meth:`ExtractorPlugin.predict`
is required.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from clinval.exceptions import PluginError
from clinval.schemas.prediction import Prediction


@dataclass(frozen=True, slots=True)
class Note:
    """One note handed to an extractor.

    ``text`` is the only required field, but ``patient_id`` and
    ``note_datetime`` are needed downstream for patient-window evaluation, so
    an adapter should pass them through whenever the source has them.
    """

    note_id: str
    text: str
    patient_id: str | None = None
    note_datetime: Any = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class ExtractorPlugin(Protocol):
    """Anything that turns notes into clinval predictions."""

    name: str

    def predict(self, notes: Iterable[Note]) -> Iterator[Prediction]:
        """Yield predictions for the given notes.

        Implementations should emit **explicit negatives** for every
        (note, label) pair they considered and rejected. clinval does not read
        silence as a negative, so a positives-only plugin will show a coverage
        warning in the report.
        """
        ...


def validate_predictions(
    records: Iterable[Prediction],
    *,
    plugin_name: str = "<plugin>",
    expect_labels: Sequence[str] | None = None,
) -> list[Prediction]:
    """Check a plugin's output before it enters the pipeline.

    Catches the failure modes an adapter actually hits: emitting something that
    is not a Prediction, repeating a (note, label) pair, or returning labels
    outside the target set.

    Args:
        records: The plugin's output.
        plugin_name: Used in error messages.
        expect_labels: When given, every emitted label must be in this set.

    Returns:
        The records as a list.

    Raises:
        PluginError: On any of the above.
    """
    out: list[Prediction] = []
    seen: set[tuple[str, str, str]] = set()
    allowed = set(expect_labels) if expect_labels else None

    for i, rec in enumerate(records):
        if not isinstance(rec, Prediction):
            raise PluginError(
                f"{plugin_name} yielded {type(rec).__name__} at position {i}; "
                "expected a clinval Prediction"
            )
        if rec.key in seen:
            raise PluginError(
                f"{plugin_name} emitted {rec.key!r} more than once. One record per "
                "(system, note, label) is required."
            )
        if allowed is not None and rec.label not in allowed:
            raise PluginError(
                f"{plugin_name} emitted label {rec.label!r}, which is not in the target "
                f"set {sorted(allowed)}"
            )
        seen.add(rec.key)
        out.append(rec)
    return out
