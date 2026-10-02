"""Annotation records: what a clinician said about one note-label pair.

Identical to :class:`~clinval.schemas.prediction.Prediction` except that
``system`` becomes ``annotator``, ``confidence`` and ``run`` are absent
(a human annotator has no token count), and ``adjudicated`` is added.

Two annotators' records for the same (note, label) may coexist. That is the
input to inter-rater agreement, and the input to ``clinval adjudicate``, which
collapses them to a single gold record. Evaluation against a two-annotator
file without adjudicating first is an error, not an average -- silently
averaging raters would hide the disagreement the study exists to measure.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clinval.schemas.prediction import Code, Span


class Annotation(BaseModel):
    """One annotator's call on one (note, label) pair."""

    model_config = ConfigDict(extra="forbid")

    annotator: str = Field(min_length=1, description="Annotator identifier (never a real name).")
    note_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: bool
    patient_id: str | None = None
    span: Span | None = None
    code: Code | None = None
    note_datetime: datetime | None = None
    adjudicated: bool = Field(
        default=False,
        description="True when this record is the resolved gold standard for the pair.",
    )

    @field_validator("annotator", "note_id", "label", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @property
    def key(self) -> tuple[str, str, str]:
        """Uniqueness key: ``(annotator, note_id, label)``."""
        return (self.annotator, self.note_id, self.label)


__all__ = ["Annotation", "Code", "Span"]
