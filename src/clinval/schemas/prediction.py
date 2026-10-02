"""Prediction records: what a system said about one note-label pair.

The record format is normative; see SPEC §6.1. A prediction is the unit a
system emits, and it is always *note-level*. Aggregation to patient windows
happens later, in :mod:`clinval.aggregate`.

The absence of a positive prediction is **not** read as a negative, because
clinval cannot tell "the system processed this note and said no" apart from
"the system never saw this note". Systems emit explicit ``value=false``
records, or materialize them with ``clinval negatives``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Span(BaseModel):
    """A character offset range into the note text.

    Offsets are Python slice semantics: ``text[start:end]``. ``start == end``
    is a zero-length span, which is allowed (some extractors mark an insertion
    point) but never matches another span under ``exact`` or ``overlap`` modes.
    """

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0, description="Inclusive start offset, 0-based.")
    end: int = Field(ge=0, description="Exclusive end offset.")
    text: str | None = Field(default=None, description="Verbatim quoted text, if available.")

    @model_validator(mode="after")
    def _check_order(self) -> Span:
        if self.end < self.start:
            raise ValueError(f"span end ({self.end}) precedes start ({self.start})")
        return self

    @property
    def length(self) -> int:
        """Number of characters covered."""
        return self.end - self.start


class Code(BaseModel):
    """An ontology code attached to a prediction.

    ``system`` is the terminology name (``ICD-10``, ``HPO``, ``SNOMED-CT``).
    clinval compares codes with the rules in :mod:`clinval.match.ontology`;
    it does not validate that a code exists in its terminology.
    """

    model_config = ConfigDict(extra="forbid")

    system: str = Field(description="Terminology name, e.g. 'ICD-10' or 'HPO'.")
    code: str = Field(description="The code itself, e.g. 'E11.9' or 'HP:0002014'.")
    display: str | None = Field(default=None, description="Human-readable label.")


class RunMeta(BaseModel):
    """Per-call metadata used for cost and latency accounting.

    Every field is optional because extractors differ in what they expose. In
    particular ``langextract`` reports no token counts at all, so the bundled
    adapter leaves ``tokens_in``/``tokens_out`` unset and
    :mod:`clinval.cost` reports cost as *unknown* rather than zero. See
    ``docs/decisions.md`` D-003.

    ``call_id`` exists because one model call usually produces *several*
    prediction records -- a single pass over a note typically answers every
    label at once. Repeating that call's token count on each record and summing
    would multiply the true cost by the number of labels. Set ``call_id`` to
    something stable per model call (the note id is the usual choice) and
    :func:`clinval.cost.summarize_cost` counts each call once.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt_version: str | None = None
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    latency_s: float | None = Field(default=None, ge=0)
    call_id: str | None = Field(
        default=None,
        description="Stable identifier for the model call these tokens belong to. "
        "Records sharing a call_id are counted once for cost and latency.",
    )


class Prediction(BaseModel):
    """One system's call on one (note, label) pair.

    ``label`` is an opaque string. It may carry structure that clinval does not
    interpret -- ``fhirground`` emits ``Condition:<code or text>``, for
    instance -- so colons and spaces are permitted and the label is never split.
    """

    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1, description="Name of the predicting system.")
    note_id: str = Field(min_length=1)
    label: str = Field(min_length=1, description="Target name; opaque to clinval.")
    value: bool = Field(description="Boolean assertion. Categorical values are out of scope in v1.")
    patient_id: str | None = Field(
        default=None,
        description="Required when the evaluation unit is patient_window.",
    )
    span: Span | None = None
    code: Code | None = None
    note_datetime: datetime | None = Field(
        default=None,
        description="Required for window assignment when the unit is patient_window.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    run: RunMeta | None = None

    @field_validator("system", "note_id", "label", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @property
    def key(self) -> tuple[str, str, str]:
        """Uniqueness key: ``(system, note_id, label)``."""
        return (self.system, self.note_id, self.label)
