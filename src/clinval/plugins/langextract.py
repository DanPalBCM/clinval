"""Reference adapter: LangExtract output to clinval predictions.

**Verified against `langextract` 1.7.0** (see ``docs/decisions.md`` D-002/D-003
for the method and the findings). Four properties of that library shape this
adapter, and none of them is hidden from the caller:

1. **No token accounting exists.** `langextract` surfaces no usage metadata --
   its ``ScoredOutput`` is ``(score, output)`` and nothing in the package
   records prompt or completion tokens. So ``tokens_in``/``tokens_out`` are
   left unset and :mod:`clinval.cost` reports cost as *unknown*, not zero.
   ``latency_s`` is measured here, by wall clock, because that clinval can
   observe.
2. **Extractions are spans, not booleans.** A ``data.Extraction`` has an
   ``extraction_class`` and quoted text; there is no truth value. The adapter
   applies an explicit **presence rule** -- an extraction of a label's class
   means ``value=True`` -- and materializes explicit ``False`` records for
   every target label a note did not produce.
3. **Negation is not handled by the library.** "denies diarrhea" extracts as
   class ``diarrhea`` unless the prompt asks for polarity in an attribute.
   The adapter reads a configurable attribute (``negated`` by default) and
   never guesses from the text; guessing silently would turn an extraction bug
   into a clinval scoring bug.
4. **Offsets may be approximate.** ``CharInterval`` is ``start_pos``/``end_pos``,
   both optional, and ``alignment_status`` can be ``MATCH_FUZZY``,
   ``MATCH_GREATER``, or ``MATCH_LESSER``. The adapter records the status and
   warns when exact span matching is combined with non-exact alignment,
   because comparing exact offsets against fuzzy ones measures the aligner,
   not the model.

`langextract` is an optional dependency (``pip install 'clinval[langextract]'``)
and is imported lazily. Nothing in clinval's test suite calls it over the
network.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from clinval.exceptions import PluginError
from clinval.plugins.base import Note
from clinval.schemas.prediction import Prediction, RunMeta, Span

#: Attribute keys read as a negation flag, in order of preference.
DEFAULT_NEGATION_KEYS: tuple[str, ...] = ("negated", "negation", "polarity", "present")

#: Values in a negation attribute that mean "this mention is negated".
_NEGATIVE_TOKENS = {"true", "yes", "y", "1", "negated", "negative", "absent", "denied"}
#: Values that mean "this mention is asserted".
_POSITIVE_TOKENS = {"false", "no", "n", "0", "affirmed", "positive", "present", "asserted"}


def _require_langextract() -> Any:
    """Import ``langextract`` or explain how to install it."""
    try:
        import langextract as lx
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise PluginError(
            "the LangExtract adapter needs the optional dependency. Install it with:\n"
            "    pip install 'clinval[langextract]'\n"
            "or\n"
            "    uv add 'clinval[langextract]'"
        ) from exc
    return lx


@dataclass
class LangExtractConfig:
    """How to map LangExtract output onto clinval predictions.

    Attributes:
        labels: The target label set. Explicit negatives are emitted for every
            label in this set that a note did not produce, which is what makes
            the output scorable -- clinval reads silence as unscored, not as a
            negative.
        class_to_label: Mapping from ``extraction_class`` to clinval label,
            for when the prompt's class names differ from the target labels.
            Unmapped classes are used as-is.
        negation_attribute: Attribute key carrying polarity. ``None`` disables
            negation handling entirely, in which case every extraction counts
            as positive and the limitation is recorded in the run metadata.
        negated_means: Whether a truthy negation attribute means negated
            (``True``, the default, for a ``negated`` flag) or asserted (set
            ``False`` when the attribute is named like ``present``).
        system: The ``system`` value stamped on every prediction.
        prompt_version: Recorded in run metadata for the audit trail.
        require_offsets: When True, an extraction whose ``char_interval`` is
            missing raises instead of yielding a span-less prediction.
    """

    labels: tuple[str, ...]
    class_to_label: dict[str, str] = field(default_factory=dict)
    negation_attribute: str | None = "negated"
    negated_means: bool = True
    system: str = "langextract"
    prompt_version: str = "v1"
    require_offsets: bool = False

    def resolve_label(self, extraction_class: str) -> str:
        """Map an extraction class to a clinval label."""
        return self.class_to_label.get(extraction_class, extraction_class)


def _read_negation(attributes: dict[str, Any] | None, cfg: LangExtractConfig) -> bool | None:
    """Read polarity from an extraction's attributes.

    Returns ``True`` for negated, ``False`` for asserted, and ``None`` when the
    attribute is absent or unrecognized -- which the caller treats as asserted
    while recording that polarity was unavailable.
    """
    if not attributes or cfg.negation_attribute is None:
        return None
    raw = attributes.get(cfg.negation_attribute)
    if raw is None:
        for key in DEFAULT_NEGATION_KEYS:
            if key in attributes:
                raw = attributes[key]
                break
    if raw is None:
        return None
    if isinstance(raw, bool):
        flag = raw
    else:
        token = str(raw).strip().lower()
        if token in _NEGATIVE_TOKENS:
            flag = True
        elif token in _POSITIVE_TOKENS:
            flag = False
        else:
            return None
    return flag if cfg.negated_means else (not flag)


def _span_from_extraction(
    extraction: Any, cfg: LangExtractConfig
) -> tuple[Span | None, str | None]:
    """Build a clinval Span from a LangExtract extraction.

    Returns ``(span, alignment_status)``. ``CharInterval`` uses
    ``start_pos``/``end_pos``, either of which may be ``None``, so a partially
    located extraction yields no span rather than a span with a guessed bound.
    """
    interval = getattr(extraction, "char_interval", None)
    status = getattr(extraction, "alignment_status", None)
    status_name = getattr(status, "name", None) or (str(status) if status is not None else None)
    text = getattr(extraction, "extraction_text", None)

    start = getattr(interval, "start_pos", None) if interval is not None else None
    end = getattr(interval, "end_pos", None) if interval is not None else None
    if start is None or end is None:
        if cfg.require_offsets:
            raise PluginError(
                f"extraction {text!r} has no usable char_interval "
                f"(start_pos={start}, end_pos={end}) and require_offsets is set"
            )
        return None, status_name
    return Span(start=int(start), end=int(end), text=text), status_name


def predictions_from_documents(
    documents: Any,
    notes: Sequence[Note],
    cfg: LangExtractConfig,
    *,
    latencies: dict[str, float] | None = None,
    model_id: str | None = None,
) -> list[Prediction]:
    """Convert LangExtract ``AnnotatedDocument`` output into predictions.

    This is the pure half of the adapter: it takes objects, not a network
    connection, which is how the adapter is tested without an API key.

    Documents are matched to notes positionally, because
    ``AnnotatedDocument.document_id`` is derived from the text when not set
    explicitly and is therefore not a reliable join key. A length mismatch is
    an error rather than a zip-truncation.

    Args:
        documents: A single ``AnnotatedDocument`` or a sequence of them.
        notes: The notes passed to ``lx.extract``, in the same order.
        cfg: Mapping configuration.
        latencies: Per-note wall-clock seconds, keyed by ``note_id``.
        model_id: Model identifier recorded in run metadata.

    Returns:
        One prediction per (note, label) in ``cfg.labels`` -- positives from
        extractions, explicit negatives for the rest.
    """
    docs = documents if isinstance(documents, list | tuple) else [documents]
    if len(docs) != len(notes):
        raise PluginError(
            f"langextract returned {len(docs)} documents for {len(notes)} notes; "
            "clinval matches them positionally, so the counts must agree"
        )

    latencies = latencies or {}
    out: list[Prediction] = []
    saw_fuzzy_alignment = False
    polarity_unavailable = 0

    for note, doc in zip(notes, docs, strict=True):
        extractions = list(getattr(doc, "extractions", None) or [])
        positives: dict[str, dict[str, Any]] = {}

        for ex in extractions:
            label = cfg.resolve_label(str(getattr(ex, "extraction_class", "")))
            if cfg.labels and label not in cfg.labels:
                continue
            span, status = _span_from_extraction(ex, cfg)
            if status and status not in {"MATCH_EXACT", None}:
                saw_fuzzy_alignment = True

            negated = _read_negation(getattr(ex, "attributes", None), cfg)
            if negated is None:
                polarity_unavailable += 1
            if negated:
                # An explicitly negated mention supports a negative, and must
                # not be allowed to overwrite a positive found elsewhere in the
                # same note.
                continue

            prior = positives.get(label)
            if prior is None or (prior["span"] is None and span is not None):
                positives[label] = {"span": span, "status": status}

        target = cfg.labels or tuple(sorted(positives))
        for label in target:
            hit = positives.get(label)
            extra: dict[str, Any] = {}
            if hit is not None and hit.get("status"):
                # RunMeta allows extra keys, so the aligner's verdict rides
                # along with the prediction instead of being dropped.
                extra["alignment_status"] = hit["status"]
            run = RunMeta(
                model=model_id,
                prompt_version=cfg.prompt_version,
                # tokens_in / tokens_out deliberately unset: langextract 1.7.0
                # exposes no usage metadata. clinval.cost reports this as
                # unknown rather than zero. See docs/decisions.md D-003.
                latency_s=latencies.get(note.note_id),
                # One lx.extract call per note answers every label, so the
                # note id identifies the billable call.
                call_id=note.note_id,
                **extra,
            )
            out.append(
                Prediction(
                    system=cfg.system,
                    note_id=note.note_id,
                    label=label,
                    value=hit is not None,
                    patient_id=note.patient_id,
                    span=hit["span"] if hit else None,
                    note_datetime=note.note_datetime,
                    run=run,
                )
            )

    if saw_fuzzy_alignment:
        warnings.warn(
            "langextract reported non-exact span alignment for at least one extraction "
            "(alignment_status is MATCH_FUZZY, MATCH_GREATER, or MATCH_LESSER). Evaluating "
            "these with span matching mode 'exact' measures the aligner rather than the "
            "model; use mode 'overlap' or 'fuzzy'.",
            UserWarning,
            stacklevel=2,
        )
    if polarity_unavailable and cfg.negation_attribute is not None:
        warnings.warn(
            f"{polarity_unavailable} extraction(s) carried no readable "
            f"{cfg.negation_attribute!r} attribute, so they were counted as asserted. "
            "langextract does not detect negation on its own: ask for polarity in the "
            "prompt's attributes, or expect negated mentions to score as false positives.",
            UserWarning,
            stacklevel=2,
        )
    return out


class LangExtractPlugin:
    """Adapter that runs LangExtract and emits clinval predictions.

    Calling :meth:`predict` **does** reach the network and needs an API key --
    it is the one place in clinval that touches a model, and it lives outside
    the core on purpose. Tests cover
    :func:`predictions_from_documents` instead, with real ``langextract``
    objects built in-process.

    Example::

        from clinval.plugins.langextract import LangExtractConfig, LangExtractPlugin

        plugin = LangExtractPlugin(
            LangExtractConfig(labels=("diarrhea", "constipation")),
            prompt_description="Extract symptom mentions with a `negated` attribute.",
            examples=[...],          # langextract ExampleData objects
            model_id="gemini-3.5-flash",
        )
        preds = list(plugin.predict(notes))
    """

    def __init__(
        self,
        config: LangExtractConfig,
        *,
        prompt_description: str | None = None,
        examples: Sequence[Any] | None = None,
        model_id: str = "gemini-3.5-flash",
        api_key: str | None = None,
        extract_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.name = config.system
        self.prompt_description = prompt_description
        self.examples = list(examples) if examples else None
        self.model_id = model_id
        self.api_key = api_key
        self.extract_kwargs = extract_kwargs or {}

    def predict(self, notes: Iterable[Note]) -> Iterator[Prediction]:
        """Run LangExtract over ``notes`` and yield predictions.

        Each note is extracted individually so that ``latency_s`` is per note
        rather than per batch. That is slower than batching, and it is the only
        way to attribute latency to a note given that the library reports none.
        """
        lx = _require_langextract()
        note_list = list(notes)
        if not note_list:
            return

        latencies: dict[str, float] = {}
        documents: list[Any] = []
        for note in note_list:
            started = time.perf_counter()
            doc = lx.extract(
                text_or_documents=note.text,
                prompt_description=self.prompt_description,
                examples=self.examples,
                model_id=self.model_id,
                api_key=self.api_key,
                **self.extract_kwargs,
            )
            latencies[note.note_id] = time.perf_counter() - started
            # lx.extract returns AnnotatedDocument | list[AnnotatedDocument];
            # a single-text call returns the bare object.
            documents.append(doc[0] if isinstance(doc, list | tuple) else doc)

        yield from predictions_from_documents(
            documents,
            note_list,
            self.config,
            latencies=latencies,
            model_id=self.model_id,
        )


def langextract_available() -> bool:
    """Whether the optional ``langextract`` dependency is importable."""
    try:
        import langextract  # noqa: F401
    except ImportError:
        return False
    return True
