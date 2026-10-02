"""Deterministic synthetic clinical notes, gold annotations, and mock predictions.

**This is the only data clinval ships or tests against.** Nothing here derives
from a real record: the notes are assembled from a fixed sentence bank, the
dates start from a constant, and the identifiers are sequential. There is no
PHI to leak because there was never any PHI.

The generator exists to make the harness testable *and* to make its failure
modes demonstrable. The two mock systems are built with specific, known
pathologies so that the error taxonomy has something real to find:

* ``careful_reader`` -- mostly right, with occasional random flips and spans
  that are a few characters off. Produces ``annotation_threshold`` and
  ``window_placement`` errors.
* ``negation_blind`` -- fires on any mention of a concept, including negated
  and historical ones, and sometimes attaches a neighbouring concept's code.
  Produces ``misreading``, ``temporal_error``, and ``ontology_mismatch``
  errors.

A validation harness whose own tests only use a well-behaved system has not
been tested.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from clinval.io import write_annotations, write_predictions
from clinval.match.ontology import BUNDLED_MAP
from clinval.schemas.annotation import Annotation
from clinval.schemas.prediction import Code, Prediction, RunMeta, Span

#: Fixed epoch for every generated timestamp, so output is reproducible.
EPOCH = datetime(2021, 1, 4, 9, 0, 0)

#: Labels the generator knows how to write sentences about.
SUPPORTED_LABELS: tuple[str, ...] = (
    "diarrhea",
    "constipation",
    "vomiting",
    "nausea",
    "abdominal_pain",
)

#: Label to the HPO code the generator attaches, derived from the bundled map.
LABEL_CODES: dict[str, str] = {
    "diarrhea": "HP:0002014",
    "constipation": "HP:0002019",
    "vomiting": "HP:0002572",
    "nausea": "HP:0002018",
    "abdominal_pain": "HP:0002013",
}

#: Affirmative phrasings per label. The mention text is what gets quoted as a span.
_AFFIRM: dict[str, tuple[str, ...]] = {
    "diarrhea": (
        "reports loose stools three to four times daily",
        "has had watery stools since the weekend",
        "continues to have diarrhea despite dietary changes",
    ),
    "constipation": (
        "describes hard stools every third day",
        "has infrequent bowel movements with straining",
        "constipation persists on current regimen",
    ),
    "vomiting": (
        "vomited twice this morning",
        "reports emesis after meals",
        "has been throwing up since yesterday",
    ),
    "nausea": (
        "feels nauseated most mornings",
        "reports persistent queasiness",
        "nausea is worse after eating",
    ),
    "abdominal_pain": (
        "complains of cramping abdominal pain",
        "reports stomach pain in the left lower quadrant",
        "abdominal pain rated four out of ten",
    ),
}

#: Negated phrasings. Gold is negative; a negation-blind system fires anyway.
_NEGATED: dict[str, tuple[str, ...]] = {
    "diarrhea": (
        "denies diarrhea or loose stools",
        "no watery stools reported this interval",
    ),
    "constipation": (
        "denies constipation and straining",
        "no hard stools since the last visit",
    ),
    "vomiting": ("denies vomiting", "no emesis since discharge"),
    "nausea": ("denies nausea", "no queasiness reported"),
    "abdominal_pain": ("denies abdominal pain", "no stomach pain on exam"),
}

#: Historical or other-subject phrasings. Gold is negative for the current
#: window; a system without temporal reasoning calls them positive.
_HISTORICAL: dict[str, tuple[str, ...]] = {
    "diarrhea": (
        "history of diarrhea two years ago during chemotherapy",
        "family history of chronic diarrhea in the mother",
    ),
    "constipation": (
        "history of constipation as an infant, long since resolved",
        "father has a history of constipation",
    ),
    "vomiting": ("history of vomiting with prior illness months ago",),
    "nausea": ("past medical history includes nausea during pregnancy",),
    "abdominal_pain": ("history of abdominal pain evaluated last year",),
}

#: Label-free filler. Keeps notes from being a list of findings, so span
#: offsets vary and overlap thresholds get exercised.
_FILLER: tuple[str, ...] = (
    "Vital signs are within expected range for age.",
    "Weight and height tracked along the prior percentile.",
    "Immunizations are up to date per the registry.",
    "Caregiver reports good adherence to the current plan.",
    "No new medications were started this interval.",
    "Sleep and appetite are described as unchanged.",
    "School attendance has been regular this term.",
    "Follow-up interval discussed and agreed with the family.",
    "Laboratory results from the prior visit were reviewed.",
    "Anticipatory guidance provided and questions answered.",
)

_OPENERS: tuple[str, ...] = (
    "Routine follow-up visit.",
    "Interval history obtained from the caregiver.",
    "Patient seen for scheduled review.",
    "Telehealth check-in completed.",
)


@dataclass
class SyntheticDataset:
    """Generated notes, gold, mock predictions, and the index-date table.

    Attributes:
        notes: One dict per note: ``note_id``, ``patient_id``,
            ``note_datetime``, ``text``.
        annotations: Adjudicated gold, one record per (note, label).
        predictions: System name to its note-level predictions.
        index: Frame of ``patient_id`` and ``index_date``.
        labels: Labels covered.
        seed: The seed used, recorded so a report can state it.
    """

    notes: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    predictions: dict[str, list[Prediction]] = field(default_factory=dict)
    index: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: tuple[str, ...] = ()
    seed: int = 0
    second_annotator: list[Annotation] = field(default_factory=list)

    @property
    def all_predictions(self) -> list[Prediction]:
        """Every system's predictions in one list."""
        out: list[Prediction] = []
        for name in sorted(self.predictions):
            out.extend(self.predictions[name])
        return out

    @property
    def patients(self) -> list[str]:
        """Patient identifiers, sorted."""
        return sorted({str(n["patient_id"]) for n in self.notes})

    def prevalence(self) -> dict[str, float]:
        """Gold-positive rate per label, for sanity-checking a generated set."""
        out: dict[str, float] = {}
        for label in self.labels:
            vals = [a.value for a in self.annotations if a.label == label]
            out[label] = (sum(vals) / len(vals)) if vals else float("nan")
        return out

    def split_by_patient(
        self, dev_fraction: float = 0.5, *, seed: int | None = None
    ) -> tuple[SyntheticDataset, SyntheticDataset]:
        """Split into dev and test by patient, never by note.

        Splitting by note would put two notes from the same patient on both
        sides of the split, which leaks: the tuned threshold would be chosen on
        a patient it is later tested on. Splitting by patient is the only
        correct choice here, and the frozen-dev protocol is pointless without
        it.

        Args:
            dev_fraction: Share of patients assigned to dev.
            seed: Shuffle seed. Defaults to the dataset's own seed.

        Returns:
            ``(dev, test)``.
        """
        if not 0.0 < dev_fraction < 1.0:
            raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
        pats = self.patients
        rng = random.Random(self.seed if seed is None else seed)
        shuffled = pats[:]
        rng.shuffle(shuffled)
        cut = max(1, min(len(shuffled) - 1, round(len(shuffled) * dev_fraction)))
        dev_set, test_set = set(shuffled[:cut]), set(shuffled[cut:])
        return self._subset(dev_set), self._subset(test_set)

    def _subset(self, patients: set[str]) -> SyntheticDataset:
        notes = [n for n in self.notes if str(n["patient_id"]) in patients]
        note_ids = {n["note_id"] for n in notes}
        return SyntheticDataset(
            notes=notes,
            annotations=[a for a in self.annotations if a.note_id in note_ids],
            predictions={
                k: [p for p in v if p.note_id in note_ids] for k, v in self.predictions.items()
            },
            index=self.index[self.index["patient_id"].isin(patients)].reset_index(drop=True),
            labels=self.labels,
            seed=self.seed,
            second_annotator=[a for a in self.second_annotator if a.note_id in note_ids],
        )

    def write(self, outdir: str | Path, *, prefix: str = "") -> dict[str, Path]:
        """Write every artifact to ``outdir``.

        Returns:
            A mapping of artifact name to the path written.
        """
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        notes_path = out / f"{prefix}notes.jsonl"
        with notes_path.open("w", encoding="utf-8") as fh:
            for n in self.notes:
                row = dict(n)
                row["note_datetime"] = pd.Timestamp(row["note_datetime"]).isoformat()
                fh.write(_json_line(row))
        paths["notes"] = notes_path

        gold_path = out / f"{prefix}gold.jsonl"
        write_annotations(self.annotations, gold_path)
        paths["gold"] = gold_path

        if self.second_annotator:
            two_path = out / f"{prefix}gold_two_annotators.jsonl"
            write_annotations(
                [*self.second_annotator, *(_unadjudicated(a) for a in self.annotations)], two_path
            )
            paths["gold_two_annotators"] = two_path

        pred_path = out / f"{prefix}pred.jsonl"
        write_predictions(self.all_predictions, pred_path)
        paths["pred"] = pred_path

        for name, recs in sorted(self.predictions.items()):
            p = out / f"{prefix}pred_{name}.jsonl"
            write_predictions(recs, p)
            paths[f"pred_{name}"] = p

        index_path = out / f"{prefix}index.csv"
        self.index.to_csv(index_path, index=False)
        paths["index"] = index_path
        return paths


def _json_line(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, default=str) + "\n"


def _unadjudicated(a: Annotation) -> Annotation:
    return a.model_copy(update={"annotator": "rater_a", "adjudicated": False})


@dataclass
class _Mention:
    """A phrase placed in a note, with its offsets and what it means."""

    label: str
    kind: str  # affirmative | negated | historical
    start: int
    end: int
    text: str


def _compose_note(
    rng: random.Random,
    labels: tuple[str, ...],
    truth: dict[str, bool],
) -> tuple[str, list[_Mention]]:
    """Assemble note text and record where each mention landed.

    Offsets are computed from the assembled string rather than guessed, so the
    spans are always exactly correct -- the generator must not be the source of
    span bugs the harness is supposed to find.
    """
    pieces: list[str] = [rng.choice(_OPENERS)]
    planned: list[tuple[str, str]] = []  # (label, kind)

    for label in labels:
        if truth[label]:
            planned.append((label, "affirmative"))
        else:
            roll = rng.random()
            if roll < 0.45:
                planned.append((label, "negated"))
            elif roll < 0.60 and label in _HISTORICAL:
                planned.append((label, "historical"))
            # else: the label simply is not mentioned

    n_filler = rng.randint(2, 4)
    filler = rng.sample(_FILLER, k=min(n_filler, len(_FILLER)))
    slots: list[tuple[str, Any]] = [("mention", m) for m in planned] + [
        ("filler", f) for f in filler
    ]
    rng.shuffle(slots)

    mentions: list[_Mention] = []
    for kind, payload in slots:
        if kind == "filler":
            pieces.append(str(payload))
            continue
        label, mkind = payload
        bank = {"affirmative": _AFFIRM, "negated": _NEGATED, "historical": _HISTORICAL}[mkind]
        phrase = rng.choice(bank[label])
        sentence = f"Patient {phrase}."
        # Offsets of the phrase within the final text are resolved after joining.
        pieces.append(sentence)
        mentions.append(_Mention(label=label, kind=mkind, start=-1, end=-1, text=phrase))

    text = " ".join(pieces)
    # Resolve each mention's true offsets by searching forward, so repeated
    # phrases cannot collapse onto the same span.
    cursor = 0
    for m in mentions:
        idx = text.find(m.text, cursor)
        if idx < 0:  # pragma: no cover - defensive; phrases are inserted verbatim
            idx = text.find(m.text)
        m.start = idx
        m.end = idx + len(m.text)
        cursor = m.end
    return text, mentions


def _jitter_span(span: Span, rng: random.Random, max_shift: int = 12) -> Span:
    """Shift and shrink a span, the way a re-tokenizing extractor would.

    Produces offsets that are close but not equal, which fails ``exact``
    matching while still clearing a 0.5 overlap threshold.
    """
    shift = rng.randint(-max_shift, max_shift)
    start = max(0, span.start + shift)
    end = max(start + 1, span.end + rng.randint(-max_shift, max_shift))
    return Span(start=start, end=end, text=span.text)


def _widen_to_context(text: str, span: Span) -> Span:
    """Expand a span backwards over the preceding sentence.

    This is the real-world boundary disagreement: the annotator marks the
    minimal phrase, the extractor quotes the surrounding context as its
    evidence. Both are pointing at the same finding, and the overlap falls
    below a 0.5 threshold, which is exactly the ``annotation_threshold``
    mechanism rather than a misreading.
    """
    prior = text.rfind(". ", 0, max(0, span.start - 1))
    # Step back one more sentence so the quote is wide enough to drop the
    # overlap below a half -- a single extra sentence would still clear it.
    prior = text.rfind(". ", 0, prior) if prior > 0 else prior
    start = 0 if prior < 0 else prior + 2
    dot = text.find(".", span.end)
    end = len(text) if dot < 0 else dot + 1
    return Span(start=start, end=end, text=text[start:end])


def generate(
    n_patients: int = 50,
    months: int = 12,
    labels: tuple[str, ...] | list[str] = ("diarrhea", "constipation"),
    seed: int = 0,
    *,
    notes_per_patient: tuple[int, int] = (2, 6),
    base_prevalence: float = 0.35,
    careful_error_rate: float = 0.06,
    include_second_annotator: bool = True,
) -> SyntheticDataset:
    """Generate a complete synthetic validation dataset.

    Determinism: identical arguments give byte-identical output. Every random
    draw comes from one :class:`random.Random` seeded with ``seed``.

    Args:
        n_patients: Number of patients.
        months: Span of note timestamps after each patient's index date.
        labels: Labels to generate. Must be in :data:`SUPPORTED_LABELS`.
        seed: Seed for all randomness.
        notes_per_patient: Inclusive range of notes per patient.
        base_prevalence: Target gold-positive rate per label.
        careful_error_rate: Flip rate for the ``careful_reader`` system.
        include_second_annotator: Also produce a second rater's records, for
            inter-annotator agreement and for the adjudication path.

    Returns:
        A :class:`SyntheticDataset`.

    Raises:
        ValueError: If a label is unsupported or a numeric argument is out of range.
    """
    labels_t = tuple(labels)
    unsupported = [x for x in labels_t if x not in SUPPORTED_LABELS]
    if unsupported:
        raise ValueError(f"unsupported label(s) {unsupported}; supported: {list(SUPPORTED_LABELS)}")
    if not labels_t:
        raise ValueError("at least one label is required")
    if n_patients < 1:
        raise ValueError(f"n_patients must be >= 1, got {n_patients}")
    if not 0.0 <= base_prevalence <= 1.0:
        raise ValueError(f"base_prevalence must be in [0, 1], got {base_prevalence}")
    if not 0.0 <= careful_error_rate <= 1.0:
        raise ValueError(f"careful_error_rate must be in [0, 1], got {careful_error_rate}")
    lo, hi = notes_per_patient
    if lo < 1 or hi < lo:
        raise ValueError(f"notes_per_patient must be (lo>=1, hi>=lo), got {notes_per_patient}")

    rng = random.Random(seed)
    notes: list[dict[str, Any]] = []
    gold: list[Annotation] = []
    rater_b: list[Annotation] = []
    preds: dict[str, list[Prediction]] = {"careful_reader": [], "negation_blind": []}
    index_rows: list[dict[str, Any]] = []

    note_counter = 0
    for pi in range(n_patients):
        patient_id = f"P{pi + 1:05d}"
        index_date = EPOCH + timedelta(days=rng.randint(0, 900))
        index_rows.append({"patient_id": patient_id, "index_date": index_date.date().isoformat()})

        n_notes = rng.randint(lo, hi)
        day_span = max(1, int(months * 30.4))
        offsets = sorted(rng.randint(0, day_span) for _ in range(n_notes))

        for off in offsets:
            note_counter += 1
            note_id = f"N{note_counter:07d}"
            note_dt = index_date + timedelta(days=off, hours=rng.randint(0, 10))
            truth = {lab: rng.random() < base_prevalence for lab in labels_t}
            text, mentions = _compose_note(rng, labels_t, truth)

            notes.append(
                {
                    "note_id": note_id,
                    "patient_id": patient_id,
                    "note_datetime": note_dt,
                    "text": text,
                }
            )

            # Drawn once per note: a single model call produced every label's
            # answer, so the token counts and latency belong to that call.
            cr_tokens_in = rng.randint(380, 900)
            cr_tokens_out = rng.randint(20, 90)
            cr_latency = round(rng.uniform(0.4, 2.2), 3)
            nb_latency = round(rng.uniform(0.2, 1.1), 3)

            by_label: dict[str, list[_Mention]] = {}
            for m in mentions:
                by_label.setdefault(m.label, []).append(m)

            for label in labels_t:
                truth_v = truth[label]
                ms = by_label.get(label, [])
                affirm = next((m for m in ms if m.kind == "affirmative"), None)
                other = next((m for m in ms if m.kind != "affirmative"), None)
                hpo = LABEL_CODES[label]

                gold_span: Span | None = None
                if affirm is not None:
                    gold_span = Span(start=affirm.start, end=affirm.end, text=affirm.text)
                elif other is not None and rng.random() < 0.5:
                    # The annotator marked the candidate mention and judged it
                    # negative. This is what makes annotation_threshold and the
                    # negative-with-a-span case reachable.
                    gold_span = Span(start=other.start, end=other.end, text=other.text)

                gold.append(
                    Annotation(
                        annotator="adjudicated",
                        note_id=note_id,
                        label=label,
                        value=truth_v,
                        patient_id=patient_id,
                        span=gold_span,
                        code=Code(system="HPO", code=hpo, display=_display(hpo))
                        if truth_v
                        else None,
                        note_datetime=note_dt,
                        adjudicated=True,
                    )
                )

                if include_second_annotator:
                    # A second rater who agrees most of the time. Disagreements
                    # concentrate on the negated and historical mentions, which
                    # is where real raters disagree too.
                    flip_p = 0.18 if (other is not None and not truth_v) else 0.04
                    b_val = (not truth_v) if rng.random() < flip_p else truth_v
                    rater_b.append(
                        Annotation(
                            annotator="rater_b",
                            note_id=note_id,
                            label=label,
                            value=b_val,
                            patient_id=patient_id,
                            span=gold_span,
                            note_datetime=note_dt,
                            adjudicated=False,
                        )
                    )

                # --- careful_reader: mostly right, spans a little off --------
                cr_val = (not truth_v) if rng.random() < careful_error_rate else truth_v
                cr_span: Span | None = None
                if cr_val and gold_span is not None:
                    roll = rng.random()
                    if roll < 0.18:
                        # Over-quotes the surrounding context: a boundary
                        # disagreement, not a misread finding.
                        cr_span = _widen_to_context(text, gold_span)
                    elif roll < 0.38:
                        cr_span = _jitter_span(gold_span, rng)
                    else:
                        cr_span = gold_span
                elif cr_val and affirm is None and other is not None:
                    cr_span = Span(start=other.start, end=other.end, text=other.text)
                preds["careful_reader"].append(
                    Prediction(
                        system="careful_reader",
                        note_id=note_id,
                        label=label,
                        value=cr_val,
                        patient_id=patient_id,
                        span=cr_span,
                        code=(
                            Code(system="HPO", code=hpo, display=_display(hpo)) if cr_val else None
                        ),
                        note_datetime=note_dt,
                        confidence=round(rng.uniform(0.70, 0.99), 3)
                        if cr_val
                        else round(rng.uniform(0.01, 0.30), 3),
                        run=RunMeta(
                            model="synthetic-careful-v1",
                            prompt_version="p1",
                            tokens_in=cr_tokens_in,
                            tokens_out=cr_tokens_out,
                            latency_s=cr_latency,
                            # One pass over the note answers every label, so
                            # all of this note's records share a call id and
                            # are charged once.
                            call_id=note_id,
                        ),
                    )
                )

                # --- negation_blind: fires on any mention at all -------------
                nb_val = bool(ms)
                nb_span: Span | None = None
                if nb_val:
                    # nb_val is True only when `ms` is non-empty, so one of
                    # these is always present.
                    pick = affirm or other or ms[0]
                    nb_span = Span(start=pick.start, end=pick.end, text=pick.text)
                nb_code = hpo
                if nb_val and rng.random() < 0.12 and len(labels_t) > 1:
                    # Occasionally attaches a neighbouring concept's code,
                    # which is what ontology_mismatch is meant to catch.
                    other_label = rng.choice([x for x in labels_t if x != label])
                    nb_code = LABEL_CODES[other_label]
                preds["negation_blind"].append(
                    Prediction(
                        system="negation_blind",
                        note_id=note_id,
                        label=label,
                        value=nb_val,
                        patient_id=patient_id,
                        span=nb_span,
                        code=(
                            Code(system="HPO", code=nb_code, display=_display(nb_code))
                            if nb_val
                            else None
                        ),
                        note_datetime=note_dt,
                        confidence=round(rng.uniform(0.55, 0.95), 3) if nb_val else 0.1,
                        # No token counts: this mock stands in for an extractor
                        # that reports no usage metadata, so the cost path is
                        # exercised in its "unknown" branch.
                        run=RunMeta(
                            model="synthetic-blind-v1",
                            prompt_version="p1",
                            latency_s=nb_latency,
                            call_id=note_id,
                        ),
                    )
                )

    index = pd.DataFrame(index_rows)
    return SyntheticDataset(
        notes=notes,
        annotations=gold,
        predictions=preds,
        index=index,
        labels=labels_t,
        seed=seed,
        second_annotator=rater_b if include_second_annotator else [],
    )


def _display(code: str) -> str | None:
    entry = BUNDLED_MAP.get(code)
    return str(entry["display"]) if entry else None
