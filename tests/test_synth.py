"""The synthetic generator: determinism, correctness of its own spans, and PHI safety."""

from __future__ import annotations

import json
import re

import pytest

from clinval.io import read_annotations, read_index_dates, read_predictions
from clinval.synth import SUPPORTED_LABELS, generate


class TestDeterminism:
    def test_same_seed_same_output(self):
        a = generate(n_patients=8, seed=3)
        b = generate(n_patients=8, seed=3)
        assert [n["text"] for n in a.notes] == [n["text"] for n in b.notes]
        assert [x.value for x in a.annotations] == [x.value for x in b.annotations]
        assert [p.value for p in a.all_predictions] == [p.value for p in b.all_predictions]

    def test_different_seeds_differ(self):
        a = generate(n_patients=8, seed=1)
        b = generate(n_patients=8, seed=2)
        assert [n["text"] for n in a.notes] != [n["text"] for n in b.notes]

    def test_written_files_are_byte_identical_across_runs(self, tmp_path):
        p1, p2 = tmp_path / "a", tmp_path / "b"
        generate(n_patients=6, seed=0).write(p1)
        generate(n_patients=6, seed=0).write(p2)
        for name in ("gold.jsonl", "pred.jsonl", "notes.jsonl", "index.csv"):
            assert (p1 / name).read_bytes() == (p2 / name).read_bytes()


class TestSpanCorrectness:
    """The generator must not be the source of the span bugs the harness hunts."""

    def test_gold_spans_index_the_note_text_exactly(self):
        data = generate(n_patients=25, labels=("diarrhea", "constipation", "vomiting"), seed=7)
        texts = {n["note_id"]: n["text"] for n in data.notes}
        checked = 0
        for a in data.annotations:
            if a.span and a.span.text:
                assert texts[a.note_id][a.span.start : a.span.end] == a.span.text
                checked += 1
        assert checked > 50

    def test_unjittered_prediction_spans_index_exactly(self):
        data = generate(n_patients=25, labels=("diarrhea", "constipation"), seed=7)
        texts = {n["note_id"]: n["text"] for n in data.notes}
        checked = 0
        for p in data.predictions["negation_blind"]:
            if p.span and p.span.text:
                assert texts[p.note_id][p.span.start : p.span.end] == p.span.text
                checked += 1
        assert checked > 50

    def test_spans_are_within_the_note(self):
        data = generate(n_patients=15, seed=2)
        texts = {n["note_id"]: n["text"] for n in data.notes}
        for a in data.annotations:
            if a.span:
                assert 0 <= a.span.start <= a.span.end <= len(texts[a.note_id])


class TestNoRealData:
    """PHI safety is structural here: the generator has no source to leak from."""

    def test_identifiers_are_sequential_placeholders(self):
        data = generate(n_patients=10, seed=0)
        assert all(re.fullmatch(r"P\d{5}", p) for p in data.patients)
        assert all(re.fullmatch(r"N\d{7}", n["note_id"]) for n in data.notes)

    def test_note_text_comes_only_from_the_fixed_sentence_bank(self):
        from clinval.synth.generator import (
            _AFFIRM,
            _FILLER,
            _HISTORICAL,
            _NEGATED,
            _OPENERS,
        )

        bank = set(_FILLER) | set(_OPENERS)
        for group in (_AFFIRM, _NEGATED, _HISTORICAL):
            for phrases in group.values():
                bank.update(f"Patient {p}." for p in phrases)

        data = generate(n_patients=20, labels=("diarrhea", "constipation"), seed=4)
        for note in data.notes:
            for sentence in note["text"].split(". "):
                s = sentence if sentence.endswith(".") else sentence + "."
                assert s in bank, f"unexpected sentence in generated note: {s!r}"

    def test_no_digits_that_could_look_like_an_identifier_in_note_text(self):
        data = generate(n_patients=20, seed=0)
        for note in data.notes:
            assert not re.search(r"\d{4,}", note["text"])

    def test_dates_derive_from_a_fixed_epoch(self):
        from clinval.synth.generator import EPOCH

        data = generate(n_patients=10, seed=0)
        assert read_index_dates_min(data) >= EPOCH.date().isoformat()


def read_index_dates_min(data) -> str:
    return str(min(data.index["index_date"]))[:10]


class TestStructure:
    def test_one_gold_record_per_note_label(self):
        labels = ("diarrhea", "constipation")
        data = generate(n_patients=10, labels=labels, seed=0)
        assert len(data.annotations) == len(data.notes) * len(labels)

    def test_gold_is_adjudicated(self):
        data = generate(n_patients=5, seed=0)
        assert all(a.adjudicated for a in data.annotations)

    def test_two_systems_with_full_coverage(self):
        data = generate(n_patients=10, labels=("diarrhea",), seed=0)
        assert set(data.predictions) == {"careful_reader", "negation_blind"}
        for recs in data.predictions.values():
            assert len(recs) == len(data.notes)

    def test_second_annotator_is_unadjudicated(self):
        data = generate(n_patients=8, seed=0)
        assert data.second_annotator
        assert all(not a.adjudicated for a in data.second_annotator)
        assert all(a.annotator == "rater_b" for a in data.second_annotator)

    def test_second_annotator_can_be_disabled(self):
        data = generate(n_patients=5, seed=0, include_second_annotator=False)
        assert data.second_annotator == []

    def test_prevalence_near_the_target(self):
        data = generate(n_patients=120, labels=("diarrhea",), seed=0, base_prevalence=0.35)
        assert 0.28 < data.prevalence()["diarrhea"] < 0.42

    def test_index_has_one_row_per_patient(self):
        data = generate(n_patients=14, seed=0)
        assert len(data.index) == 14
        assert data.index["patient_id"].is_unique

    def test_careful_reader_outperforms_negation_blind(self):
        """The two mocks must differ in the way the demo claims they do."""
        data = generate(n_patients=40, labels=("diarrhea",), seed=0)
        gold = {(a.note_id, a.label): a.value for a in data.annotations}
        scores = {}
        for name, recs in data.predictions.items():
            correct = sum(1 for p in recs if gold[(p.note_id, p.label)] == p.value)
            scores[name] = correct / len(recs)
        assert scores["careful_reader"] > scores["negation_blind"]

    def test_negation_blind_produces_false_positives_on_negated_text(self):
        data = generate(n_patients=30, labels=("diarrhea",), seed=0)
        gold = {(a.note_id, a.label): a.value for a in data.annotations}
        fps = [
            p
            for p in data.predictions["negation_blind"]
            if p.value and not gold[(p.note_id, p.label)]
        ]
        assert fps
        assert any(p.span and "denies" in (p.span.text or "") for p in fps)


class TestSplitByPatient:
    def test_splits_are_patient_disjoint(self):
        data = generate(n_patients=20, seed=0)
        dev, test = data.split_by_patient(0.5)
        assert set(dev.patients).isdisjoint(test.patients)

    def test_notes_follow_their_patient(self):
        """Splitting by note would leak a patient across the split."""
        data = generate(n_patients=20, seed=0)
        dev, test = data.split_by_patient(0.5)
        assert {n["note_id"] for n in dev.notes}.isdisjoint(n["note_id"] for n in test.notes)
        for note in dev.notes:
            assert str(note["patient_id"]) in set(dev.patients)

    def test_all_patients_accounted_for(self):
        data = generate(n_patients=21, seed=0)
        dev, test = data.split_by_patient(0.5)
        assert sorted(dev.patients + test.patients) == data.patients

    def test_index_is_subset(self):
        data = generate(n_patients=20, seed=0)
        dev, _test = data.split_by_patient(0.5)
        assert set(dev.index["patient_id"]) == set(dev.patients)

    def test_predictions_are_subset(self):
        data = generate(n_patients=20, seed=0)
        dev, _ = data.split_by_patient(0.5)
        dev_notes = {n["note_id"] for n in dev.notes}
        for recs in dev.predictions.values():
            assert all(p.note_id in dev_notes for p in recs)

    def test_both_sides_non_empty_even_at_extremes(self):
        data = generate(n_patients=4, seed=0)
        for frac in (0.01, 0.5, 0.99):
            dev, test = data.split_by_patient(frac)
            assert dev.patients and test.patients

    @pytest.mark.parametrize("frac", [0.0, 1.0, -0.2, 1.5])
    def test_invalid_fraction(self, frac):
        with pytest.raises(ValueError, match="dev_fraction"):
            generate(n_patients=5, seed=0).split_by_patient(frac)

    def test_reproducible(self):
        data = generate(n_patients=20, seed=0)
        a, _ = data.split_by_patient(0.5, seed=42)
        b, _ = data.split_by_patient(0.5, seed=42)
        assert a.patients == b.patients


class TestWrite:
    def test_artifacts_written_and_readable(self, tmp_path):
        data = generate(n_patients=8, labels=("diarrhea", "constipation"), seed=0)
        paths = data.write(tmp_path)
        assert read_predictions(paths["pred"])
        assert read_annotations(paths["gold"])
        assert len(read_index_dates(paths["index"])) == 8
        assert "gold_two_annotators" in paths

    def test_notes_file_is_valid_jsonl(self, tmp_path):
        paths = generate(n_patients=5, seed=0).write(tmp_path)
        rows = [
            json.loads(line) for line in paths["notes"].read_text().splitlines() if line.strip()
        ]
        assert rows
        assert all({"note_id", "patient_id", "note_datetime", "text"} <= set(r) for r in rows)

    def test_per_system_files(self, tmp_path):
        paths = generate(n_patients=5, seed=0).write(tmp_path)
        assert "pred_careful_reader" in paths
        assert "pred_negation_blind" in paths
        assert read_predictions(paths["pred_careful_reader"])

    def test_two_annotator_file_is_resolvable_after_adjudication(self, tmp_path):
        from clinval.aggregate import adjudicate, resolve_gold

        paths = generate(n_patients=6, seed=0).write(tmp_path)
        recs = read_annotations(paths["gold_two_annotators"])
        assert len({a.annotator for a in recs}) == 2
        resolve_gold(adjudicate(recs, rule="agreement"))

    def test_prefix(self, tmp_path):
        paths = generate(n_patients=4, seed=0).write(tmp_path, prefix="dev_")
        assert paths["gold"].name == "dev_gold.jsonl"


class TestValidation:
    def test_unsupported_label(self):
        with pytest.raises(ValueError, match="unsupported label"):
            generate(labels=("pneumonia",))

    def test_no_labels(self):
        with pytest.raises(ValueError, match="at least one label"):
            generate(labels=())

    def test_zero_patients(self):
        with pytest.raises(ValueError, match="n_patients"):
            generate(n_patients=0)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_prevalence(self, bad):
        with pytest.raises(ValueError, match="base_prevalence"):
            generate(n_patients=2, base_prevalence=bad)

    def test_bad_error_rate(self):
        with pytest.raises(ValueError, match="careful_error_rate"):
            generate(n_patients=2, careful_error_rate=2.0)

    @pytest.mark.parametrize("bad", [(0, 3), (5, 2)])
    def test_bad_notes_per_patient(self, bad):
        with pytest.raises(ValueError, match="notes_per_patient"):
            generate(n_patients=2, notes_per_patient=bad)

    def test_all_supported_labels_work(self):
        data = generate(n_patients=4, labels=SUPPORTED_LABELS, seed=0)
        assert set(data.labels) == set(SUPPORTED_LABELS)
