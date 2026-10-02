"""IO: malformed rows, missing fields, duplicates, index dates, negatives."""

from __future__ import annotations

import json

import pytest

from clinval.exceptions import DuplicateRecordError, RecordError
from clinval.io import (
    materialize_negatives,
    read_annotations,
    read_index_dates,
    read_json,
    read_predictions,
    write_annotations,
    write_predictions,
)
from tests.conftest import make_gold, make_pred


def write_lines(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


class TestReadPredictions:
    def test_roundtrip(self, tmp_path):
        recs = [
            make_pred("N1", "diarrhea", True, span=(5, 12, "loose")),
            make_pred("N2", "diarrhea", False),
        ]
        p = tmp_path / "pred.jsonl"
        assert write_predictions(recs, p) == 2
        back = read_predictions(p)
        assert [r.note_id for r in back] == ["N1", "N2"]
        assert back[0].span is not None
        assert back[0].span.text == "loose"
        assert back[1].span is None

    def test_blank_lines_and_comments_skipped(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(
            p,
            [
                '{"system":"s","note_id":"N1","label":"l","value":true}',
                "",
                "   ",
                "// a comment line",
                '{"system":"s","note_id":"N2","label":"l","value":false}',
            ],
        )
        assert len(read_predictions(p)) == 2

    def test_malformed_json_reports_line_number(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(
            p,
            [
                '{"system":"s","note_id":"N1","label":"l","value":true}',
                '{"system":"s","note_id":"N2","label":"l","value":',
            ],
        )
        with pytest.raises(RecordError) as exc:
            read_predictions(p)
        assert ":2]" in str(exc.value)

    def test_missing_required_field_names_the_field(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(p, ['{"system":"s","note_id":"N1","label":"l"}'])
        with pytest.raises(RecordError) as exc:
            read_predictions(p)
        assert "value" in str(exc.value)
        assert ":1]" in str(exc.value)

    def test_non_object_line_rejected(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(p, ["[1, 2, 3]"])
        with pytest.raises(RecordError, match="must be a JSON object"):
            read_predictions(p)

    def test_unknown_field_rejected(self, tmp_path):
        """extra='forbid' means a typo in a field name is caught, not ignored."""
        p = tmp_path / "pred.jsonl"
        write_lines(p, ['{"system":"s","note_id":"N1","label":"l","value":true,"valeu":1}'])
        with pytest.raises(RecordError):
            read_predictions(p)

    def test_duplicate_detected(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        row = '{"system":"s","note_id":"N1","label":"l","value":true}'
        write_lines(p, [row, row])
        with pytest.raises(DuplicateRecordError, match="duplicate prediction"):
            read_predictions(p)

    def test_duplicate_allowed_when_requested(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        row = '{"system":"s","note_id":"N1","label":"l","value":true}'
        write_lines(p, [row, row])
        assert len(read_predictions(p, allow_duplicates=True)) == 2

    def test_same_note_label_different_systems_is_not_duplicate(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(
            p,
            [
                '{"system":"a","note_id":"N1","label":"l","value":true}',
                '{"system":"b","note_id":"N1","label":"l","value":true}',
            ],
        )
        assert len(read_predictions(p)) == 2

    def test_missing_file(self, tmp_path):
        with pytest.raises(RecordError, match="file not found"):
            read_predictions(tmp_path / "nope.jsonl")

    def test_confidence_out_of_range_rejected(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(p, ['{"system":"s","note_id":"N1","label":"l","value":true,"confidence":1.4}'])
        with pytest.raises(RecordError):
            read_predictions(p)

    def test_inverted_span_rejected(self, tmp_path):
        p = tmp_path / "pred.jsonl"
        write_lines(
            p,
            ['{"system":"s","note_id":"N1","label":"l","value":true,"span":{"start":9,"end":4}}'],
        )
        with pytest.raises(RecordError, match="precedes start"):
            read_predictions(p)

    def test_label_with_colon_preserved(self, tmp_path):
        """fhirground emits labels like `Condition:E11.9`; clinval must not split them."""
        p = tmp_path / "pred.jsonl"
        write_lines(
            p,
            ['{"system":"fhirground","note_id":"N1","label":"Condition:E11.9","value":true}'],
        )
        assert read_predictions(p)[0].label == "Condition:E11.9"


class TestReadAnnotations:
    def test_two_annotators_is_not_a_duplicate(self, tmp_path):
        recs = [
            make_gold("N1", annotator="rater_a", adjudicated=False),
            make_gold("N1", annotator="rater_b", adjudicated=False),
        ]
        p = tmp_path / "gold.jsonl"
        write_annotations(recs, p)
        assert len(read_annotations(p)) == 2

    def test_same_annotator_twice_is_a_duplicate(self, tmp_path):
        p = tmp_path / "gold.jsonl"
        row = '{"annotator":"r","note_id":"N1","label":"l","value":true}'
        write_lines(p, [row, row])
        with pytest.raises(DuplicateRecordError, match="duplicate annotation"):
            read_annotations(p)

    def test_adjudicated_defaults_false(self, tmp_path):
        p = tmp_path / "gold.jsonl"
        write_lines(p, ['{"annotator":"r","note_id":"N1","label":"l","value":true}'])
        assert read_annotations(p)[0].adjudicated is False

    def test_annotation_rejects_prediction_only_fields(self, tmp_path):
        """A human annotator has no confidence score; the schema says so."""
        p = tmp_path / "gold.jsonl"
        write_lines(
            p, ['{"annotator":"r","note_id":"N1","label":"l","value":true,"confidence":0.9}']
        )
        with pytest.raises(RecordError):
            read_annotations(p)


class TestIndexDates:
    def test_canonical_columns(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("patient_id,index_date\nP1,2022-01-01\nP2,2022-03-15\n")
        df = read_index_dates(p)
        assert list(df.columns) == ["patient_id", "index_date"]
        assert len(df) == 2
        assert df["index_date"].dt.tz is None

    @pytest.mark.parametrize(
        ("pcol", "dcol"),
        [("subject_id", "anchor_date"), ("person_id", "date"), ("patient", "index_datetime")],
    )
    def test_column_aliases(self, tmp_path, pcol, dcol):
        """An index file exported from OMOP or MEDS works without renaming."""
        p = tmp_path / "idx.csv"
        p.write_text(f"{pcol},{dcol}\nP1,2022-01-01\n")
        assert read_index_dates(p)["patient_id"].tolist() == ["P1"]

    def test_missing_patient_column_lists_what_it_looked_for(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("pid,index_date\nP1,2022-01-01\n")
        with pytest.raises(RecordError, match="no patient column"):
            read_index_dates(p)

    def test_missing_date_column(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("patient_id,when\nP1,2022-01-01\n")
        with pytest.raises(RecordError, match="no index-date column"):
            read_index_dates(p)

    def test_unparseable_date(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("patient_id,index_date\nP1,not-a-date\n")
        with pytest.raises(RecordError, match="unparseable index date"):
            read_index_dates(p)

    def test_duplicate_patient_rejected(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("patient_id,index_date\nP1,2022-01-01\nP1,2022-06-01\n")
        with pytest.raises(DuplicateRecordError, match="more than one index date"):
            read_index_dates(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("patient_id,index_date\n")
        with pytest.raises(RecordError, match="no rows"):
            read_index_dates(p)

    def test_timezone_stripped(self, tmp_path):
        p = tmp_path / "idx.csv"
        p.write_text("patient_id,index_date\nP1,2022-01-01T00:00:00+02:00\n")
        assert read_index_dates(p)["index_date"].dt.tz is None


class TestMaterializeNegatives:
    def test_fills_missing_pairs(self):
        preds = [make_pred("N1", "diarrhea", True)]
        out = materialize_negatives(
            preds, note_ids=["N1", "N2"], labels=["diarrhea", "constipation"]
        )
        assert len(out) == 4
        keys = {(p.note_id, p.label, p.value) for p in out}
        assert ("N1", "diarrhea", True) in keys
        assert ("N2", "constipation", False) in keys

    def test_does_not_modify_input(self):
        preds = [make_pred("N1", "diarrhea", True)]
        before = len(preds)
        materialize_negatives(preds, note_ids=["N1", "N2"], labels=["diarrhea"])
        assert len(preds) == before

    def test_never_overwrites_an_existing_record(self):
        preds = [make_pred("N1", "diarrhea", True)]
        out = materialize_negatives(preds, note_ids=["N1"], labels=["diarrhea"])
        assert len(out) == 1
        assert out[0].value is True

    def test_inherits_patient_and_datetime_from_siblings(self):
        preds = [make_pred("N1", "diarrhea", True, patient_id="P9")]
        out = materialize_negatives(preds, note_ids=["N1"], labels=["diarrhea", "constipation"])
        added = next(p for p in out if p.label == "constipation")
        assert added.patient_id == "P9"
        assert added.note_datetime is not None

    def test_reads_universe_from_notes_file(self, tmp_path):
        notes = tmp_path / "notes.jsonl"
        notes.write_text(
            json.dumps(
                {"note_id": "N1", "patient_id": "P1", "note_datetime": "2022-01-10T09:00:00"}
            )
            + "\n"
            + json.dumps({"note_id": "N2", "patient_id": "P2"})
            + "\n"
        )
        out = materialize_negatives([make_pred("N1", "diarrhea", True)], notes_path=notes)
        assert {p.note_id for p in out} == {"N1", "N2"}
        n2 = next(p for p in out if p.note_id == "N2")
        assert n2.patient_id == "P2"
        assert n2.value is False

    def test_notes_file_without_note_id_rejected(self, tmp_path):
        notes = tmp_path / "notes.jsonl"
        notes.write_text(json.dumps({"patient_id": "P1"}) + "\n")
        with pytest.raises(RecordError, match="no note_id"):
            materialize_negatives([make_pred()], notes_path=notes)

    def test_requires_a_universe(self):
        with pytest.raises(RecordError, match="needs note_ids or notes_path"):
            materialize_negatives([make_pred()])

    def test_per_system(self):
        preds = [
            make_pred("N1", "diarrhea", True, system="a"),
            make_pred("N1", "diarrhea", True, system="b"),
        ]
        out = materialize_negatives(preds, note_ids=["N1", "N2"], labels=["diarrhea"])
        assert len(out) == 4
        assert {(p.system, p.note_id) for p in out} == {
            ("a", "N1"),
            ("a", "N2"),
            ("b", "N1"),
            ("b", "N2"),
        }


class TestReadJson:
    def test_malformed(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text("{not json")
        with pytest.raises(RecordError, match="malformed JSON"):
            read_json(p)

    def test_missing(self, tmp_path):
        with pytest.raises(RecordError, match="file not found"):
            read_json(tmp_path / "nope.json")
