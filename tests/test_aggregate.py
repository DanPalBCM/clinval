"""Alignment, window assignment, aggregation, and gold adjudication."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from clinval.aggregate import (
    adjudicate,
    aggregate_to_windows,
    align_predictions,
    assign_windows,
    build_windows,
    resolve_gold,
)
from clinval.exceptions import AggregationError
from clinval.schemas.config import (
    Duration,
    EvalConfig,
    Matching,
    SpanMatching,
    TieBreak,
    Unit,
    Window,
)
from tests.conftest import make_gold, make_pred

NOTE_CFG = EvalConfig(unit=Unit.NOTE)


def idx(patients: dict[str, str]) -> pd.DataFrame:
    """Index-date frame from ``{patient_id: iso_date}``."""
    return pd.DataFrame(
        {
            "patient_id": list(patients),
            "index_date": [pd.Timestamp(v) for v in patients.values()],
        }
    )


class TestResolveGold:
    def test_single_annotator(self):
        resolved, raters = resolve_gold([make_gold("N1", adjudicated=False, annotator="r1")])
        assert len(resolved) == 1
        assert raters == ("r1",)

    def test_adjudicated_record_wins(self):
        recs = [
            make_gold("N1", value=False, annotator="r1", adjudicated=False),
            make_gold("N1", value=False, annotator="r2", adjudicated=False),
            make_gold("N1", value=True, annotator="adj", adjudicated=True),
        ]
        resolved, _ = resolve_gold(recs)
        assert resolved[("N1", "diarrhea")].value is True

    def test_two_adjudicated_records_is_a_contradiction(self):
        recs = [
            make_gold("N1", value=True, annotator="a", adjudicated=True),
            make_gold("N1", value=False, annotator="b", adjudicated=True),
        ]
        with pytest.raises(AggregationError, match="records marked adjudicated"):
            resolve_gold(recs)

    def test_disagreeing_annotators_without_adjudication_refused(self):
        """Averaging raters would hide the disagreement the study measures."""
        recs = [
            make_gold("N1", value=True, annotator="r1", adjudicated=False),
            make_gold("N1", value=False, annotator="r2", adjudicated=False),
        ]
        with pytest.raises(AggregationError, match="they disagree"):
            resolve_gold(recs)

    def test_agreeing_annotators_still_need_adjudication(self):
        recs = [
            make_gold("N1", value=True, annotator="r1", adjudicated=False),
            make_gold("N1", value=True, annotator="r2", adjudicated=False),
        ]
        with pytest.raises(AggregationError, match="they agree"):
            resolve_gold(recs)


class TestAdjudicate:
    def test_agreement_rule_drops_disagreements(self):
        recs = [
            make_gold("N1", "a", True, annotator="r1", adjudicated=False),
            make_gold("N1", "a", True, annotator="r2", adjudicated=False),
            make_gold("N2", "a", True, annotator="r1", adjudicated=False),
            make_gold("N2", "a", False, annotator="r2", adjudicated=False),
        ]
        out = adjudicate(recs, rule="agreement")
        assert len(out) == 1
        assert out[0].note_id == "N1"
        assert out[0].adjudicated is True

    def test_majority_rule(self):
        recs = [
            make_gold("N1", "a", True, annotator="r1", adjudicated=False),
            make_gold("N1", "a", True, annotator="r2", adjudicated=False),
            make_gold("N1", "a", False, annotator="r3", adjudicated=False),
        ]
        out = adjudicate(recs, rule="majority")
        assert len(out) == 1
        assert out[0].value is True

    def test_majority_tie_is_dropped(self):
        recs = [
            make_gold("N1", "a", True, annotator="r1", adjudicated=False),
            make_gold("N1", "a", False, annotator="r2", adjudicated=False),
        ]
        assert adjudicate(recs, rule="majority") == []

    def test_first_rule(self):
        recs = [
            make_gold("N1", "a", False, annotator="zz", adjudicated=False),
            make_gold("N1", "a", True, annotator="aa", adjudicated=False),
        ]
        out = adjudicate(recs, rule="first")
        assert out[0].value is True

    def test_existing_adjudication_preserved(self):
        recs = [make_gold("N1", "a", True, annotator="adj", adjudicated=True)]
        assert adjudicate(recs)[0].annotator == "adj"

    def test_unknown_rule(self):
        with pytest.raises(AggregationError, match="unknown adjudication rule"):
            adjudicate([make_gold()], rule="vibes")

    def test_output_is_resolvable(self):
        recs = [
            make_gold("N1", "a", True, annotator="r1", adjudicated=False),
            make_gold("N1", "a", True, annotator="r2", adjudicated=False),
        ]
        resolved, _ = resolve_gold(adjudicate(recs))
        assert len(resolved) == 1


class TestAlign:
    def test_union_of_pairs_nothing_dropped(self):
        preds = [make_pred("N1"), make_pred("N2")]
        golds = [make_gold("N2"), make_gold("N3")]
        a = align_predictions(preds, golds, NOTE_CFG)
        assert len(a) == 3
        assert set(a.df["note_id"]) == {"N1", "N2", "N3"}

    def test_pred_without_gold_is_unscored(self):
        a = align_predictions([make_pred("N1")], [], NOTE_CFG)
        row = a.df.iloc[0]
        assert row["pred_present"]
        assert not row["gold_present"]
        assert pd.isna(row["gold"])

    def test_gold_without_pred_is_unscored_not_negative(self):
        """SPEC §6.1: silence is not a negative."""
        a = align_predictions([], [make_gold("N1")], NOTE_CFG)
        row = a.df.iloc[0]
        assert not row["pred_present"]
        assert pd.isna(row["pred"])

    def test_coverage_reports_the_gap(self):
        preds = [make_pred("N1")]
        golds = [make_gold("N1"), make_gold("N2")]
        a = align_predictions(preds, golds, NOTE_CFG)
        assert a.coverage()["sysA"] == pytest.approx(0.5)

    def test_multiple_systems_kept_separate(self):
        preds = [make_pred("N1", system="a"), make_pred("N1", system="b")]
        a = align_predictions(preds, [make_gold("N1")], NOTE_CFG)
        assert a.systems() == ["a", "b"]
        assert len(a) == 2

    def test_label_filter_applied(self):
        cfg = EvalConfig(unit=Unit.NOTE, labels=["diarrhea"])
        preds = [make_pred("N1", "diarrhea"), make_pred("N1", "vomiting")]
        golds = [make_gold("N1", "diarrhea"), make_gold("N1", "vomiting")]
        a = align_predictions(preds, golds, cfg)
        assert set(a.df["label"]) == {"diarrhea"}

    def test_patient_id_falls_back_to_gold(self):
        preds = [make_pred("N1", patient_id=None)]
        golds = [make_gold("N1", patient_id="P7")]
        a = align_predictions(preds, golds, NOTE_CFG)
        assert a.df.iloc[0]["patient_id"] == "P7"

    def test_require_span_downgrades_an_unsupported_positive(self):
        """With require_span on, a positive without matching evidence is not credited."""
        cfg = EvalConfig(
            unit=Unit.NOTE,
            matching=Matching(
                span=SpanMatching(mode="overlap", min_overlap=0.5, require_span=True)
            ),
        )
        a = align_predictions(
            [make_pred("N1", value=True, span=(0, 10, "x" * 10))],
            [make_gold("N1", value=True, span=(50, 60, "y" * 10))],
            cfg,
        )
        row = a.df.iloc[0]
        assert bool(row["pred_raw"]) is True
        assert bool(row["credited"]) is False
        assert bool(row["pred"]) is False

    def test_require_span_off_leaves_pred_untouched(self):
        a = align_predictions(
            [make_pred("N1", value=True, span=(0, 10, "x" * 10))],
            [make_gold("N1", value=True, span=(50, 60, "y" * 10))],
            NOTE_CFG,
        )
        row = a.df.iloc[0]
        assert bool(row["pred"]) is True
        assert bool(row["credited"]) is True

    def test_empty_inputs(self):
        a = align_predictions([], [], NOTE_CFG)
        assert len(a) == 0


class TestBuildWindows:
    def test_single_window(self):
        g = build_windows(idx({"P1": "2022-01-01"}), Window(size=Duration(n=6, unit="M")))
        assert len(g) == 1
        assert g.iloc[0]["window_start"] == pd.Timestamp("2022-01-01")
        assert g.iloc[0]["window_end"] == pd.Timestamp("2022-07-01")

    def test_multiple_consecutive_windows_do_not_overlap(self):
        g = build_windows(
            idx({"P1": "2022-01-01"}), Window(size=Duration(n=6, unit="M"), n_windows=3)
        )
        assert len(g) == 3
        assert g.iloc[0]["window_end"] == g.iloc[1]["window_start"]
        assert g.iloc[1]["window_end"] == g.iloc[2]["window_start"]

    def test_include_prior_adds_a_window_before_the_anchor(self):
        g = build_windows(
            idx({"P1": "2022-01-01"}),
            Window(size=Duration(n=6, unit="M"), n_windows=1, include_prior=True),
        )
        assert sorted(g["window_index"]) == [-1, 0]
        prior = g[g["window_index"] == -1].iloc[0]
        assert prior["window_start"] == pd.Timestamp("2021-07-01")
        assert prior["window_end"] == pd.Timestamp("2022-01-01")

    def test_calendar_months_not_30_days(self):
        g = build_windows(idx({"P1": "2022-01-31"}), Window(size=Duration(n=1, unit="M")))
        assert g.iloc[0]["window_end"] == pd.Timestamp("2022-02-28")

    @pytest.mark.parametrize(
        ("spec", "expected_end"),
        [("30D", "2022-01-31"), ("2W", "2022-01-15"), ("6M", "2022-07-01"), ("1Y", "2023-01-01")],
    )
    def test_duration_units(self, spec, expected_end):
        g = build_windows(idx({"P1": "2022-01-01"}), Window(size=Duration.parse(spec)))
        assert g.iloc[0]["window_end"] == pd.Timestamp(expected_end)

    def test_missing_index_column(self):
        with pytest.raises(AggregationError, match="missing column"):
            build_windows(pd.DataFrame({"patient_id": ["P1"]}), Window())


class TestAssignWindows:
    CFG = EvalConfig(
        unit=Unit.PATIENT_WINDOW, window=Window(size=Duration(n=6, unit="M"), n_windows=2)
    )

    def _assign(self, notes: list[tuple[str, str]], patient: str = "P1") -> pd.DataFrame:
        preds = [
            make_pred(nid, patient_id=patient, note_datetime=pd.Timestamp(dt).to_pydatetime())
            for nid, dt in notes
        ]
        golds = [
            make_gold(nid, patient_id=patient, note_datetime=pd.Timestamp(dt).to_pydatetime())
            for nid, dt in notes
        ]
        a = align_predictions(preds, golds, self.CFG)
        return assign_windows(a, idx({patient: "2022-01-01"}), self.CFG.window)

    def test_note_inside_first_window(self):
        out = self._assign([("N1", "2022-02-01")])
        assert out.iloc[0]["window_index"] == 0

    def test_note_exactly_on_the_lower_boundary_lands_in_that_window(self):
        """Half-open [start, end): the anchor itself is in window 0."""
        out = self._assign([("N1", "2022-01-01")])
        assert out.iloc[0]["window_index"] == 0

    def test_note_exactly_on_the_internal_boundary_lands_in_the_later_window(self):
        """2022-07-01 ends window 0 and starts window 1. Counted once, in window 1."""
        out = self._assign([("N1", "2022-07-01")])
        assert out.iloc[0]["window_index"] == 1

    def test_note_exactly_on_the_final_boundary_is_outside(self):
        out = self._assign([("N1", "2023-01-01")])
        assert pd.isna(out.iloc[0]["window_index"])
        assert out.iloc[0]["window_miss_reason"] == "outside_all_windows"

    def test_note_before_the_anchor_is_outside(self):
        out = self._assign([("N1", "2021-12-31")])
        assert pd.isna(out.iloc[0]["window_index"])
        assert out.iloc[0]["window_miss_reason"] == "outside_all_windows"

    def test_note_without_a_timestamp_is_flagged(self):
        preds = [make_pred("N1", patient_id="P1", note_datetime=None)]
        golds = [make_gold("N1", patient_id="P1", note_datetime=None)]
        a = align_predictions(preds, golds, self.CFG)
        out = assign_windows(a, idx({"P1": "2022-01-01"}), self.CFG.window)
        assert out.iloc[0]["window_miss_reason"] == "no_note_datetime"

    def test_patient_missing_from_index_is_flagged(self):
        preds = [make_pred("N1", patient_id="PX", note_datetime=datetime(2022, 2, 1))]
        golds = [make_gold("N1", patient_id="PX", note_datetime=datetime(2022, 2, 1))]
        a = align_predictions(preds, golds, self.CFG)
        out = assign_windows(a, idx({"P1": "2022-01-01"}), self.CFG.window)
        assert out.iloc[0]["window_miss_reason"] == "patient_not_in_index"

    def test_rows_are_kept_not_dropped(self):
        out = self._assign([("N1", "2022-02-01"), ("N2", "2030-01-01")])
        assert len(out) == 2


class TestAggregateToWindows:
    CFG = EvalConfig(
        unit=Unit.PATIENT_WINDOW, window=Window(size=Duration(n=6, unit="M"), n_windows=2)
    )

    def test_any_positive_rule(self):
        """One documented episode in a window makes that window positive."""
        preds = [
            make_pred("N1", value=False, note_datetime=datetime(2022, 2, 1)),
            make_pred("N2", value=True, note_datetime=datetime(2022, 3, 1)),
        ]
        golds = [
            make_gold("N1", value=False, note_datetime=datetime(2022, 2, 1)),
            make_gold("N2", value=True, note_datetime=datetime(2022, 3, 1)),
        ]
        a = align_predictions(preds, golds, self.CFG)
        w = aggregate_to_windows(a, self.CFG.window, index=idx({"P1": "2022-01-01"}))
        assert len(w) == 1
        assert bool(w.df.iloc[0]["pred"]) is True
        assert w.df.iloc[0]["n_notes"] == 2

    def test_notes_split_across_windows(self):
        preds = [
            make_pred("N1", value=True, note_datetime=datetime(2022, 2, 1)),
            make_pred("N2", value=False, note_datetime=datetime(2022, 9, 1)),
        ]
        golds = [
            make_gold("N1", value=True, note_datetime=datetime(2022, 2, 1)),
            make_gold("N2", value=False, note_datetime=datetime(2022, 9, 1)),
        ]
        a = align_predictions(preds, golds, self.CFG)
        w = aggregate_to_windows(a, self.CFG.window, index=idx({"P1": "2022-01-01"}))
        assert len(w) == 2
        assert sorted(w.df["window_index"].tolist()) == [0, 1]

    @pytest.mark.parametrize(
        ("tie_break", "expected"),
        [(TieBreak.EARLIEST, "N1"), (TieBreak.LATEST, "N2")],
    )
    def test_tie_break_picks_the_representative(self, tie_break, expected):
        window = Window(size=Duration(n=6, unit="M"), n_windows=1, tie_break=tie_break)
        cfg = EvalConfig(unit=Unit.PATIENT_WINDOW, window=window)
        preds = [
            make_pred("N1", value=True, note_datetime=datetime(2022, 2, 1)),
            make_pred("N2", value=True, note_datetime=datetime(2022, 4, 1)),
        ]
        golds = [
            make_gold("N1", value=True, note_datetime=datetime(2022, 2, 1)),
            make_gold("N2", value=True, note_datetime=datetime(2022, 4, 1)),
        ]
        a = align_predictions(preds, golds, cfg)
        w = aggregate_to_windows(a, window, index=idx({"P1": "2022-01-01"}))
        assert w.df.iloc[0]["representative_note_id"] == expected

    def test_highest_confidence_tie_break(self):
        window = Window(
            size=Duration(n=6, unit="M"), n_windows=1, tie_break=TieBreak.HIGHEST_CONFIDENCE
        )
        cfg = EvalConfig(unit=Unit.PATIENT_WINDOW, window=window)
        preds = [
            make_pred("N1", value=True, note_datetime=datetime(2022, 2, 1), confidence=0.6),
            make_pred("N2", value=True, note_datetime=datetime(2022, 4, 1), confidence=0.95),
        ]
        golds = [make_gold(n, value=True, note_datetime=datetime(2022, 2, 1)) for n in ("N1", "N2")]
        a = align_predictions(preds, golds, cfg)
        w = aggregate_to_windows(a, window, index=idx({"P1": "2022-01-01"}))
        assert w.df.iloc[0]["representative_note_id"] == "N2"

    def test_note_unit_is_a_passthrough(self):
        a = align_predictions([make_pred("N1")], [make_gold("N1")], NOTE_CFG)
        w = aggregate_to_windows(a, unit=Unit.NOTE)
        assert len(w) == 1
        assert w.df.iloc[0]["unit_key"] == "N1"

    def test_note_unit_is_idempotent(self):
        """SPEC §8.2: aggregating an already-aggregated frame is a no-op."""
        a = align_predictions(
            [make_pred("N1"), make_pred("N2")], [make_gold("N1"), make_gold("N2")], NOTE_CFG
        )
        once = aggregate_to_windows(a, unit=Unit.NOTE)
        twice = aggregate_to_windows(
            type(a)(df=once.df, unit=Unit.NOTE, annotators=a.annotators), unit=Unit.NOTE
        )
        assert len(once) == len(twice)
        assert once.df["unit_key"].tolist() == twice.df["unit_key"].tolist()

    def test_patient_unit_ignores_time(self):
        cfg = EvalConfig(unit=Unit.PATIENT)
        preds = [
            make_pred("N1", value=False, note_datetime=datetime(2022, 2, 1)),
            make_pred("N2", value=True, note_datetime=datetime(2030, 2, 1)),
        ]
        golds = [
            make_gold("N1", value=False, note_datetime=datetime(2022, 2, 1)),
            make_gold("N2", value=True, note_datetime=datetime(2030, 2, 1)),
        ]
        a = align_predictions(preds, golds, cfg)
        w = aggregate_to_windows(a, unit=Unit.PATIENT)
        assert len(w) == 1
        assert w.df.iloc[0]["unit_key"] == "P1"
        assert bool(w.df.iloc[0]["pred"]) is True

    def test_patient_window_without_index_is_an_error(self):
        a = align_predictions([make_pred("N1")], [make_gold("N1")], self.CFG)
        with pytest.raises(AggregationError, match="requires per-patient index dates"):
            aggregate_to_windows(a, self.CFG.window, index=None, unit=Unit.PATIENT_WINDOW)

    def test_patient_unit_without_patient_ids_is_an_error(self):
        cfg = EvalConfig(unit=Unit.PATIENT)
        a = align_predictions(
            [make_pred("N1", patient_id=None)], [make_gold("N1", patient_id=None)], cfg
        )
        with pytest.raises(AggregationError, match="needs patient_id"):
            aggregate_to_windows(a, unit=Unit.PATIENT)

    def test_uncredited_count_carried(self):
        cfg = EvalConfig(
            unit=Unit.PATIENT_WINDOW,
            window=Window(size=Duration(n=6, unit="M")),
            matching=Matching(
                span=SpanMatching(mode="overlap", min_overlap=0.5, require_span=True)
            ),
        )
        a = align_predictions(
            [
                make_pred(
                    "N1", value=True, span=(0, 10, "x" * 10), note_datetime=datetime(2022, 2, 1)
                )
            ],
            [
                make_gold(
                    "N1", value=True, span=(50, 60, "y" * 10), note_datetime=datetime(2022, 2, 1)
                )
            ],
            cfg,
        )
        w = aggregate_to_windows(a, cfg.window, index=idx({"P1": "2022-01-01"}))
        assert w.df.iloc[0]["n_uncredited"] == 1

    def test_empty(self):
        a = align_predictions([], [], self.CFG)
        w = aggregate_to_windows(a, self.CFG.window, index=idx({"P1": "2022-01-01"}))
        assert len(w) == 0
