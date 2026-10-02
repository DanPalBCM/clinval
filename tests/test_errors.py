"""Error mechanism tagging.

SPEC §8.1 requires a triggering fixture for **every** mechanism in the
taxonomy. :class:`TestEveryMechanismHasAFixture` is that requirement written as
a test, and it fails if a tag becomes unreachable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from clinval.aggregate import aggregate_to_windows, align_predictions
from clinval.errors import (
    ERROR_COLUMNS,
    MECHANISMS,
    confusion_export,
    has_negation_cue,
    has_temporal_cue,
    tag_errors,
)
from clinval.schemas.config import EvalConfig, Matching, SpanMatching, Unit, Window
from tests.conftest import make_gold, make_pred

NOTE_CFG = EvalConfig(unit=Unit.NOTE)


def tag(preds, golds, config=None, *, index=None):
    """Align, aggregate, and tag in one step."""
    cfg = config or NOTE_CFG
    aligned = align_predictions(preds, golds, cfg)
    windowed = aggregate_to_windows(aligned, cfg.window, index=index, unit=cfg.unit)
    return tag_errors(aligned, cfg, windowed=windowed)


def mechanisms_of(table) -> set[str]:
    return set(table.df["mechanism"]) if len(table) else set()


class TestCueDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "denies diarrhea or loose stools",
            "no evidence of vomiting",
            "negative for constipation",
            "ruled out appendicitis",
            "patient is free of nausea",
            "symptoms have resolved",
        ],
    )
    def test_negation_cues(self, text):
        assert has_negation_cue(text)

    @pytest.mark.parametrize(
        "text",
        [
            "reports loose stools three times daily",
            "continues to have diarrhea",
            "vomited twice this morning",
        ],
    )
    def test_no_false_negation(self, text):
        assert not has_negation_cue(text)

    @pytest.mark.parametrize(
        "text",
        [
            "history of diarrhea two years ago",
            "h/o constipation",
            "family history of chronic diarrhea",
            "past medical history includes nausea",
            "evaluated last year",
            "mother has similar symptoms",
            "plans to start therapy",
        ],
    )
    def test_temporal_cues(self, text):
        assert has_temporal_cue(text)

    def test_none_is_safe(self):
        assert not has_negation_cue(None)
        assert not has_temporal_cue(None)
        assert not has_negation_cue("")


class TestFalsePositiveMechanisms:
    def test_missing_evidence_when_no_span(self):
        t = tag([make_pred("N1", value=True)], [make_gold("N1", value=False)])
        row = t.df.iloc[0]
        assert row["error_type"] == "FP"
        assert row["mechanism"] == "missing_evidence"
        assert "no supporting span" in row["mechanism_reason"]

    def test_misreading_on_a_negated_quote(self):
        t = tag(
            [make_pred("N1", value=True, span=(0, 31, "denies diarrhea or loose stools"))],
            [make_gold("N1", value=False)],
        )
        row = t.df.iloc[0]
        assert row["mechanism"] == "misreading"
        assert "negation cue" in row["mechanism_reason"]

    def test_temporal_error_on_a_historical_quote(self):
        t = tag(
            [make_pred("N1", value=True, span=(0, 33, "history of diarrhea two years ago"))],
            [make_gold("N1", value=False)],
        )
        assert t.df.iloc[0]["mechanism"] == "temporal_error"

    def test_temporal_error_on_a_family_history_quote(self):
        t = tag(
            [make_pred("N1", value=True, span=(0, 38, "family history of diarrhea in the mother"))],
            [make_gold("N1", value=False)],
        )
        assert t.df.iloc[0]["mechanism"] == "temporal_error"

    def test_other_when_nothing_explains_it(self):
        """A clean quote, a negative gold, and no cue: the fallback must be reachable."""
        t = tag(
            [make_pred("N1", value=True, span=(0, 26, "reports loose stools daily"))],
            [make_gold("N1", value=False)],
        )
        row = t.df.iloc[0]
        assert row["mechanism"] == "other"
        assert row["mechanism_reason"]

    def test_annotation_threshold_on_a_low_overlap_candidate(self):
        """The annotator marked the mention and called it negative; spans barely overlap."""
        cfg = EvalConfig(
            unit=Unit.NOTE, matching=Matching(span=SpanMatching(mode="overlap", min_overlap=0.8))
        )
        t = tag(
            [make_pred("N1", value=True, span=(0, 40, "reports loose stools three times daily"))],
            [make_gold("N1", value=False, span=(30, 70, "and has been afebrile throughout"))],
            cfg,
        )
        row = t.df.iloc[0]
        assert row["mechanism"] == "annotation_threshold"
        assert "below the" in row["mechanism_reason"]


class TestFalseNegativeMechanisms:
    def test_misreading_when_evidence_was_quoted(self):
        t = tag(
            [make_pred("N1", value=False)],
            [make_gold("N1", value=True, span=(0, 26, "reports loose stools daily"))],
        )
        row = t.df.iloc[0]
        assert row["error_type"] == "FN"
        assert row["mechanism"] == "misreading"

    def test_missing_evidence_when_gold_has_no_span(self):
        t = tag([make_pred("N1", value=False)], [make_gold("N1", value=True)])
        row = t.df.iloc[0]
        assert row["error_type"] == "FN"
        assert row["mechanism"] == "missing_evidence"

    def test_temporal_error_when_gold_span_is_historical(self):
        t = tag(
            [make_pred("N1", value=False)],
            [make_gold("N1", value=True, span=(0, 31, "history of diarrhea last year"))],
        )
        assert t.df.iloc[0]["mechanism"] == "temporal_error"

    def test_uncredited_positive_is_blamed_on_its_evidence_not_on_misreading(self):
        """Regression: the uncredited check must survive numpy bool scalars.

        Under ``require_span`` an unsupported positive scores as a negative,
        so it surfaces as an FN. Its mechanism is the reason it lost credit --
        a boundary disagreement -- not a misreading of the finding.
        """
        cfg = EvalConfig(
            unit=Unit.NOTE,
            matching=Matching(
                span=SpanMatching(mode="overlap", min_overlap=0.5, require_span=True)
            ),
        )
        t = tag(
            [make_pred("N1", value=True, span=(0, 100, "z" * 100))],
            [make_gold("N1", value=True, span=(60, 100, "z" * 40))],
            cfg,
        )
        row = t.df.iloc[0]
        assert row["error_type"] == "FN"
        assert row["mechanism"] == "annotation_threshold"
        assert "not credited under require_span" in row["mechanism_reason"]


class TestMatchMechanisms:
    def test_ontology_mismatch_when_codes_disagree(self):
        t = tag(
            [make_pred("N1", value=True, span=(0, 10, "diarrhea"), code=("HPO", "HP:0002019"))],
            [make_gold("N1", value=True, span=(0, 10, "diarrhea"), code=("HPO", "HP:0002014"))],
        )
        row = t.df.iloc[0]
        assert row["error_type"] == "MATCH"
        assert row["mechanism"] == "ontology_mismatch"

    def test_annotation_threshold_on_an_over_quoted_span(self):
        cfg = EvalConfig(
            unit=Unit.NOTE, matching=Matching(span=SpanMatching(mode="overlap", min_overlap=0.5))
        )
        t = tag(
            [make_pred("N1", value=True, span=(0, 100, "a" * 100))],
            [make_gold("N1", value=True, span=(60, 100, "b" * 40))],
            cfg,
        )
        row = t.df.iloc[0]
        assert row["error_type"] == "MATCH"
        assert row["mechanism"] == "annotation_threshold"

    def test_misreading_when_spans_do_not_overlap_at_all(self):
        cfg = EvalConfig(
            unit=Unit.NOTE, matching=Matching(span=SpanMatching(mode="overlap", min_overlap=0.5))
        )
        t = tag(
            [make_pred("N1", value=True, span=(0, 20, "x" * 20))],
            [make_gold("N1", value=True, span=(50, 70, "y" * 20))],
            cfg,
        )
        row = t.df.iloc[0]
        assert row["mechanism"] == "misreading"
        assert "do not overlap" in row["mechanism_reason"]

    def test_missing_evidence_when_only_one_side_quotes(self):
        cfg = EvalConfig(
            unit=Unit.NOTE,
            matching=Matching(
                span=SpanMatching(mode="overlap", min_overlap=0.5, require_span=True)
            ),
        )
        t = tag(
            [make_pred("N1", value=True)],
            [make_gold("N1", value=True, span=(0, 10, "diarrhea"))],
            cfg,
        )
        row = t.df.iloc[0]
        assert row["mechanism"] == "missing_evidence"

    def test_agreement_with_matching_spans_is_not_an_error(self):
        t = tag(
            [make_pred("N1", value=True, span=(0, 10, "diarrhea"))],
            [make_gold("N1", value=True, span=(0, 10, "diarrhea"))],
        )
        assert len(t) == 0


class TestWindowPlacement:
    """``window_placement`` needs the window grid, so it is tested separately."""

    def _index(self) -> pd.DataFrame:
        return pd.DataFrame({"patient_id": ["P1"], "index_date": [pd.Timestamp("2022-01-01")]})

    def _cfg(self) -> EvalConfig:
        return EvalConfig(unit=Unit.PATIENT_WINDOW, window=Window(size="6M", n_windows=2))

    def test_false_positive_in_the_wrong_window(self):
        """Gold is positive in window 1; the system fired in window 0."""
        early = pd.Timestamp("2022-02-01").to_pydatetime()
        late = pd.Timestamp("2022-08-01").to_pydatetime()
        preds = [
            make_pred(
                "N1", value=True, span=(0, 26, "reports loose stools daily"), note_datetime=early
            ),
            make_pred(
                "N2", value=True, span=(0, 26, "reports loose stools daily"), note_datetime=late
            ),
        ]
        golds = [
            make_gold("N1", value=False, note_datetime=early),
            make_gold(
                "N2", value=True, span=(0, 26, "reports loose stools daily"), note_datetime=late
            ),
        ]
        t = tag(preds, golds, self._cfg(), index=self._index())
        fps = t.df[t.df["error_type"] == "FP"]
        assert len(fps) == 1
        assert fps.iloc[0]["mechanism"] == "window_placement"
        assert "gold is positive" in fps.iloc[0]["mechanism_reason"]

    def test_false_negative_when_the_system_filed_it_elsewhere(self):
        early = pd.Timestamp("2022-02-01").to_pydatetime()
        late = pd.Timestamp("2022-08-01").to_pydatetime()
        preds = [
            make_pred(
                "N1", value=True, span=(0, 26, "reports loose stools daily"), note_datetime=early
            ),
            make_pred("N2", value=False, note_datetime=late),
        ]
        golds = [
            make_gold("N1", value=False, note_datetime=early),
            make_gold(
                "N2", value=True, span=(0, 26, "reports loose stools daily"), note_datetime=late
            ),
        ]
        t = tag(preds, golds, self._cfg(), index=self._index())
        fns = t.df[t.df["error_type"] == "FN"]
        assert len(fns) == 1
        assert fns.iloc[0]["mechanism"] == "window_placement"
        assert "system is positive" in fns.iloc[0]["mechanism_reason"]

    def test_every_note_in_a_multi_note_unit_gets_window_context(self):
        """Regression: window context came only from the representative note.

        A unit aggregating several notes must give each of them a window, or
        the notes that are neither the representative nor the gold note carry
        no window and can be mistagged.
        """
        cfg = self._cfg()
        dates = ["2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01"]
        preds, golds = [], []
        for i, d in enumerate(dates, start=1):
            dt = pd.Timestamp(d).to_pydatetime()
            preds.append(
                make_pred(
                    f"N{i}",
                    value=True,
                    span=(0, 26, "reports loose stools daily"),
                    note_datetime=dt,
                )
            )
            golds.append(make_gold(f"N{i}", value=False, note_datetime=dt))

        aligned = align_predictions(preds, golds, cfg)
        windowed = aggregate_to_windows(aligned, cfg.window, index=self._index(), unit=cfg.unit)
        assert len(windowed.df) == 1
        assert len(windowed.df.iloc[0]["note_ids"]) == 4

        table = tag_errors(aligned, cfg, windowed=windowed)
        assert len(table) == 4
        # every error row knows which window it came from
        assert (table.df["window"].astype(str).str.len() > 0).all()

    def test_without_a_windowed_frame_the_tag_is_never_assigned(self):
        """Honest degradation: no window grid means no window-placement claim."""
        early = pd.Timestamp("2022-02-01").to_pydatetime()
        preds = [
            make_pred(
                "N1", value=True, span=(0, 26, "reports loose stools daily"), note_datetime=early
            )
        ]
        golds = [make_gold("N1", value=False, note_datetime=early)]
        aligned = align_predictions(preds, golds, NOTE_CFG)
        t = tag_errors(aligned, NOTE_CFG, windowed=None)
        assert "window_placement" not in mechanisms_of(t)


class TestEveryMechanismHasAFixture:
    """SPEC §8.1 and §11: no mechanism may be unreachable."""

    def test_all_seven_mechanisms_are_triggerable(self):
        index = pd.DataFrame({"patient_id": ["P1"], "index_date": [pd.Timestamp("2022-01-01")]})
        cfg = EvalConfig(
            unit=Unit.PATIENT_WINDOW,
            window=Window(size="6M", n_windows=2),
            matching=Matching(span=SpanMatching(mode="overlap", min_overlap=0.5)),
        )
        early = pd.Timestamp("2022-02-01").to_pydatetime()
        mid = pd.Timestamp("2022-03-01").to_pydatetime()
        late = pd.Timestamp("2022-08-01").to_pydatetime()

        preds = [
            # missing_evidence: positive with no span
            make_pred("N1", "a", True, note_datetime=early),
            # misreading: negated quote read as positive
            make_pred(
                "N2",
                "b",
                True,
                span=(0, 31, "denies diarrhea or loose stools"),
                note_datetime=early,
            ),
            # temporal_error: historical quote read as current
            make_pred(
                "N3",
                "c",
                True,
                span=(0, 33, "history of diarrhea two years ago"),
                note_datetime=early,
            ),
            # ontology_mismatch: right label, wrong code
            make_pred(
                "N4",
                "d",
                True,
                span=(0, 8, "diarrhea"),
                code=("HPO", "HP:0002019"),
                note_datetime=early,
            ),
            # annotation_threshold: over-quoted evidence
            make_pred("N5", "e", True, span=(0, 100, "z" * 100), note_datetime=mid),
            # other: clean quote, nothing explains the disagreement
            make_pred(
                "N6", "f", True, span=(0, 26, "reports loose stools daily"), note_datetime=mid
            ),
            # window_placement: fires in window 0, gold is in window 1
            make_pred(
                "N7", "g", True, span=(0, 26, "reports loose stools daily"), note_datetime=mid
            ),
            make_pred(
                "N8", "g", True, span=(0, 26, "reports loose stools daily"), note_datetime=late
            ),
        ]
        golds = [
            make_gold("N1", "a", False, note_datetime=early),
            make_gold("N2", "b", False, note_datetime=early),
            make_gold("N3", "c", False, note_datetime=early),
            make_gold(
                "N4",
                "d",
                True,
                span=(0, 8, "diarrhea"),
                code=("HPO", "HP:0002014"),
                note_datetime=early,
            ),
            make_gold("N5", "e", True, span=(60, 100, "z" * 40), note_datetime=mid),
            make_gold("N6", "f", False, note_datetime=mid),
            make_gold("N7", "g", False, note_datetime=mid),
            make_gold(
                "N8", "g", True, span=(0, 26, "reports loose stools daily"), note_datetime=late
            ),
        ]
        t = tag(preds, golds, cfg, index=index)
        found = mechanisms_of(t)
        missing = set(MECHANISMS) - found
        assert not missing, (
            f"mechanisms with no triggering fixture: {sorted(missing)}; found {sorted(found)}"
        )


class TestErrorTable:
    def test_columns_match_the_spec(self):
        t = tag([make_pred("N1", value=True)], [make_gold("N1", value=False)])
        assert list(t.df.columns) == list(ERROR_COLUMNS)
        # SPEC §6.4 names these; they must be present and first.
        for col in (
            "error_id",
            "system",
            "label",
            "unit_key",
            "error_type",
            "patient_id",
            "window",
            "mechanism",
            "span_text",
            "note_id",
        ):
            assert col in t.df.columns

    def test_error_ids_unique_and_stable(self):
        preds = [make_pred(f"N{i}", value=True) for i in range(5)]
        golds = [make_gold(f"N{i}", value=False) for i in range(5)]
        a = tag(preds, golds)
        b = tag(preds, golds)
        assert a.df["error_id"].is_unique
        assert a.df["error_id"].tolist() == b.df["error_id"].tolist()

    def test_by_mechanism_lists_every_tag_including_zeros(self):
        t = tag([make_pred("N1", value=True)], [make_gold("N1", value=False)])
        bm = t.by_mechanism()
        assert list(bm["mechanism"]) == list(MECHANISMS)
        assert bm["n"].sum() == len(t)

    def test_breakdowns(self):
        preds = [make_pred("N1", "a", True), make_pred("N2", "b", True)]
        golds = [make_gold("N1", "a", False), make_gold("N2", "b", False)]
        t = tag(preds, golds)
        assert set(t.by_label()["label"]) == {"a", "b"}
        assert not t.by_system().empty
        assert not t.by_window().empty

    def test_empty_breakdowns_do_not_crash(self):
        t = tag([make_pred("N1", value=True)], [make_gold("N1", value=True)])
        assert len(t) == 0
        assert t.by_label().empty
        assert t.by_window().empty
        assert t.by_mechanism()["n"].sum() == 0

    def test_sample_rejects_unknown_mechanism(self):
        t = tag([make_pred("N1", value=True)], [make_gold("N1", value=False)])
        with pytest.raises(ValueError, match="unknown mechanism"):
            t.sample("not_a_mechanism")

    def test_to_csv(self, tmp_path):
        t = tag([make_pred("N1", value=True)], [make_gold("N1", value=False)])
        out = tmp_path / "errors.csv"
        t.to_csv(str(out))
        assert out.exists()
        assert "error_id" in out.read_text()

    def test_unscored_rows_are_not_errors(self):
        """A prediction with no gold is not a mistake; there is nothing to be wrong about."""
        t = tag([make_pred("N1", value=True)], [make_gold("N2", value=True)])
        assert len(t) == 0


class TestConfusionExport:
    def test_cells_named(self):
        preds = [
            make_pred("N1", "a", True),
            make_pred("N2", "a", True),
            make_pred("N3", "a", False),
            make_pred("N4", "a", False),
        ]
        golds = [
            make_gold("N1", "a", True),
            make_gold("N2", "a", False),
            make_gold("N3", "a", True),
            make_gold("N4", "a", False),
        ]
        aligned = align_predictions(preds, golds, NOTE_CFG)
        windowed = aggregate_to_windows(aligned, unit=Unit.NOTE)
        out = confusion_export(windowed)
        assert set(out["cell"]) == {"TP", "FP", "FN", "TN"}

    def test_unscored_cell(self):
        aligned = align_predictions(
            [make_pred("N1", value=True)], [make_gold("N2", value=True)], NOTE_CFG
        )
        windowed = aggregate_to_windows(aligned, unit=Unit.NOTE)
        out = confusion_export(windowed)
        assert "unscored" in set(out["cell"])

    def test_empty(self):
        aligned = align_predictions([], [], NOTE_CFG)
        windowed = aggregate_to_windows(aligned, unit=Unit.NOTE)
        assert confusion_export(windowed).empty
