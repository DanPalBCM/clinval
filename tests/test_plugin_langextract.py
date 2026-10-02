"""The LangExtract adapter, exercised against the real langextract data model.

No test here reaches the network. :func:`predictions_from_documents` is the
pure half of the adapter: it takes ``AnnotatedDocument`` objects built
in-process from the genuine ``langextract.data`` classes, so the mapping is
verified against the library's actual field names rather than a guess.

The one test that would call a model is marked ``live_llm`` and skipped unless
``--run-live-llm`` is passed.
"""

from __future__ import annotations

import pytest

from clinval.exceptions import PluginError
from clinval.plugins.base import Note, validate_predictions
from clinval.plugins.langextract import (
    LangExtractConfig,
    LangExtractPlugin,
    langextract_available,
    predictions_from_documents,
)

lx = pytest.importorskip("langextract", reason="optional extra `langextract` not installed")

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

NOTES = [
    Note(note_id="N1", text="Patient reports loose stools daily.", patient_id="P1"),
    Note(note_id="N2", text="Patient denies diarrhea.", patient_id="P1"),
]


def extraction(
    cls: str, text: str, start: int | None = None, end: int | None = None, **attrs: object
):
    """Build a real ``langextract.data.Extraction``."""
    interval = None
    if start is not None and end is not None:
        interval = lx.data.CharInterval(start_pos=start, end_pos=end)
    return lx.data.Extraction(
        extraction_class=cls,
        extraction_text=text,
        char_interval=interval,
        attributes=dict(attrs) if attrs else None,
    )


def document(*extractions, text: str = "x"):
    """Build a real ``langextract.data.AnnotatedDocument``."""
    return lx.data.AnnotatedDocument(extractions=list(extractions), text=text)


class TestLibraryShape:
    """Guards on the langextract API this adapter depends on.

    If a future langextract release renames these, the adapter's assumptions
    are wrong and these tests say so directly rather than through a confusing
    mapping failure.
    """

    def test_char_interval_field_names(self):
        ci = lx.data.CharInterval(start_pos=3, end_pos=9)
        assert ci.start_pos == 3
        assert ci.end_pos == 9

    def test_extraction_fields_present(self):
        ex = extraction("diarrhea", "loose stools", 0, 12)
        assert ex.extraction_class == "diarrhea"
        assert ex.extraction_text == "loose stools"
        assert hasattr(ex, "alignment_status")
        assert hasattr(ex, "attributes")

    def test_annotated_document_holds_extractions(self):
        doc = document(extraction("diarrhea", "loose stools", 0, 12))
        assert len(doc.extractions) == 1

    def test_library_still_reports_no_token_usage(self):
        """D-003: the basis for reporting cost as unknown.

        If langextract ever adds usage metadata, this test fails and the
        adapter should start populating tokens_in/tokens_out.
        """
        from langextract.core import types as lx_types

        assert {f.name for f in __import__("dataclasses").fields(lx_types.ScoredOutput)} == {
            "score",
            "output",
        }
        doc = document(extraction("diarrhea", "loose stools", 0, 12))
        for attr in ("usage", "usage_metadata", "tokens_in", "tokens_out", "token_count"):
            assert not hasattr(doc, attr)


class TestPresenceRule:
    CFG = LangExtractConfig(labels=("diarrhea", "constipation"))

    def test_extraction_becomes_a_positive(self):
        docs = [document(extraction("diarrhea", "loose stools", 8, 20)), document()]
        preds = predictions_from_documents(docs, NOTES, self.CFG)
        by_key = {(p.note_id, p.label): p for p in preds}
        assert by_key[("N1", "diarrhea")].value is True
        assert by_key[("N1", "diarrhea")].span is not None
        assert by_key[("N1", "diarrhea")].span.start == 8
        assert by_key[("N1", "diarrhea")].span.end == 20

    def test_explicit_negatives_are_materialized(self):
        """clinval reads silence as unscored, so the adapter must fill it in."""
        docs = [document(extraction("diarrhea", "loose stools", 8, 20)), document()]
        preds = predictions_from_documents(docs, NOTES, self.CFG)
        assert len(preds) == 4  # 2 notes x 2 labels
        assert {(p.note_id, p.label): p.value for p in preds} == {
            ("N1", "diarrhea"): True,
            ("N1", "constipation"): False,
            ("N2", "diarrhea"): False,
            ("N2", "constipation"): False,
        }

    def test_labels_outside_the_target_set_are_ignored(self):
        docs = [document(extraction("fever", "febrile", 0, 7)), document()]
        preds = predictions_from_documents(docs, NOTES, self.CFG)
        assert all(p.label in self.CFG.labels for p in preds)
        assert all(p.value is False for p in preds)

    def test_class_to_label_mapping(self):
        cfg = LangExtractConfig(
            labels=("diarrhea",), class_to_label={"loose_stool_symptom": "diarrhea"}
        )
        docs = [document(extraction("loose_stool_symptom", "loose stools", 8, 20)), document()]
        preds = predictions_from_documents(docs, NOTES, cfg)
        assert next(p for p in preds if p.note_id == "N1").value is True

    def test_patient_and_datetime_passed_through(self):
        docs = [document(), document()]
        preds = predictions_from_documents(docs, NOTES, self.CFG)
        assert all(p.patient_id == "P1" for p in preds)

    def test_duplicate_class_keeps_the_located_one(self):
        docs = [
            document(
                extraction("diarrhea", "loose stools"),
                extraction("diarrhea", "watery stools", 30, 43),
            ),
            document(),
        ]
        preds = predictions_from_documents(docs, NOTES, self.CFG)
        hit = next(p for p in preds if p.note_id == "N1" and p.label == "diarrhea")
        assert hit.span is not None
        assert hit.span.start == 30

    def test_output_passes_plugin_validation(self):
        docs = [document(extraction("diarrhea", "loose stools", 8, 20)), document()]
        preds = predictions_from_documents(docs, NOTES, self.CFG)
        assert len(validate_predictions(preds, expect_labels=self.CFG.labels)) == 4


class TestNegationHandling:
    CFG = LangExtractConfig(labels=("diarrhea",), negation_attribute="negated")

    def test_negated_attribute_suppresses_the_positive(self):
        docs = [document(extraction("diarrhea", "denies diarrhea", 8, 23, negated=True))]
        preds = predictions_from_documents(docs, NOTES[:1], self.CFG)
        assert preds[0].value is False

    @pytest.mark.parametrize("raw", [True, "true", "yes", "1", "negated", "absent"])
    def test_truthy_negation_tokens(self, raw):
        docs = [document(extraction("diarrhea", "x", 0, 1, negated=raw))]
        assert predictions_from_documents(docs, NOTES[:1], self.CFG)[0].value is False

    @pytest.mark.parametrize("raw", [False, "false", "no", "0", "asserted", "present"])
    def test_falsy_negation_tokens(self, raw):
        docs = [document(extraction("diarrhea", "x", 0, 1, negated=raw))]
        assert predictions_from_documents(docs, NOTES[:1], self.CFG)[0].value is True

    def test_unreadable_attribute_counts_as_asserted_and_warns(self):
        """Guessing polarity from the text would turn an extraction bug into a
        clinval scoring bug, so the adapter says so instead."""
        docs = [document(extraction("diarrhea", "x", 0, 1, negated="perhaps"))]
        with pytest.warns(UserWarning, match="no readable"):
            preds = predictions_from_documents(docs, NOTES[:1], self.CFG)
        assert preds[0].value is True

    def test_missing_attribute_warns(self):
        docs = [document(extraction("diarrhea", "x", 0, 1))]
        with pytest.warns(UserWarning, match="does not detect negation"):
            predictions_from_documents(docs, NOTES[:1], self.CFG)

    def test_inverted_attribute_semantics(self):
        """An attribute named `present` means the opposite of `negated`."""
        cfg = LangExtractConfig(
            labels=("diarrhea",), negation_attribute="present", negated_means=False
        )
        docs = [document(extraction("diarrhea", "x", 0, 1, present=True))]
        assert predictions_from_documents(docs, NOTES[:1], cfg)[0].value is True

    def test_negation_disabled_treats_everything_as_asserted(self):
        cfg = LangExtractConfig(labels=("diarrhea",), negation_attribute=None)
        docs = [document(extraction("diarrhea", "denies diarrhea", 0, 15, negated=True))]
        preds = predictions_from_documents(docs, NOTES[:1], cfg)
        assert preds[0].value is True

    def test_a_negated_mention_does_not_cancel_a_positive_one(self):
        docs = [
            document(
                extraction("diarrhea", "denies diarrhea", 0, 15, negated=True),
                extraction("diarrhea", "loose stools today", 30, 48, negated=False),
            )
        ]
        preds = predictions_from_documents(docs, NOTES[:1], self.CFG)
        assert preds[0].value is True


class TestOffsetsAndAlignment:
    CFG = LangExtractConfig(labels=("diarrhea",))

    def test_missing_interval_yields_no_span(self):
        docs = [document(extraction("diarrhea", "loose stools"))]
        preds = predictions_from_documents(docs, NOTES[:1], self.CFG)
        assert preds[0].value is True
        assert preds[0].span is None

    def test_partial_interval_yields_no_span_rather_than_a_guess(self):
        ex = lx.data.Extraction(
            extraction_class="diarrhea",
            extraction_text="loose stools",
            char_interval=lx.data.CharInterval(start_pos=5, end_pos=None),
        )
        preds = predictions_from_documents([document(ex)], NOTES[:1], self.CFG)
        assert preds[0].span is None

    def test_require_offsets_raises_instead(self):
        cfg = LangExtractConfig(labels=("diarrhea",), require_offsets=True)
        docs = [document(extraction("diarrhea", "loose stools"))]
        with pytest.raises(PluginError, match="no usable char_interval"):
            predictions_from_documents(docs, NOTES[:1], cfg)

    def test_fuzzy_alignment_is_recorded_and_warned_about(self):
        ex = extraction("diarrhea", "loose stools", 0, 12)
        ex.alignment_status = lx.data.AlignmentStatus.MATCH_FUZZY
        with pytest.warns(UserWarning, match="non-exact span alignment"):
            preds = predictions_from_documents([document(ex)], NOTES[:1], self.CFG)
        assert preds[0].run is not None
        assert preds[0].run.model_dump()["alignment_status"] == "MATCH_FUZZY"

    def test_exact_alignment_does_not_warn(self):
        """Polarity is supplied here so the only possible warning is about alignment."""
        import warnings

        ex = extraction("diarrhea", "loose stools", 0, 12, negated=False)
        ex.alignment_status = lx.data.AlignmentStatus.MATCH_EXACT
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            predictions_from_documents([document(ex)], NOTES[:1], self.CFG)
        assert not [w for w in caught if "alignment" in str(w.message)]


class TestCostMetadata:
    CFG = LangExtractConfig(labels=("diarrhea",))

    def test_tokens_are_left_unset(self):
        """D-003: langextract reports no token counts, so clinval reports none."""
        docs = [document(extraction("diarrhea", "x", 0, 1))]
        preds = predictions_from_documents(docs, NOTES[:1], self.CFG, model_id="gemini-3.5-flash")
        run = preds[0].run
        assert run is not None
        assert run.tokens_in is None
        assert run.tokens_out is None
        assert run.model == "gemini-3.5-flash"

    def test_latency_is_carried_when_measured(self):
        docs = [document(extraction("diarrhea", "x", 0, 1))]
        preds = predictions_from_documents(docs, NOTES[:1], self.CFG, latencies={"N1": 2.41})
        assert preds[0].run is not None
        assert preds[0].run.latency_s == pytest.approx(2.41)

    def test_cost_is_reported_unknown_end_to_end(self):
        from clinval.cost import summarize_cost
        from clinval.schemas.config import CostModel

        docs = [document(extraction("diarrhea", "x", 0, 1))]
        preds = predictions_from_documents(docs, NOTES[:1], self.CFG, latencies={"N1": 1.0})
        rep = summarize_cost(preds, CostModel(input_per_1k=0.15, output_per_1k=0.60))
        s = rep.per_system["langextract"]
        assert s.cost is None
        assert s.cost_status == "unknown"
        assert s.latency_mean_s == pytest.approx(1.0)


class TestDocumentCountMismatch:
    def test_mismatch_is_an_error_not_a_silent_truncation(self):
        cfg = LangExtractConfig(labels=("diarrhea",))
        with pytest.raises(PluginError, match="clinval matches them positionally"):
            predictions_from_documents([document()], NOTES, cfg)

    def test_single_document_accepted_unwrapped(self):
        cfg = LangExtractConfig(labels=("diarrhea",))
        preds = predictions_from_documents(document(), NOTES[:1], cfg)
        assert len(preds) == 1


class TestPluginSurface:
    def test_availability_probe(self):
        assert langextract_available() is True

    def test_plugin_name_follows_the_config(self):
        plugin = LangExtractPlugin(LangExtractConfig(labels=("diarrhea",), system="my-pipeline"))
        assert plugin.name == "my-pipeline"

    def test_empty_notes_yields_nothing_without_calling_the_model(self):
        plugin = LangExtractPlugin(LangExtractConfig(labels=("diarrhea",)))
        assert list(plugin.predict([])) == []

    def test_satisfies_the_extractor_protocol(self):
        from clinval.plugins.base import ExtractorPlugin

        plugin = LangExtractPlugin(LangExtractConfig(labels=("diarrhea",)))
        assert isinstance(plugin, ExtractorPlugin)


@pytest.mark.live_llm
class TestLiveExtraction:
    def test_end_to_end_against_a_real_model(self):
        """Needs --run-live-llm, network access, and LANGEXTRACT_API_KEY."""
        plugin = LangExtractPlugin(
            LangExtractConfig(labels=("diarrhea",)),
            prompt_description=(
                "Extract symptom mentions. For each, add a `negated` attribute "
                "that is true when the note denies the symptom."
            ),
            examples=[
                lx.data.ExampleData(
                    text="Patient reports loose stools.",
                    extractions=[extraction("diarrhea", "loose stools", 16, 28, negated="false")],
                )
            ],
        )
        preds = list(plugin.predict(NOTES))
        assert preds
        validate_predictions(preds, expect_labels=("diarrhea",))


class TestBaseValidation:
    def test_rejects_non_predictions(self):
        with pytest.raises(PluginError, match="expected a clinval Prediction"):
            validate_predictions([{"not": "a prediction"}])  # type: ignore[list-item]

    def test_rejects_duplicates(self):
        from tests.conftest import make_pred

        p = make_pred("N1")
        with pytest.raises(PluginError, match="more than once"):
            validate_predictions([p, p])

    def test_rejects_unexpected_labels(self):
        from tests.conftest import make_pred

        with pytest.raises(PluginError, match="not in the target set"):
            validate_predictions([make_pred("N1", "fever")], expect_labels=("diarrhea",))
