"""Span and ontology matching."""

from __future__ import annotations

import pytest

from clinval.match.ontology import (
    BUNDLED_MAP,
    bundled_map,
    icd10_prefix,
    load_ontology_map,
    match_codes,
    normalize_code,
)
from clinval.match.spans import (
    fuzzy_text_score,
    jaccard_overlap,
    match_spans,
    overlap_fraction,
)
from clinval.schemas.config import OntologyMatching, SpanMatching, SpanMode
from clinval.schemas.prediction import Code, Span


def sp(start: int, end: int, text: str | None = None) -> Span:
    return Span(start=start, end=end, text=text)


class TestJaccard:
    def test_identical(self):
        assert jaccard_overlap(sp(0, 10), sp(0, 10)) == 1.0

    def test_disjoint(self):
        assert jaccard_overlap(sp(0, 10), sp(20, 30)) == 0.0

    def test_adjacent_is_disjoint(self):
        assert jaccard_overlap(sp(0, 10), sp(10, 20)) == 0.0

    def test_half_overlap(self):
        # intersection 5, union 15
        assert jaccard_overlap(sp(0, 10), sp(5, 15)) == pytest.approx(5 / 15)

    def test_nested(self):
        assert jaccard_overlap(sp(0, 20), sp(5, 10)) == pytest.approx(5 / 20)

    def test_zero_length_spans_report_zero_not_an_error(self):
        assert jaccard_overlap(sp(5, 5), sp(5, 5)) == 0.0

    def test_symmetric(self):
        a, b = sp(3, 17), sp(10, 25)
        assert jaccard_overlap(a, b) == jaccard_overlap(b, a)


class TestOverlapFraction:
    def test_relative_to_min(self):
        assert overlap_fraction(sp(0, 20), sp(5, 10), relative_to="min") == 1.0

    def test_relative_to_max(self):
        assert overlap_fraction(sp(0, 20), sp(5, 10), relative_to="max") == pytest.approx(0.25)

    def test_relative_to_a_and_b(self):
        assert overlap_fraction(sp(0, 10), sp(5, 15), relative_to="a") == pytest.approx(0.5)
        assert overlap_fraction(sp(0, 10), sp(5, 15), relative_to="b") == pytest.approx(0.5)

    def test_zero_length_denominator(self):
        assert overlap_fraction(sp(5, 5), sp(0, 10), relative_to="a") == 0.0

    def test_bad_argument(self):
        with pytest.raises(ValueError, match="relative_to must be"):
            overlap_fraction(sp(0, 1), sp(0, 1), relative_to="sideways")


class TestFuzzyText:
    def test_word_order_ignored(self):
        assert (
            fuzzy_text_score(sp(0, 1, "loose watery stools"), sp(0, 1, "watery stools loose"))
            == 1.0
        )

    def test_unrelated_text_scores_low(self):
        assert fuzzy_text_score(sp(0, 1, "diarrhea"), sp(0, 1, "fractured femur")) < 0.5

    def test_unicode(self):
        assert fuzzy_text_score(sp(0, 1, "diarrhée sévère"), sp(0, 1, "diarrhée sévère")) == 1.0

    def test_case_is_folded(self):
        """A sentence-initial capital is agreement, not a near miss."""
        assert (
            fuzzy_text_score(sp(0, 1, "Diarrhea since Monday"), sp(0, 1, "diarrhea since Monday"))
            == 1.0
        )

    def test_accents_are_not_folded(self):
        """Case folding is safe; accent folding could merge distinct terms."""
        assert fuzzy_text_score(sp(0, 1, "diarrhée"), sp(0, 1, "diarrhee")) < 1.0

    def test_missing_text_is_none(self):
        assert fuzzy_text_score(sp(0, 1, None), sp(0, 1, "x")) is None


class TestMatchSpans:
    EXACT = SpanMatching(mode=SpanMode.EXACT)
    OVERLAP = SpanMatching(mode=SpanMode.OVERLAP, min_overlap=0.5)
    FUZZY = SpanMatching(mode=SpanMode.FUZZY, fuzzy_threshold=0.85)

    def test_exact_match(self):
        r = match_spans(sp(0, 10), sp(0, 10), self.EXACT)
        assert r.matched
        assert r.reason == "exact_offsets"

    def test_exact_rejects_one_character_drift(self):
        r = match_spans(sp(0, 10), sp(0, 11), self.EXACT)
        assert not r.matched
        assert r.reason == "offset_mismatch"
        # jaccard is still reported, so the caller can see how close it was
        assert r.jaccard > 0.9

    def test_overlap_above_threshold(self):
        r = match_spans(sp(0, 10), sp(1, 10), self.OVERLAP)
        assert r.matched

    def test_overlap_below_threshold(self):
        r = match_spans(sp(0, 10), sp(8, 30), self.OVERLAP)
        assert not r.matched
        assert "<" in r.reason

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_threshold_is_inclusive(self, threshold):
        policy = SpanMatching(mode=SpanMode.OVERLAP, min_overlap=threshold)
        r = match_spans(sp(0, 10), sp(0, 10), policy)
        assert r.matched  # jaccard 1.0 >= any threshold

    def test_fuzzy_uses_text_when_offsets_drift(self):
        r = match_spans(
            sp(100, 120, "reports loose stools"), sp(0, 20, "reports loose stools"), self.FUZZY
        )
        assert r.matched
        assert r.jaccard == 0.0  # offsets disjoint, but the text agrees
        assert r.fuzzy_score == 1.0

    def test_fuzzy_falls_back_to_overlap_without_text(self):
        r = match_spans(sp(0, 10), sp(0, 10), self.FUZZY)
        assert r.matched
        assert "fell_back_to_overlap" in r.reason

    def test_fuzzy_below_threshold(self):
        r = match_spans(sp(0, 10, "diarrhea"), sp(0, 10, "femur fracture"), self.FUZZY)
        assert not r.matched

    def test_missing_spans_are_not_failures_by_default(self):
        assert match_spans(None, None, self.OVERLAP).matched
        assert match_spans(None, sp(0, 5), self.OVERLAP).matched
        assert match_spans(sp(0, 5), None, self.OVERLAP).matched

    def test_require_span_makes_absence_a_failure(self):
        policy = SpanMatching(mode=SpanMode.OVERLAP, require_span=True)
        assert not match_spans(None, sp(0, 5), policy).matched
        assert not match_spans(sp(0, 5), None, policy).matched
        assert not match_spans(None, None, policy).matched

    def test_absence_reason_names_the_missing_side(self):
        assert match_spans(None, sp(0, 5), self.OVERLAP).reason == "pred_span_absent"
        assert match_spans(sp(0, 5), None, self.OVERLAP).reason == "gold_span_absent"
        assert match_spans(None, None, self.OVERLAP).reason == "both_spans_absent"

    def test_zero_length_never_matches_under_overlap(self):
        assert not match_spans(sp(5, 5), sp(5, 5), self.OVERLAP).matched


class TestNormalizeCode:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HP:0002014", "HP:0002014"),
            ("hp:2014", "HP:0002014"),
            ("HP_0002014", "HP:0002014"),
            ("HP:2014", "HP:0002014"),
        ],
    )
    def test_hpo_forms(self, raw, expected):
        assert normalize_code(raw, "HPO") == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("E11.9", "E11.9"),
            ("e119", "E11.9"),
            ("E11", "E11"),
            ("e11", "E11"),
            ("E 11.9", "E11.9"),
        ],
    )
    def test_icd10_forms(self, raw, expected):
        assert normalize_code(raw, "ICD-10") == expected

    def test_decimal_point_retained_so_levels_stay_distinct(self):
        """Dropping the point would conflate E1.19 with E11.9."""
        assert normalize_code("E11.9", "ICD-10") != normalize_code("E119", "ICD-10").replace(
            ".", ""
        )

    def test_empty(self):
        assert normalize_code("") == ""
        assert normalize_code("   ") == ""

    def test_unknown_system_uppercased(self):
        assert normalize_code("abc123", "SNOMED-CT") == "ABC123"


class TestIcd10Prefix:
    def test_category_level(self):
        assert icd10_prefix("E11.9", 3) == "E11"

    def test_subcategory_level(self):
        assert icd10_prefix("E11.9", 4) == "E11.9"

    def test_depth_beyond_the_code(self):
        assert icd10_prefix("E11", 6) == "E11"


class TestOntologyMap:
    def test_bundled_map_normalized(self):
        m = bundled_map()
        assert "HP:0002014" in m.concepts
        assert m.display("HP:0002014") == "Diarrhea"

    def test_synonyms_include_the_display_term(self):
        syns = bundled_map().synonyms("HP:0002014")
        assert "Diarrhea" in syns
        assert "loose stools" in syns

    def test_crosswalk_both_directions(self):
        m = bundled_map()
        assert "K59.1" in m.crosswalk("HP:0002014")
        assert "HP:0002014" in m.crosswalk("K59.1")

    def test_crosswalk_excludes_self(self):
        assert "HP:0002014" not in bundled_map().crosswalk("HP:0002014")

    def test_lookup_by_text(self):
        hit = bundled_map().lookup_by_text("watery stools")
        assert hit is not None
        assert hit[0] == "HP:0002014"

    def test_lookup_below_threshold_returns_none(self):
        assert bundled_map().lookup_by_text("fractured femur", threshold=0.9) is None

    def test_lookup_empty_text(self):
        assert bundled_map().lookup_by_text("") is None

    def test_unknown_code_has_no_display(self):
        assert bundled_map().display("HP:9999999") is None

    def test_load_from_json(self, tmp_path):
        import json

        p = tmp_path / "map.json"
        p.write_text(json.dumps({"HP:0000001": {"display": "Thing", "synonyms": ["widget"]}}))
        m = load_ontology_map(p)
        assert m.display("HP:0000001") == "Thing"

    def test_load_rejects_non_object(self, tmp_path):
        import json

        p = tmp_path / "map.json"
        p.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="keyed by code"):
            load_ontology_map(p)

    def test_bundled_map_is_documented_as_demo_scope(self):
        """A guard against the map quietly growing into a pretend terminology."""
        assert len(BUNDLED_MAP) <= 20


class TestMatchCodes:
    def test_identical(self):
        r = match_codes(
            Code(system="HPO", code="HP:0002014"), Code(system="HPO", code="HP:0002014")
        )
        assert r.matched
        assert r.how == "identical"

    def test_normalization_makes_forms_equal(self):
        r = match_codes(Code(system="HPO", code="hp:2014"), Code(system="HPO", code="HP:0002014"))
        assert r.matched
        assert r.how == "identical"

    def test_crosswalk_across_terminologies(self):
        r = match_codes(Code(system="ICD-10", code="K59.1"), Code(system="HPO", code="HP:0002014"))
        assert r.matched
        assert r.how == "crosswalk"

    def test_mismatch(self):
        r = match_codes(
            Code(system="HPO", code="HP:0002014"), Code(system="HPO", code="HP:0002019")
        )
        assert not r.matched
        assert r.how == "mismatch"

    def test_prefix_matching_when_enabled(self):
        policy = OntologyMatching(prefix_depth=3)
        r = match_codes(
            Code(system="ICD-10", code="E11.9"), Code(system="ICD-10", code="E11.65"), policy
        )
        assert r.matched
        assert r.how == "prefix"

    def test_prefix_not_applied_across_systems(self):
        policy = OntologyMatching(prefix_depth=3)
        r = match_codes(
            Code(system="ICD-10", code="E11.9"), Code(system="ICD-9", code="E11.65"), policy
        )
        assert not r.matched

    def test_fuzzy_display_matching(self):
        policy = OntologyMatching(fuzzy=True, threshold=0.8)
        r = match_codes(
            Code(system="HPO", code="HP:1111111", display="Loose stools diarrhea"),
            Code(system="HPO", code="HP:2222222", display="Diarrhea loose stools"),
            policy,
        )
        assert r.matched
        assert r.how == "fuzzy_display"

    def test_fuzzy_below_threshold_is_a_mismatch_with_a_score(self):
        policy = OntologyMatching(fuzzy=True, threshold=0.95)
        r = match_codes(
            Code(system="HPO", code="HP:1111111", display="Diarrhea"),
            Code(system="HPO", code="HP:2222222", display="Femur fracture"),
            policy,
        )
        assert not r.matched
        assert r.score < 0.95

    def test_absent_code_is_not_a_failure(self):
        r = match_codes(None, Code(system="HPO", code="HP:0002014"))
        assert r.matched
        assert r.how == "absent"

    def test_disabled_policy_short_circuits(self):
        r = match_codes(
            Code(system="HPO", code="HP:0002014"),
            Code(system="HPO", code="HP:0002019"),
            OntologyMatching(enabled=False),
        )
        assert r.matched
        assert r.how == "disabled"
