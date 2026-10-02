"""Cost accounting. The one rule: missing tokens are unknown, never zero."""

from __future__ import annotations

import math

import pytest

from clinval.cost import summarize_cost, token_cost
from clinval.schemas.config import CostModel
from tests.conftest import make_pred

RATES = CostModel(input_per_1k=0.15, output_per_1k=0.60)


class TestTokenCost:
    def test_arithmetic(self):
        assert token_cost(1000, 1000, RATES) == pytest.approx(0.75)

    def test_partial_thousands(self):
        assert token_cost(500, 250, RATES) == pytest.approx(0.075 + 0.15)

    def test_zero(self):
        assert token_cost(0, 0, RATES) == 0.0

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            token_cost(-1, 0, RATES)


class TestSummarizeCost:
    def test_complete_metadata(self):
        preds = [
            make_pred("N1", tokens=(1000, 100), latency=1.0),
            make_pred("N2", tokens=(1000, 100), latency=2.0),
        ]
        rep = summarize_cost(preds, RATES)
        s = rep.per_system["sysA"]
        assert s.cost_status == "complete"
        assert s.tokens_in == 2000
        assert s.tokens_out == 200
        assert s.cost == pytest.approx(2 * (0.15 + 0.06))
        assert s.token_coverage == 1.0
        assert s.cost_note == ""

    def test_no_metadata_is_unknown_not_zero(self):
        """SPEC §8.1. Reporting zero would make the least-instrumented system
        look cheapest, which is exactly backwards."""
        rep = summarize_cost([make_pred("N1"), make_pred("N2")], RATES)
        s = rep.per_system["sysA"]
        assert s.cost is None
        assert s.cost_status == "unknown"
        assert s.cost_per_record is None
        assert "cannot be estimated" in s.cost_note

    def test_latency_without_tokens_still_unknown_cost(self):
        """The langextract case exactly: latency measurable, tokens not."""
        rep = summarize_cost([make_pred("N1", latency=1.5)], RATES)
        s = rep.per_system["sysA"]
        assert s.cost is None
        assert s.cost_status == "unknown"
        assert s.latency_mean_s == pytest.approx(1.5)
        assert "langextract" in s.cost_note

    def test_partial_metadata_is_a_lower_bound(self):
        preds = [make_pred("N1", tokens=(1000, 0)), make_pred("N2", latency=1.0)]
        rep = summarize_cost(preds, RATES)
        s = rep.per_system["sysA"]
        assert s.cost_status == "partial"
        assert s.cost == pytest.approx(0.15)
        assert "lower bound" in s.cost_note
        assert s.token_coverage == pytest.approx(0.5)

    def test_cost_per_record_uses_reporting_denominator(self):
        """Dividing a partial total by every record would understate unit cost."""
        preds = [make_pred("N1", tokens=(1000, 0)), make_pred("N2")]
        s = summarize_cost(preds, RATES).per_system["sysA"]
        assert s.cost_per_record == pytest.approx(0.15)

    def test_unset_rates_are_distinguished_from_unknown_tokens(self):
        preds = [make_pred("N1", tokens=(1000, 1000))]
        s = summarize_cost(preds, CostModel()).per_system["sysA"]
        assert s.cost == 0.0
        assert s.cost_status == "complete"
        assert "rates are unset" in s.cost_note

    def test_one_token_side_missing_does_not_count(self):
        """A record with only tokens_in cannot be priced; it is not partial credit."""
        p = make_pred("N1", tokens=(1000, 0))
        p.run.tokens_out = None  # type: ignore[union-attr]
        s = summarize_cost([p], RATES).per_system["sysA"]
        assert s.cost_status == "unknown"

    def test_latency_percentiles(self):
        preds = [make_pred(f"N{i}", latency=float(i)) for i in range(1, 11)]
        s = summarize_cost(preds, RATES).per_system["sysA"]
        assert s.latency_max_s == 10.0
        assert s.latency_p50_s == pytest.approx(5.5)
        assert s.latency_p95_s == pytest.approx(9.55)
        assert s.latency_total_s == pytest.approx(55.0)

    def test_single_latency(self):
        s = summarize_cost([make_pred("N1", latency=2.0)], RATES).per_system["sysA"]
        assert s.latency_p50_s == 2.0
        assert s.latency_p95_s == 2.0

    def test_models_and_prompt_versions_collected(self):
        s = summarize_cost([make_pred("N1", tokens=(10, 10))], RATES).per_system["sysA"]
        assert s.models == ("test-model",)
        assert s.prompt_versions == ("v1",)

    def test_per_system_separation(self):
        preds = [
            make_pred("N1", system="a", tokens=(1000, 0)),
            make_pred("N1", system="b"),
        ]
        rep = summarize_cost(preds, RATES)
        assert rep.per_system["a"].cost_status == "complete"
        assert rep.per_system["b"].cost_status == "unknown"
        assert rep.as_dict()["any_unknown"] is True

    def test_empty(self):
        rep = summarize_cost([], RATES)
        assert rep.per_system == {}
        assert "No run metadata" in rep.to_markdown()

    def test_markdown_prints_unknown_not_zero(self):
        md = summarize_cost([make_pred("N1")], RATES).to_markdown()
        assert "unknown" in md
        assert "0.0000" not in md

    def test_markdown_marks_partial_with_a_floor(self):
        preds = [make_pred("N1", tokens=(1000, 0)), make_pred("N2")]
        md = summarize_cost(preds, RATES).to_markdown()
        assert "≥" in md

    def test_as_dict_is_json_safe(self):
        import json

        json.dumps(summarize_cost([make_pred("N1")], RATES).as_dict())

    def test_token_coverage_nan_when_no_records(self):
        from clinval.cost import SystemCost

        assert math.isnan(SystemCost(system="x").token_coverage)


class TestCallIdDeduplication:
    """One model call usually answers several labels; it must be charged once."""

    def _records(self, call_id: str | None):
        out = []
        for label in ("diarrhea", "constipation", "vomiting"):
            p = make_pred("N1", label, True, tokens=(1000, 100), latency=2.0)
            p.run.call_id = call_id  # type: ignore[union-attr]
            out.append(p)
        return out

    def test_shared_call_id_is_counted_once(self):
        s = summarize_cost(self._records("N1"), RATES).per_system["sysA"]
        assert s.n_calls == 1
        assert s.n_records == 1
        assert s.tokens_in == 1000
        assert s.tokens_out == 100
        assert s.cost == pytest.approx(0.15 + 0.06)

    def test_without_call_id_each_record_is_charged(self):
        """The documented hazard: three records, three times the cost."""
        s = summarize_cost(self._records(None), RATES).per_system["sysA"]
        assert s.n_calls is None
        assert s.n_records == 3
        assert s.tokens_in == 3000
        assert s.cost == pytest.approx(3 * (0.15 + 0.06))

    def test_latency_is_not_inflated_by_shared_calls(self):
        s = summarize_cost(self._records("N1"), RATES).per_system["sysA"]
        assert s.latency_total_s == pytest.approx(2.0)
        assert s.n_with_latency == 1

    def test_distinct_calls_are_each_charged(self):
        recs = []
        for note in ("N1", "N2"):
            for label in ("diarrhea", "constipation"):
                p = make_pred(note, label, True, tokens=(500, 50), latency=1.0)
                p.run.call_id = note  # type: ignore[union-attr]
                recs.append(p)
        s = summarize_cost(recs, RATES).per_system["sysA"]
        assert s.n_calls == 2
        assert s.tokens_in == 1000

    def test_coverage_uses_the_call_denominator(self):
        recs = self._records("N1")
        extra = make_pred("N2", "diarrhea", True, latency=1.0)
        extra.run.call_id = "N2"  # type: ignore[union-attr]
        s = summarize_cost([*recs, extra], RATES).per_system["sysA"]
        assert s.n_calls == 2
        assert s.token_coverage == pytest.approx(0.5)
        assert s.cost_status == "partial"

    def test_generator_sets_call_id_per_note(self):
        """The synth generator must not double-count its own cost."""
        from clinval.synth import generate

        data = generate(n_patients=10, labels=("diarrhea", "constipation"), seed=0)
        s = summarize_cost(data.predictions["careful_reader"], RATES).per_system["careful_reader"]
        assert s.n_calls == len(data.notes)
        assert s.n_records == len(data.notes)
