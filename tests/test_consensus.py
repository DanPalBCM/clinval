"""Consensus gating: rules, ties, and whether the gate is worth anything."""

from __future__ import annotations

import pytest

from clinval.consensus import apply_rule, consensus_gate
from clinval.exceptions import ConfigError
from clinval.schemas.config import Bootstrap, EvalConfig, Unit
from tests.conftest import make_gold, make_pred

CFG = EvalConfig(unit=Unit.NOTE, bootstrap=Bootstrap(n=30, seed=0))


class TestApplyRule:
    @pytest.mark.parametrize(
        ("n_pos", "n_sys", "expected"),
        [
            (0, 2, False),
            (1, 2, None),
            (2, 2, True),
            (0, 3, False),
            (1, 3, False),
            (2, 3, True),
            (3, 3, True),
        ],
    )
    def test_majority(self, n_pos, n_sys, expected):
        value, _ = apply_rule(n_pos, n_sys, "majority")
        assert value == expected

    def test_majority_even_split_is_undecided_not_negative(self):
        """Scoring a 1-1 tie as negative is the easiest way to fake a good gate."""
        value, review = apply_rule(1, 2, "majority")
        assert value is None
        assert review is True

    @pytest.mark.parametrize(
        ("n_pos", "n_sys", "expected"),
        [(0, 2, False), (1, 2, False), (2, 2, True), (2, 3, False), (3, 3, True)],
    )
    def test_unanimous(self, n_pos, n_sys, expected):
        assert apply_rule(n_pos, n_sys, "unanimous")[0] == expected

    @pytest.mark.parametrize(
        ("n_pos", "n_sys", "expected"),
        [(0, 2, False), (1, 2, True), (2, 2, True), (1, 3, True)],
    )
    def test_any(self, n_pos, n_sys, expected):
        assert apply_rule(n_pos, n_sys, "any")[0] == expected

    @pytest.mark.parametrize("rule", ["majority", "unanimous", "any"])
    def test_unanimous_votes_never_need_review(self, rule):
        assert apply_rule(0, 3, rule)[1] is False
        assert apply_rule(3, 3, rule)[1] is False

    @pytest.mark.parametrize("rule", ["majority", "unanimous", "any"])
    def test_split_votes_always_need_review(self, rule):
        """A 2-1 majority still has a dissenter worth a human's attention."""
        assert apply_rule(2, 3, rule)[1] is True

    def test_unknown_rule(self):
        with pytest.raises(ConfigError, match="unknown consensus rule"):
            apply_rule(1, 2, "vibes")  # type: ignore[arg-type]

    def test_no_systems(self):
        assert apply_rule(0, 0, "majority") == (None, True)


class TestConsensusGate:
    def _two_systems(self, a_vals: list[bool], b_vals: list[bool], gold_vals: list[bool]):
        notes = [f"N{i}" for i in range(len(gold_vals))]
        preds_a = [
            make_pred(n, value=v, system="sysA", patient_id=f"P{i}")
            for i, (n, v) in enumerate(zip(notes, a_vals, strict=True))
        ]
        preds_b = [
            make_pred(n, value=v, system="sysB", patient_id=f"P{i}")
            for i, (n, v) in enumerate(zip(notes, b_vals, strict=True))
        ]
        golds = [
            make_gold(n, value=v, patient_id=f"P{i}")
            for i, (n, v) in enumerate(zip(notes, gold_vals, strict=True))
        ]
        return preds_a, preds_b, golds

    def test_two_systems_majority(self):
        a, b, g = self._two_systems([True, True, False], [True, False, False], [True, True, False])
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        assert res.systems == ("sysA", "sysB")
        assert res.n_units == 3
        # the middle unit is a 1-1 split
        assert res.n_review == 1
        assert res.review_fraction == pytest.approx(1 / 3)

    def test_three_systems_majority_decides_everything(self):
        notes = [f"N{i}" for i in range(4)]
        vals = {
            "a": [True, True, False, False],
            "b": [True, False, True, False],
            "c": [True, False, False, False],
        }
        frames = {
            name: [
                make_pred(n, value=v, system=name, patient_id=f"P{i}")
                for i, (n, v) in enumerate(zip(notes, vs, strict=True))
            ]
            for name, vs in vals.items()
        }
        golds = [make_gold(n, value=True, patient_id=f"P{i}") for i, n in enumerate(notes)]
        res = consensus_gate(frames, golds, rule="majority", config=CFG)
        assert res.df["consensus"].notna().all()
        assert res.df["consensus"].tolist() == [True, False, False, False]

    def test_review_queue_sorted_by_contention(self):
        a, b, g = self._two_systems([True, True], [False, True], [True, True])
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        q = res.to_review_queue()
        assert len(q) == 1
        assert "split_margin" in q.columns

    def test_agreed_vs_disagreed_is_reported(self):
        a, b, g = self._two_systems(
            [True, True, False, False, True],
            [True, True, False, False, False],
            [True, True, False, False, True],
        )
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        avd = res.agreed_vs_disagreed
        assert avd["agreed"]["n"] == 4
        assert avd["agreed"]["accuracy"] == pytest.approx(1.0)
        assert "accuracy_lift" in avd

    def test_undecided_units_are_excluded_and_counted(self):
        """The gate must say how many questions it declined to answer."""
        a, b, g = self._two_systems([True, True], [False, True], [True, True])
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        assert res.agreed_vs_disagreed["disagreed"]["undecided_excluded"] == 1

    def test_pairwise_agreement_is_gold_free(self):
        a, b, g = self._two_systems([True, False], [True, False], [False, True])
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        pair = res.pairwise["sysA|sysB"]
        # the systems agree perfectly with each other while both being wrong
        assert pair["observed_agreement"] == pytest.approx(1.0)
        assert res.agreement is not None
        assert res.agreement.confusion.tp == 0

    def test_single_system_refused(self):
        a, _, g = self._two_systems([True], [True], [True])
        with pytest.raises(ConfigError, match="at least 2 systems"):
            consensus_gate({"sysA": a}, g, config=CFG)

    def test_raw_predictions_without_gold_refused(self):
        a, b, _ = self._two_systems([True], [True], [True])
        with pytest.raises(ConfigError, match="gold annotations are needed"):
            consensus_gate({"sysA": a, "sysB": b}, None, config=CFG)

    def test_markdown_renders(self):
        a, b, g = self._two_systems([True, False], [True, True], [True, False])
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        md = res.to_markdown()
        assert "Consensus gate" in md
        assert "routed to review" in md

    def test_as_dict_is_json_safe(self):
        import json

        a, b, g = self._two_systems([True, False], [True, True], [True, False])
        res = consensus_gate({"sysA": a, "sysB": b}, g, rule="majority", config=CFG)
        json.dumps(res.as_dict())

    def test_accepts_windowed_frames(self):
        from clinval.aggregate import aggregate_to_windows, align_predictions

        a, b, g = self._two_systems([True, False], [True, True], [True, False])
        frames = {}
        for name, preds in (("sysA", a), ("sysB", b)):
            aligned = align_predictions(preds, g, CFG)
            frames[name] = aggregate_to_windows(aligned, unit=Unit.NOTE)
        res = consensus_gate(frames, g, rule="unanimous", config=CFG)
        assert res.n_units == 2

    def test_any_rule_maximizes_sensitivity_at_the_cost_of_specificity(self):
        a, b, g = self._two_systems(
            [True, False, False], [False, True, False], [True, False, False]
        )
        res_any = consensus_gate({"sysA": a, "sysB": b}, g, rule="any", config=CFG)
        res_unan = consensus_gate({"sysA": a, "sysB": b}, g, rule="unanimous", config=CFG)
        assert res_any.agreement is not None
        assert res_unan.agreement is not None
        # `any` catches the true positive but adds a false one
        assert res_any.agreement.confusion.tp == 1
        assert res_any.agreement.confusion.fp == 1
        assert res_unan.agreement.confusion.tp == 0
        assert res_unan.agreement.confusion.fp == 0
