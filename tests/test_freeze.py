"""The frozen-dev protocol: guards, hashing, and tamper detection."""

from __future__ import annotations

import json

import pytest

from clinval.exceptions import ConfigError, FrozenConfigError, SplitGuardError
from clinval.freeze import (
    check_split_disjoint,
    evaluate_guard,
    freeze,
    git_commit,
    load_frozen,
    tune,
    write_frozen,
)
from clinval.schemas.config import EvalConfig, Split
from clinval.synth import generate
from tests.conftest import make_gold, make_pred

DEV_CFG = {
    "split": "dev",
    "unit": "patient_window",
    "window": {"size": "6M", "n_windows": 2},
    "bootstrap": {"n": 20, "seed": 0},
}


@pytest.fixture(scope="module")
def data():
    return generate(n_patients=24, labels=("diarrhea", "constipation"), seed=5)


@pytest.fixture(scope="module")
def splits(data):
    return data.split_by_patient(0.5)


class TestHashing:
    def test_identical_configs_hash_identically(self):
        a = EvalConfig.from_dict(DEV_CFG)
        b = EvalConfig.from_dict(DEV_CFG)
        assert a.content_hash() == b.content_hash()

    def test_key_order_does_not_change_the_hash(self):
        a = EvalConfig.from_dict({"split": "dev", "unit": "note"})
        b = EvalConfig.from_dict({"unit": "note", "split": "dev"})
        assert a.content_hash() == b.content_hash()

    def test_duration_shorthand_hashes_like_its_object_form(self):
        """`6M` and Duration(n=6, unit='M') must not look like different configs."""
        a = EvalConfig.from_dict({"window": {"size": "6M"}})
        b = EvalConfig.from_dict({"window": {"size": {"n": 6, "unit": "M"}}})
        assert a.content_hash() == b.content_hash()

    def test_content_change_changes_the_hash(self):
        """SPEC §8.2: the hash changes if and only if content changes."""
        a = EvalConfig.from_dict(DEV_CFG)
        b = a.with_overrides(min_overlap=0.9)
        assert a.content_hash() != b.content_hash()

    def test_hash_is_hex_sha256(self):
        h = EvalConfig().content_hash()
        assert len(h) == 64
        int(h, 16)


class TestTune:
    def test_refuses_to_tune_on_test_data(self, splits):
        """The failure the protocol exists to prevent."""
        dev, _ = splits
        cfg = EvalConfig.from_dict({**DEV_CFG, "split": "test"})
        with pytest.raises(SplitGuardError, match="refusing to tune"):
            tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)

    def test_escape_hatch_exists_for_tests_only(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict({**DEV_CFG, "split": "test"})
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index, allow_test=True)
        assert result.config.split is Split.DEV

    def test_no_grid_evaluates_one_candidate(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(DEV_CFG)
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        assert result.n_evaluated == 1
        assert result.best_params == {}

    def test_min_overlap_grid_is_explored(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(
            {
                **DEV_CFG,
                "matching": {"span": {"mode": "overlap", "require_span": True}},
                "freeze": {"objective": "cohen_kappa", "grid": {"min_overlap": [0.2, 0.5, 0.9]}},
            }
        )
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        assert result.n_evaluated == 3
        assert result.best_params["min_overlap"] in (0.2, 0.5, 0.9)
        assert result.config.matching.span.min_overlap == result.best_params["min_overlap"]

    def test_min_overlap_is_flat_without_require_span(self, splits):
        """Documented interaction: span policy only moves the boolean metrics
        when an unsupported positive is actually withheld credit."""
        dev, _ = splits
        cfg = EvalConfig.from_dict(
            {**DEV_CFG, "freeze": {"objective": "cohen_kappa", "grid": {"min_overlap": [0.2, 0.9]}}}
        )
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        assert result.is_flat

    def test_consensus_grid_scores_the_gate(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(
            {
                **DEV_CFG,
                "freeze": {
                    "objective": "cohen_kappa",
                    "grid": {"consensus": ["majority", "unanimous", "any"]},
                },
            }
        )
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        assert result.n_evaluated == 3
        assert all("consensus rule" in c["scored_over"] for c in result.candidates)
        assert all("decided_fraction" in c for c in result.candidates)

    def test_consensus_grid_needs_two_systems(self, splits):
        dev, _ = splits
        one_system = [p for p in dev.all_predictions if p.system == "careful_reader"]
        cfg = EvalConfig.from_dict(
            {
                **DEV_CFG,
                "freeze": {"objective": "cohen_kappa", "grid": {"consensus": ["majority", "any"]}},
            }
        )
        with pytest.raises(ConfigError, match="at least two systems"):
            tune(cfg, one_system, dev.annotations, index=dev.index)

    def test_deferral_warning_when_the_winner_answers_less(self, splits):
        """A rule that defers more is scored on fewer units; say so."""
        dev, _ = splits
        cfg = EvalConfig.from_dict(
            {
                **DEV_CFG,
                "freeze": {
                    "objective": "cohen_kappa",
                    "grid": {"consensus": ["majority", "unanimous", "any"]},
                },
            }
        )
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        if result.best_params.get("consensus") == "majority":
            assert result.warnings
            assert "decided only" in result.warnings[0]

    def test_tune_on_one_named_system(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(DEV_CFG)
        result = tune(
            cfg, dev.all_predictions, dev.annotations, index=dev.index, system="careful_reader"
        )
        assert "careful_reader" in result.note

    def test_unknown_system_rejected(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(DEV_CFG)
        with pytest.raises(ConfigError, match="is not present"):
            tune(cfg, dev.all_predictions, dev.annotations, index=dev.index, system="nope")

    def test_no_overlap_between_pred_and_gold_is_an_informative_error(self):
        cfg = EvalConfig.from_dict({"split": "dev", "unit": "note"})
        with pytest.raises(ConfigError, match="no usable score"):
            tune(cfg, [make_pred("N1")], [make_gold("N2")])

    def test_untunable_parameter_rejected_at_config_load(self):
        with pytest.raises(ConfigError, match="is not tunable"):
            EvalConfig.from_dict({"freeze": {"grid": {"bootstrap_n": [1, 2]}}})

    def test_unknown_objective_rejected_at_config_load(self):
        with pytest.raises(ConfigError, match="unknown tuning objective"):
            EvalConfig.from_dict({"freeze": {"objective": "auroc"}})

    def test_markdown_shows_the_whole_grid(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(
            {
                **DEV_CFG,
                "matching": {"span": {"mode": "overlap", "require_span": True}},
                "freeze": {"objective": "cohen_kappa", "grid": {"min_overlap": [0.3, 0.7]}},
            }
        )
        md = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index).to_markdown()
        assert "0.3" in md
        assert "0.7" in md
        assert "chosen" in md


class TestFreezeAndLoad:
    def test_freeze_forces_split_to_test(self):
        frozen = freeze(EvalConfig.from_dict(DEV_CFG))
        assert frozen.config.split is Split.TEST

    def test_hash_covers_the_config_only(self):
        """Re-freezing an identical config is idempotent in hash terms."""
        cfg = EvalConfig.from_dict(DEV_CFG)
        a, b = freeze(cfg), freeze(cfg)
        # The hash covers the config body only, so the timestamps are free to
        # differ (or to tie) without affecting verification.
        assert a.config_hash == b.config_hash
        assert a.config.content_hash() == b.config.content_hash()

    def test_roundtrip(self, tmp_path):
        frozen = freeze(EvalConfig.from_dict(DEV_CFG))
        p = write_frozen(frozen, tmp_path / "frozen.json")
        back = load_frozen(p)
        assert back.config_hash == frozen.config_hash
        assert back.config.content_hash() == frozen.config_hash

    def test_tamper_detected(self, tmp_path):
        """SPEC §8.1: modify frozen.json, and evaluate must refuse."""
        frozen = freeze(EvalConfig.from_dict(DEV_CFG))
        p = tmp_path / "frozen.json"
        write_frozen(frozen, p)

        raw = json.loads(p.read_text())
        raw["config"]["matching"]["span"]["min_overlap"] = 0.99
        p.write_text(json.dumps(raw))

        with pytest.raises(FrozenConfigError, match="modified after freezing"):
            load_frozen(p)

    def test_tampering_with_the_hash_is_also_caught(self, tmp_path):
        frozen = freeze(EvalConfig.from_dict(DEV_CFG))
        p = tmp_path / "frozen.json"
        write_frozen(frozen, p)
        raw = json.loads(p.read_text())
        raw["config_hash"] = "0" * 64
        p.write_text(json.dumps(raw))
        with pytest.raises(FrozenConfigError, match="failed verification"):
            load_frozen(p)

    def test_not_a_manifest(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text(json.dumps({"hello": "world"}))
        with pytest.raises(FrozenConfigError, match="not a clinval freeze manifest"):
            load_frozen(p)

    def test_malformed_manifest(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text(json.dumps({"config_hash": "abc", "config": {"unit": "banana"}}))
        with pytest.raises(FrozenConfigError, match="malformed freeze manifest"):
            load_frozen(p)

    def test_tuning_summary_is_carried(self, splits):
        dev, _ = splits
        cfg = EvalConfig.from_dict(
            {
                **DEV_CFG,
                "matching": {"span": {"mode": "overlap", "require_span": True}},
                "freeze": {"objective": "cohen_kappa", "grid": {"min_overlap": [0.3, 0.7]}},
            }
        )
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        frozen = freeze(result.config, tuning=result)
        assert frozen.tuning_summary is not None
        assert frozen.tuning_summary["objective"] == "cohen_kappa"
        assert len(frozen.tuning_summary["candidates"]) == 2

    def test_version_recorded(self):
        from clinval import __version__

        assert freeze(EvalConfig()).clinval_version == __version__


class TestEvaluateGuard:
    def test_verified_frozen_config_is_allowed(self):
        evaluate_guard(EvalConfig(), frozen=freeze(EvalConfig.from_dict(DEV_CFG)))

    def test_unfrozen_test_config_refused(self):
        cfg = EvalConfig.from_dict({**DEV_CFG, "split": "test"})
        with pytest.raises(SplitGuardError, match="without a frozen config"):
            evaluate_guard(cfg, frozen=None)

    def test_refusal_message_shows_the_protocol(self):
        cfg = EvalConfig.from_dict({**DEV_CFG, "split": "test"})
        with pytest.raises(SplitGuardError) as exc:
            evaluate_guard(cfg, frozen=None)
        msg = str(exc.value)
        assert "clinval tune" in msg
        assert "clinval freeze" in msg
        assert "clinval evaluate" in msg

    def test_unfrozen_dev_config_needs_acknowledgement(self):
        with pytest.raises(SplitGuardError, match="in-sample"):
            evaluate_guard(EvalConfig.from_dict(DEV_CFG), frozen=None)

    def test_acknowledged_dev_config_is_allowed(self):
        evaluate_guard(EvalConfig.from_dict(DEV_CFG), frozen=None, acknowledge_dev=True)

    def test_tampered_frozen_config_refused_at_the_guard(self):
        frozen = freeze(EvalConfig.from_dict(DEV_CFG))
        frozen.config_hash = "0" * 64
        with pytest.raises(FrozenConfigError):
            evaluate_guard(EvalConfig(), frozen=frozen)


class TestCheckSplitDisjoint:
    def test_disjoint_splits(self, splits):
        dev, test = splits
        out = check_split_disjoint(dev.annotations, test.annotations)
        assert out["ok"] is True
        assert out["n_shared_patients"] == 0
        assert out["n_shared_notes"] == 0

    def test_overlap_detected(self, data):
        out = check_split_disjoint(data.annotations, data.annotations)
        assert out["ok"] is False
        assert out["n_shared_patients"] > 0
        assert out["n_shared_notes"] > 0

    def test_shared_ids_are_truncated_for_reporting(self, data):
        out = check_split_disjoint(data.annotations, data.annotations)
        assert len(out["shared_patients"]) <= 50
        assert len(out["shared_notes"]) <= 50

    def test_notes_detect_overlap_even_without_patient_ids(self):
        a = [make_gold("N1", patient_id=None)]
        b = [make_gold("N1", patient_id=None)]
        out = check_split_disjoint(a, b)
        assert out["ok"] is False
        assert out["patient_ids_available"] is False


class TestGitCommit:
    def test_returns_a_hash_or_none(self, tmp_path):
        assert git_commit(tmp_path) is None or isinstance(git_commit(tmp_path), str)

    def test_non_repo_is_not_an_error(self, tmp_path):
        assert git_commit(tmp_path) is None
