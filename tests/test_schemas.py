"""Schemas: record validation, config loading, shorthands, and overrides."""

from __future__ import annotations

import pytest
import yaml

from clinval.exceptions import ConfigError
from clinval.schemas.config import (
    KNOWN_METRICS,
    Duration,
    EvalConfig,
    Matching,
    SpanMatching,
    SpanMode,
    Split,
    TieBreak,
    Unit,
    Window,
)
from clinval.schemas.prediction import Prediction, Span


class TestDuration:
    @pytest.mark.parametrize(
        ("text", "n", "unit"),
        [("6M", 6, "M"), ("30D", 30, "D"), ("2W", 2, "W"), ("1Y", 1, "Y"), (" 12 m ", 12, "M")],
    )
    def test_parse(self, text, n, unit):
        d = Duration.parse(text)
        assert (d.n, d.unit) == (n, unit)

    def test_parse_is_idempotent(self):
        d = Duration(n=6, unit="M")
        assert Duration.parse(d) is d

    def test_str_roundtrip(self):
        assert str(Duration.parse("6M")) == "6M"

    @pytest.mark.parametrize("bad", ["6", "M", "6Q", "six months", "", "-3M", "0M"])
    def test_invalid(self, bad):
        with pytest.raises(ConfigError, match="cannot parse duration"):
            Duration.parse(bad)

    def test_zero_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 - pydantic or ConfigError
            Duration(n=0, unit="M")


class TestSpanAndPrediction:
    def test_span_length(self):
        assert Span(start=5, end=12).length == 7

    def test_zero_length_allowed(self):
        assert Span(start=5, end=5).length == 0

    def test_inverted_rejected(self):
        with pytest.raises(ValueError, match="precedes start"):
            Span(start=9, end=4)

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            Span(start=-1, end=4)

    def test_prediction_key(self):
        p = Prediction(system="s", note_id="N1", label="l", value=True)
        assert p.key == ("s", "N1", "l")

    def test_whitespace_stripped_from_identifiers(self):
        p = Prediction(system="  s  ", note_id=" N1 ", label=" l ", value=True)
        assert p.key == ("s", "N1", "l")

    def test_empty_identifier_rejected(self):
        with pytest.raises(ValueError):
            Prediction(system="", note_id="N1", label="l", value=True)

    def test_label_is_opaque(self):
        """fhirground emits `Condition:<code>`; clinval must not interpret it."""
        p = Prediction(system="s", note_id="N1", label="Condition:E11.9", value=True)
        assert p.label == "Condition:E11.9"

    def test_run_meta_allows_extra_keys(self):
        """Adapters attach their own provenance; it must not be dropped."""
        p = Prediction(
            system="s",
            note_id="N1",
            label="l",
            value=True,
            run={"model": "m", "alignment_status": "MATCH_FUZZY"},  # type: ignore[arg-type]
        )
        assert p.run is not None
        assert p.run.model_dump()["alignment_status"] == "MATCH_FUZZY"


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.split is Split.DEV
        assert cfg.unit is Unit.PATIENT_WINDOW
        assert cfg.window.size.n == 6
        assert cfg.bootstrap.cluster_by == "patient_id"
        assert set(cfg.metrics) == set(KNOWN_METRICS)

    def test_spec_yaml_example_loads(self, tmp_path):
        """The config in SPEC §5.3, verbatim."""
        text = """
split: dev
unit: patient_window
labels: [diarrhea, constipation, vomiting]
matching:
  span: {mode: overlap, min_overlap: 0.5}
  ontology:
    hpo: {fuzzy: true, threshold: 0.85}
window:
  size: 6M
  anchor: index_date
  tie_break: earliest
metrics: [cohen_kappa, sensitivity, specificity, ppv, npv]
bootstrap: {n: 2000, cluster_by: patient_id, seed: 0}
cost:
  input_per_1k: 0.15
  output_per_1k: 0.60
freeze:
  grid:
    min_overlap: [0.3, 0.5, 0.7]
    consensus: [majority, unanimous]
"""
        p = tmp_path / "eval.yaml"
        p.write_text(text)
        cfg = EvalConfig.from_yaml(p)
        assert cfg.labels == ["diarrhea", "constipation", "vomiting"]
        assert cfg.matching.span.mode is SpanMode.OVERLAP
        assert cfg.matching.span.min_overlap == 0.5
        assert cfg.matching.ontology["HPO"].fuzzy is True
        assert cfg.matching.ontology["HPO"].threshold == 0.85
        assert cfg.window.tie_break is TieBreak.EARLIEST
        assert cfg.bootstrap.n == 2000
        assert cfg.cost.output_per_1k == 0.60
        assert cfg.freeze.grid["min_overlap"] == [0.3, 0.5, 0.7]

    def test_spec_python_shorthand(self):
        """The call in SPEC §5.2, verbatim."""
        cfg = EvalConfig(
            unit="patient_window",  # type: ignore[arg-type]
            window=Window(size="6M", anchor="index_date"),  # type: ignore[arg-type]
            matching=Matching(span="overlap", min_overlap=0.5),  # type: ignore[call-arg]
        )
        assert cfg.matching.span.mode is SpanMode.OVERLAP
        assert cfg.matching.span.min_overlap == 0.5
        assert cfg.window.size.n == 6

    def test_matching_accepts_nested_form_too(self):
        cfg = Matching(span=SpanMatching(mode="fuzzy", fuzzy_threshold=0.9))
        assert cfg.span.mode is SpanMode.FUZZY
        assert cfg.span.fuzzy_threshold == 0.9

    def test_ontology_keys_are_upper_cased(self):
        cfg = EvalConfig.from_dict({"matching": {"ontology": {"icd-10": {"prefix_depth": 3}}}})
        assert "ICD-10" in cfg.matching.ontology

    def test_ontology_bool_shorthand(self):
        cfg = EvalConfig.from_dict({"matching": {"ontology": {"hpo": False}}})
        assert cfg.matching.ontology["HPO"].enabled is False

    def test_unknown_metric_rejected(self):
        with pytest.raises(ConfigError, match="unknown metric"):
            EvalConfig.from_dict({"metrics": ["auroc"]})

    def test_unknown_key_rejected(self):
        with pytest.raises(ConfigError):
            EvalConfig.from_dict({"untis": "note"})

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            EvalConfig.from_yaml(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("unit: [unclosed\n")
        with pytest.raises(ConfigError, match="invalid YAML"):
            EvalConfig.from_yaml(p)

    def test_non_mapping_root(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            EvalConfig.from_yaml(p)

    def test_empty_yaml_is_defaults(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert EvalConfig.from_yaml(p).unit is Unit.PATIENT_WINDOW

    def test_yaml_roundtrip(self, tmp_path):
        cfg = EvalConfig.from_dict(
            {"unit": "note", "labels": ["a"], "window": {"size": "30D"}, "bootstrap": {"n": 7}}
        )
        p = tmp_path / "out.yaml"
        cfg.to_yaml(p)
        back = EvalConfig.from_yaml(p)
        assert back.content_hash() == cfg.content_hash()
        assert yaml.safe_load(p.read_text())["unit"] == "note"

    def test_note_unit_normalizes_window_count(self):
        cfg = EvalConfig.from_dict({"unit": "note", "window": {"n_windows": 4}})
        assert cfg.window.n_windows == 1


class TestWithOverrides:
    def test_min_overlap_routed_into_span(self):
        cfg = EvalConfig().with_overrides(min_overlap=0.75)
        assert cfg.matching.span.min_overlap == 0.75

    def test_span_mode(self):
        cfg = EvalConfig().with_overrides(span_mode="fuzzy")
        assert cfg.matching.span.mode is SpanMode.FUZZY

    def test_require_span(self):
        assert EvalConfig().with_overrides(require_span=True).matching.span.require_span is True

    def test_fuzzy_threshold(self):
        assert EvalConfig().with_overrides(fuzzy_threshold=0.5).matching.span.fuzzy_threshold == 0.5

    def test_consensus(self):
        assert EvalConfig().with_overrides(consensus="unanimous").consensus == "unanimous"

    def test_unknown_parameter_rejected(self):
        with pytest.raises(ConfigError, match="is not tunable"):
            EvalConfig().with_overrides(bootstrap_n=5)

    def test_original_unchanged(self):
        cfg = EvalConfig()
        before = cfg.content_hash()
        cfg.with_overrides(min_overlap=0.9)
        assert cfg.content_hash() == before
