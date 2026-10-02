"""End-to-end integration. SPEC §8.4.

Two scenarios:

1. The full ``tune → freeze → evaluate`` protocol on synthetic data, including
   that the frozen config beats a naive baseline and that the report contains
   every section.
2. fhirground interoperability: clinval accepts a sibling project's
   ``clinval_predictions.jsonl`` with no translation step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinval.freeze import check_split_disjoint, freeze, load_frozen, tune, write_frozen
from clinval.io import read_annotations, read_index_dates, read_predictions
from clinval.runner import run_eval
from clinval.schemas.config import EvalConfig, Split
from clinval.synth import generate

FIXTURES = Path(__file__).parent / "fixtures"

BASE_CFG = {
    "split": "dev",
    "unit": "patient_window",
    "labels": ["diarrhea", "constipation"],
    "matching": {"span": {"mode": "overlap", "min_overlap": 0.5, "require_span": True}},
    "window": {"size": "6M", "anchor": "index_date", "n_windows": 2},
    "metrics": [
        "cohen_kappa",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "accuracy",
        "prevalence",
    ],
    "bootstrap": {"n": 120, "cluster_by": "patient_id", "seed": 0},
    "cost": {"input_per_1k": 0.15, "output_per_1k": 0.60},
    "freeze": {"objective": "cohen_kappa", "grid": {"min_overlap": [0.2, 0.5, 0.8]}},
}


@pytest.fixture(scope="module")
def dataset():
    return generate(n_patients=60, labels=("diarrhea", "constipation"), seed=0)


@pytest.fixture(scope="module")
def splits(dataset):
    return dataset.split_by_patient(0.5)


class TestFullProtocol:
    def test_tune_freeze_evaluate_end_to_end(self, splits, tmp_path_factory):
        dev, test = splits
        out = tmp_path_factory.mktemp("protocol")

        # 1. tune on dev
        cfg = EvalConfig.from_dict(BASE_CFG)
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)
        assert result.n_evaluated == 3
        tuned_path = out / "tuned.yaml"
        result.config.to_yaml(tuned_path)

        # 2. freeze
        frozen = freeze(EvalConfig.from_yaml(tuned_path), tuning=result)
        frozen_path = write_frozen(frozen, out / "frozen.json")
        assert frozen.config.split is Split.TEST

        # 3. evaluate on test with the verified frozen config
        manifest = load_frozen(frozen_path)
        report = run_eval(
            test.all_predictions,
            test.annotations,
            manifest.config,
            index=test.index,
            frozen=manifest,
            annotator_records=[*test.second_annotator, *test.annotations],
        )
        report.split_check = check_split_disjoint(dev.annotations, test.annotations)
        written = report.write_all(out / "report")

        # --- the frozen config produced a usable, well-formed result ---
        assert report.n_units > 0
        assert report.config_hash == manifest.config_hash
        assert set(report.metrics.systems()) == {"careful_reader", "negation_blind"}
        assert report.split_check["ok"] is True

        for name in (
            "html",
            "markdown",
            "metrics_json",
            "errors_csv",
            "confusion_csv",
            "units_csv",
            "metrics_csv",
        ):
            assert written[name].exists(), name
            assert written[name].stat().st_size > 0, name

    def test_tuned_config_beats_a_naive_baseline(self, splits):
        """SPEC §8.4: tuning must actually buy something on held-out data."""
        dev, test = splits
        cfg = EvalConfig.from_dict(BASE_CFG)
        result = tune(cfg, dev.all_predictions, dev.annotations, index=dev.index)

        def kappa(config: EvalConfig) -> float:
            rep = run_eval(
                test.all_predictions,
                test.annotations,
                config.model_copy(update={"split": Split.TEST}),
                index=test.index,
                frozen=freeze(config),
                compare_systems=False,
            )
            return rep.metrics["careful_reader"].metrics["cohen_kappa"].point

        worst_grid_value = min(BASE_CFG["freeze"]["grid"]["min_overlap"])  # type: ignore[index]
        tuned = kappa(result.config)
        naive = kappa(cfg.with_overrides(min_overlap=worst_grid_value))
        assert tuned >= naive

    def test_evaluate_refuses_a_tampered_frozen_config(self, splits, tmp_path):
        """SPEC §11: the protocol's central guarantee, end to end."""
        from clinval.exceptions import FrozenConfigError

        dev, _ = splits
        result = tune(
            EvalConfig.from_dict(BASE_CFG), dev.all_predictions, dev.annotations, index=dev.index
        )
        path = write_frozen(freeze(result.config, tuning=result), tmp_path / "frozen.json")

        raw = json.loads(path.read_text())
        raw["config"]["matching"]["span"]["min_overlap"] = 0.123
        path.write_text(json.dumps(raw))

        with pytest.raises(FrozenConfigError, match="modified after freezing"):
            load_frozen(path)

    def test_report_contains_every_section(self, splits, tmp_path):
        dev, test = splits
        result = tune(
            EvalConfig.from_dict(BASE_CFG), dev.all_predictions, dev.annotations, index=dev.index
        )
        manifest = load_frozen(
            write_frozen(freeze(result.config, tuning=result), tmp_path / "f.json")
        )
        report = run_eval(
            test.all_predictions,
            test.annotations,
            manifest.config,
            index=test.index,
            frozen=manifest,
            annotator_records=[*test.second_annotator, *test.annotations],
        )
        report.split_check = check_split_disjoint(dev.annotations, test.annotations)
        written = report.write_all(tmp_path / "report")

        md = written["markdown"].read_text()
        html = written["html"].read_text()
        for section in (
            "Provenance",
            "Agreement",
            "Confusion",
            "Per label",
            "Error analysis",
            "Cost and latency",
            "Inter-annotator agreement",
            "System comparisons",
            "Tuning record",
            "How to read this",
        ):
            assert section in md, f"missing from markdown: {section}"
            assert section in html, f"missing from html: {section}"

        assert "{{" not in html and "{%" not in html
        assert "Frozen config verified" in html

    def test_metrics_json_matches_the_spec_shape(self, splits, tmp_path):
        """SPEC §6.3."""
        _, test = splits
        cfg = EvalConfig.from_dict({**BASE_CFG, "split": "dev"})
        report = run_eval(test.all_predictions, test.annotations, cfg, index=test.index)
        path = report.write_json(tmp_path / "metrics.json")
        data = json.loads(path.read_text())
        for key in ("config_hash", "units", "metrics", "cis", "per_label", "error_counts", "cost"):
            assert key in data, key
        assert isinstance(data["units"], int)
        assert isinstance(data["per_label"], list)

    def test_unfrozen_run_is_labelled_in_sample(self, splits, tmp_path):
        _, test = splits
        cfg = EvalConfig.from_dict({**BASE_CFG, "split": "dev"})
        report = run_eval(test.all_predictions, test.annotations, cfg, index=test.index)
        html = report.write_all(tmp_path / "r")["html"].read_text()
        assert "not an out-of-sample result" in html

    def test_paired_comparison_reports_the_right_direction(self, splits):
        """The paired bootstrap must point the right way and report a usable interval.

        Significance is deliberately **not** asserted. This config sets
        ``require_span`` with ``min_overlap`` 0.5, which withholds credit from
        careful_reader's over-quoted spans and narrows the gap; at 30 test
        patients the interval can straddle zero. Pinning a p-value here would
        be a flaky overclaim about an underpowered comparison -- which is the
        same mistake clinval exists to help people avoid.
        """
        _, test = splits
        cfg = EvalConfig.from_dict({**BASE_CFG, "split": "dev"})
        report = run_eval(test.all_predictions, test.annotations, cfg, index=test.index)
        kappa_rows = [c for c in report.comparisons if c["metric"] == "cohen_kappa"]
        assert len(kappa_rows) == 1
        row = kappa_rows[0]
        assert row["n_shared_units"] > 0
        # careful_reader sorts first, and is the better system
        assert row["system_a"] == "careful_reader"
        assert row["difference"] > 0
        assert row["ci_low"] <= row["difference"] <= row["ci_high"]
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["n_clusters"] == len(test.patients)

    def test_paired_comparison_detects_the_gap_under_a_lenient_span_policy(self, splits):
        """With spans not gating credit, the two mocks are far enough apart to separate."""
        _, test = splits
        cfg = EvalConfig.from_dict(
            {
                **BASE_CFG,
                "split": "dev",
                "matching": {"span": {"mode": "overlap", "min_overlap": 0.5}},
                "bootstrap": {"n": 600, "cluster_by": "patient_id", "seed": 0},
            }
        )
        report = run_eval(test.all_predictions, test.annotations, cfg, index=test.index)
        row = next(c for c in report.comparisons if c["metric"] == "cohen_kappa")
        assert row["difference"] > 0
        assert row["ci_low"] > 0
        assert row["p_value"] < 0.05


class TestFhirgroundInterop:
    """SPEC §8.4: clinval must accept fhirground's output as-is.

    The fixture follows the contract in the fhirground spec: ``system`` is the
    config name, ``label`` is ``<ResourceType>:<code>``, spans come from the
    evidence quote, and ``run`` carries model provenance. The colon-bearing,
    high-cardinality label is the part most likely to break a consumer.
    """

    def test_predictions_parse(self):
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        assert len(preds) == 12
        assert {p.system for p in preds} == {"fhirground-demo"}

    def test_resource_qualified_labels_survive_intact(self):
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        labels = {p.label for p in preds}
        assert labels == {"Condition:K59.1", "Condition:K59.0"}
        assert all(":" in label for label in labels)

    def test_full_pipeline_accepts_the_output(self):
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        golds = read_annotations(FIXTURES / "fhirground_gold.jsonl")
        index = read_index_dates(FIXTURES / "fhirground_index.csv")

        cfg = EvalConfig.from_dict(
            {
                "split": "dev",
                "unit": "patient_window",
                "matching": {
                    "span": {"mode": "overlap", "min_overlap": 0.5},
                    "ontology": {"icd-10": {"prefix_depth": 3}},
                },
                "window": {"size": "6M", "n_windows": 2},
                "bootstrap": {"n": 50, "cluster_by": "patient_id", "seed": 0},
                "cost": {"input_per_1k": 0.15, "output_per_1k": 0.60},
            }
        )
        report = run_eval(preds, golds, cfg, index=index)

        assert report.n_units > 0
        assert report.metrics.systems() == ["fhirground-demo"]
        rep = report.metrics["fhirground-demo"]
        assert rep.confusion.n > 0
        # the fixture includes a negated mention scored as positive, so there is
        # at least one tagged disagreement to find
        assert len(report.errors) > 0

    def test_token_metadata_gives_a_known_cost(self):
        """Unlike langextract, fhirground does report tokens, so cost is complete."""
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        golds = read_annotations(FIXTURES / "fhirground_gold.jsonl")
        index = read_index_dates(FIXTURES / "fhirground_index.csv")
        cfg = EvalConfig.from_dict(
            {
                "split": "dev",
                "unit": "patient_window",
                "window": {"size": "6M", "n_windows": 2},
                "bootstrap": {"n": 20, "seed": 0},
                "cost": {"input_per_1k": 0.15, "output_per_1k": 0.60},
            }
        )
        report = run_eval(preds, golds, cfg, index=index)
        cost = report.cost.per_system["fhirground-demo"]
        assert cost.cost_status == "complete"
        assert cost.cost is not None
        assert cost.cost > 0
        assert cost.latency_mean_s is not None

    def test_negated_mention_is_tagged_as_a_misreading(self):
        """The fixture's FN0000003 quotes 'denies diarrhea' and calls it positive."""
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        golds = read_annotations(FIXTURES / "fhirground_gold.jsonl")
        index = read_index_dates(FIXTURES / "fhirground_index.csv")
        cfg = EvalConfig.from_dict(
            {
                "split": "dev",
                "unit": "patient_window",
                "window": {"size": "6M", "n_windows": 2},
                "bootstrap": {"n": 20, "seed": 0},
            }
        )
        report = run_eval(preds, golds, cfg, index=index)
        rows = report.errors.df
        assert "misreading" in set(rows["mechanism"])
        hit = rows[rows["mechanism"] == "misreading"].iloc[0]
        assert "denies" in str(hit["span_text"])

    def test_note_unit_also_works(self):
        """No index file needed when the unit is the note."""
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        golds = read_annotations(FIXTURES / "fhirground_gold.jsonl")
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 20, "seed": 0}}
        )
        report = run_eval(preds, golds, cfg)
        assert report.n_units == 12

    def test_report_renders_for_a_single_system(self, tmp_path):
        preds = read_predictions(FIXTURES / "fhirground_clinval_predictions.jsonl")
        golds = read_annotations(FIXTURES / "fhirground_gold.jsonl")
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 20, "seed": 0}}
        )
        report = run_eval(preds, golds, cfg)
        written = report.write_all(tmp_path / "r")
        html = written["html"].read_text()
        assert "fhirground-demo" in html
        # with one system there is nothing to compare, and the section is omitted
        assert report.comparisons == []


class TestRunnerGuards:
    def test_patient_window_without_index_is_a_clear_error(self, dataset):
        from clinval.exceptions import ConfigError

        cfg = EvalConfig.from_dict({**BASE_CFG, "split": "dev"})
        with pytest.raises(ConfigError, match="needs per-patient index dates"):
            run_eval(dataset.all_predictions, dataset.annotations, cfg, index=None)

    def test_accepts_file_paths(self, dataset, tmp_path):
        paths = dataset.write(tmp_path)
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 20, "seed": 0}}
        )
        report = run_eval(paths["pred"], paths["gold"], cfg)
        assert report.n_units > 0

    def test_accepts_a_frozen_config_object_as_config(self, splits):
        _, test = splits
        frozen = freeze(EvalConfig.from_dict(BASE_CFG))
        report = run_eval(
            test.all_predictions, test.annotations, frozen, index=test.index, frozen=frozen
        )
        assert report.frozen is not None

    def test_coverage_warning_when_negatives_are_missing(self, dataset):
        positives = [p for p in dataset.predictions["careful_reader"] if p.value]
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 20, "seed": 0}}
        )
        report = run_eval(positives, dataset.annotations, cfg)
        assert any("unscored, not as negatives" in w for w in report.warnings)

    def test_materializing_negatives_removes_the_warning(self, dataset, tmp_path):
        from clinval.io import materialize_negatives

        positives = [p for p in dataset.predictions["careful_reader"] if p.value]
        filled = materialize_negatives(
            positives,
            note_ids=[n["note_id"] for n in dataset.notes],
            labels=list(dataset.labels),
        )
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 20, "seed": 0}}
        )
        report = run_eval(filled, dataset.annotations, cfg)
        assert not any("unscored, not as negatives" in w for w in report.warnings)

    def test_cost_model_from_a_json_file(self, dataset, tmp_path):
        p = tmp_path / "cost.json"
        p.write_text(json.dumps({"input_per_1k": 1.0, "output_per_1k": 2.0}))
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 10, "seed": 0}}
        )
        report = run_eval(dataset.predictions["careful_reader"], dataset.annotations, cfg, cost=p)
        assert report.cost.model.input_per_1k == 1.0

    def test_single_annotator_records_warn_instead_of_crashing(self, dataset):
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 10, "seed": 0}}
        )
        report = run_eval(
            dataset.predictions["careful_reader"],
            dataset.annotations,
            cfg,
            annotator_records=dataset.annotations,
        )
        assert report.iaa == {}
        assert any("single annotator" in w for w in report.warnings)

    def test_unit_note_needs_no_index(self, dataset):
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 10, "seed": 0}}
        )
        assert run_eval(dataset.all_predictions, dataset.annotations, cfg).n_units > 0

    def test_metrics_table_dataframe(self, dataset):
        cfg = EvalConfig.from_dict(
            {"split": "dev", "unit": "note", "bootstrap": {"n": 10, "seed": 0}}
        )
        report = run_eval(dataset.all_predictions, dataset.annotations, cfg)
        df = report.metrics.to_dataframe()
        assert set(df.columns) >= {"system", "metric", "point", "ci_low", "ci_high", "method"}
        assert len(df) == len(report.metrics.systems()) * len(cfg.metrics)
