"""Report generation: figures, template rendering, and the caveats that must appear."""

from __future__ import annotations

import pytest

from clinval.errors import MECHANISMS
from clinval.report.builder import (
    plot_confusion,
    plot_error_mechanisms,
    plot_metrics,
    plot_per_label,
    render_html,
    render_markdown,
    write_report,
)
from clinval.runner import run_eval
from clinval.schemas.config import EvalConfig
from clinval.synth import generate

CFG = {
    "split": "dev",
    "unit": "patient_window",
    "window": {"size": "6M", "n_windows": 2},
    "bootstrap": {"n": 25, "cluster_by": "patient_id", "seed": 0},
    "cost": {"input_per_1k": 0.15, "output_per_1k": 0.60},
}


@pytest.fixture(scope="module")
def report():
    data = generate(n_patients=24, labels=("diarrhea", "constipation"), seed=0)
    return run_eval(
        data.all_predictions,
        data.annotations,
        EvalConfig.from_dict(CFG),
        index=data.index,
        annotator_records=[*data.second_annotator, *data.annotations],
    )


@pytest.fixture(scope="module")
def single_system_report():
    data = generate(n_patients=14, labels=("diarrhea",), seed=1)
    return run_eval(
        data.predictions["careful_reader"],
        data.annotations,
        EvalConfig.from_dict(CFG),
        index=data.index,
    )


class TestFigures:
    @pytest.mark.parametrize(
        "fn", [plot_metrics, plot_per_label, plot_error_mechanisms, plot_confusion]
    )
    def test_each_figure_is_written(self, report, tmp_path, fn):
        out = tmp_path / f"{fn.__name__}.png"
        produced = fn(report, out)
        assert produced is not None
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_figures_work_for_one_system(self, single_system_report, tmp_path):
        for fn in (plot_metrics, plot_per_label, plot_error_mechanisms, plot_confusion):
            assert fn(single_system_report, tmp_path / f"{fn.__name__}.png") is not None

    def test_metrics_figure_skipped_when_no_metrics_requested(self, report, tmp_path):
        stripped = report
        original = stripped.config
        stripped.config = original.model_copy(update={"metrics": ["prevalence"]})
        try:
            assert plot_metrics(stripped, tmp_path / "x.png") is None
        finally:
            stripped.config = original


class TestMarkdown:
    def test_sections_present(self, report):
        md = render_markdown(report)
        for section in (
            "Provenance",
            "Agreement",
            "Confusion matrices",
            "Prediction coverage",
            "Per label",
            "Error analysis",
            "Cost and latency",
            "Inter-annotator agreement",
            "System comparisons",
            "How to read this",
        ):
            assert section in md, section

    def test_every_mechanism_row_is_printed_including_zeros(self, report):
        """A zero must be distinguishable from 'not checked'."""
        md = render_markdown(report)
        for mech in MECHANISMS:
            assert mech in md, mech

    def test_unfrozen_run_is_flagged(self, report):
        md = render_markdown(report)
        assert "not an out-of-sample result" in md

    def test_interval_methods_are_named(self, report):
        md = render_markdown(report)
        assert "wilson" in md
        assert "clustered_bootstrap" in md

    def test_unscored_caveat_present(self, report):
        md = render_markdown(report)
        assert "Unscored is not negative" in md

    def test_no_clinical_validity_claim(self, report):
        md = render_markdown(report)
        assert "does not certify clinical validity" in md

    def test_no_unrendered_template_syntax(self, report):
        md = render_markdown(report)
        assert "{{" not in md
        assert "{%" not in md

    def test_figure_links_included_when_supplied(self, report):
        md = render_markdown(report, figures={"metrics": "figures/metrics.png"})
        assert "figures/metrics.png" in md


class TestHtml:
    def test_is_self_contained_static_html(self, report):
        html = render_html(report)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "<script" not in html.lower()
        assert "</html>" in html

    def test_no_unrendered_template_syntax(self, report):
        html = render_html(report)
        assert "{{" not in html
        assert "{%" not in html

    def test_no_raw_nan_cells(self, report):
        """An undefined number must read as n/a, not as the float nan."""
        html = render_html(report)
        assert ">nan<" not in html
        assert ">None<" not in html

    def test_dark_mode_styles_present(self, report):
        assert "prefers-color-scheme: dark" in render_html(report)

    def test_unfrozen_warning_is_prominent(self, report):
        html = render_html(report)
        assert "not an out-of-sample result" in html
        assert "note bad" in html

    def test_cost_unknown_is_stated_not_zeroed(self, report):
        """negation_blind reports no tokens, so its cost must read unknown."""
        html = render_html(report)
        assert "<em>unknown</em>" in html

    def test_single_system_omits_the_comparison_section(self, single_system_report):
        html = render_html(single_system_report)
        assert "System comparisons" not in html


class TestWriteReport:
    def test_all_artifacts(self, report, tmp_path):
        written = write_report(report, tmp_path)
        for key in (
            "html",
            "markdown",
            "metrics_json",
            "errors_csv",
            "confusion_csv",
            "units_csv",
            "metrics_csv",
        ):
            assert written[key].exists(), key
            assert written[key].stat().st_size > 0, key
        for key in (
            "figure_metrics",
            "figure_per_label",
            "figure_error_mechanisms",
            "figure_confusion",
        ):
            assert written[key].exists(), key

    def test_figures_can_be_disabled(self, report, tmp_path):
        written = write_report(report, tmp_path, figures=False)
        assert not (tmp_path / "figures").exists()
        assert written["html"].exists()

    def test_html_references_the_figures_it_wrote(self, report, tmp_path):
        write_report(report, tmp_path)
        html = (tmp_path / "report.html").read_text()
        for name in ("metrics", "per_label", "error_mechanisms", "confusion"):
            assert f"figures/{name}.png" in html
            assert (tmp_path / "figures" / f"{name}.png").exists()

    def test_errors_csv_has_the_spec_columns(self, report, tmp_path):
        write_report(report, tmp_path)
        header = (tmp_path / "errors.csv").read_text().splitlines()[0]
        for col in ("error_id", "system", "label", "unit_key", "error_type", "mechanism"):
            assert col in header

    def test_confusion_csv_names_cells(self, report, tmp_path):
        write_report(report, tmp_path)
        text = (tmp_path / "confusion.csv").read_text()
        assert "cell" in text.splitlines()[0]
        assert any(c in text for c in ("TP", "FP", "FN", "TN"))

    def test_report_writers_on_the_object(self, report, tmp_path):
        assert report.write_json(tmp_path / "m.json").exists()
        assert report.write_markdown(tmp_path / "m.md").exists()
        assert report.write_html(tmp_path / "html_out").exists()
        assert report.to_markdown().startswith("# clinval")

    def test_idempotent(self, report, tmp_path):
        write_report(report, tmp_path)
        first = (tmp_path / "report.md").read_text()
        write_report(report, tmp_path)
        assert (tmp_path / "report.md").read_text() == first
