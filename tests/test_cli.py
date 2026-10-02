"""CLI smoke tests: every command, plus the dev-acknowledgement guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clinval.cli import app
from clinval.synth import generate

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def run(*args: str):
    """Invoke the CLI and return the result."""
    return runner.invoke(app, list(args))


def flat(result) -> str:
    """CLI output with whitespace collapsed.

    rich wraps to the terminal width, so a phrase can be split across lines.
    Assertions about message content should not depend on where it wrapped.
    """
    return " ".join(result.output.split())


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """A synth dataset on disk, dev/test split, with a config.

    Module-scoped and deliberately small: these are smoke tests for the command
    surface, not for the statistics. The generated config's bootstrap count is
    reduced here so the suite stays fast -- the interval machinery itself is
    covered properly in ``test_intervals.py``.
    """
    path = tmp_path_factory.mktemp("cli_demo")
    res = run("synth", "--out", str(path), "--n-patients", "16", "--seed", "0")
    assert res.exit_code == 0, res.output

    cfg = path / "eval.yaml"
    cfg.write_text(cfg.read_text().replace("n: 500", "n: 25"))
    return path


class TestHelpAndVersion:
    def test_no_args_shows_help(self):
        """Click exits 2 when it prints help because no command was given."""
        res = run()
        assert res.exit_code in (0, 2)
        assert "Validation harness" in flat(res)

    def test_help(self):
        assert run("--help").exit_code == 0

    @pytest.mark.parametrize(
        "cmd",
        [
            "synth",
            "tune",
            "freeze",
            "evaluate",
            "run",
            "consensus",
            "negatives",
            "adjudicate",
            "validate",
        ],
    )
    def test_every_command_has_help(self, cmd):
        res = run(cmd, "--help")
        assert res.exit_code == 0, res.output

    def test_version(self):
        from clinval import __version__

        res = run("version")
        assert res.exit_code == 0
        assert __version__ in res.output


class TestSynth:
    def test_writes_split_artifacts_and_a_config(self, tmp_path):
        res = run("synth", "--out", str(tmp_path), "--n-patients", "12", "--seed", "1")
        assert res.exit_code == 0, res.output
        for name in (
            "dev_pred.jsonl",
            "dev_gold.jsonl",
            "dev_index.csv",
            "dev_notes.jsonl",
            "test_pred.jsonl",
            "test_gold.jsonl",
            "test_index.csv",
            "eval.yaml",
        ):
            assert (tmp_path / name).exists(), name

    def test_states_that_no_real_data_was_used(self, tmp_path):
        res = run("synth", "--out", str(tmp_path), "--n-patients", "4")
        assert "No real record" in flat(res)

    def test_no_split(self, tmp_path):
        res = run("synth", "--out", str(tmp_path), "--n-patients", "6", "--no-split")
        assert res.exit_code == 0
        assert (tmp_path / "gold.jsonl").exists()
        assert not (tmp_path / "dev_gold.jsonl").exists()

    def test_custom_labels(self, tmp_path):
        res = run(
            "synth", "--out", str(tmp_path), "--n-patients", "5", "--labels", "vomiting,nausea"
        )
        assert res.exit_code == 0
        cfg = (tmp_path / "eval.yaml").read_text()
        assert "vomiting" in cfg
        assert "nausea" in cfg

    def test_unsupported_label_is_a_clean_error(self, tmp_path):
        res = run("synth", "--out", str(tmp_path), "--labels", "pneumonia")
        assert res.exit_code == 1
        assert "unsupported label" in flat(res)
        assert "Traceback" not in res.output

    def test_deterministic_output(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        run("synth", "--out", str(a), "--n-patients", "6", "--seed", "5")
        run("synth", "--out", str(b), "--n-patients", "6", "--seed", "5")
        assert (a / "dev_gold.jsonl").read_bytes() == (b / "dev_gold.jsonl").read_bytes()


class TestProtocolCommands:
    def test_tune(self, demo):
        res = run(
            "tune",
            "--config",
            str(demo / "eval.yaml"),
            "--pred",
            str(demo / "dev_pred.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--index",
            str(demo / "dev_index.csv"),
            "--out",
            str(demo / "tuned.yaml"),
        )
        assert res.exit_code == 0, res.output
        assert (demo / "tuned.yaml").exists()
        assert "chosen" in flat(res)
        assert "clinval freeze" in flat(res)

    def test_tune_refuses_a_test_config(self, demo):
        cfg = demo / "test_split.yaml"
        cfg.write_text((demo / "eval.yaml").read_text().replace("split: dev", "split: test"))
        res = run(
            "tune",
            "--config",
            str(cfg),
            "--pred",
            str(demo / "dev_pred.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--index",
            str(demo / "dev_index.csv"),
            "--out",
            str(demo / "t.yaml"),
        )
        assert res.exit_code == 1
        assert "refusing to tune" in flat(res)

    def test_freeze(self, demo):
        res = run("freeze", "--config", str(demo / "eval.yaml"), "--out", str(demo / "frozen.json"))
        assert res.exit_code == 0, res.output
        manifest = json.loads((demo / "frozen.json").read_text())
        assert len(manifest["config_hash"]) == 64
        assert "config hash" in flat(res)

    def test_evaluate(self, demo):
        run("freeze", "--config", str(demo / "eval.yaml"), "--out", str(demo / "frozen.json"))
        res = run(
            "evaluate",
            "--frozen",
            str(demo / "frozen.json"),
            "--pred",
            str(demo / "test_pred.jsonl"),
            "--gold",
            str(demo / "test_gold.jsonl"),
            "--index",
            str(demo / "test_index.csv"),
            "--dev-gold",
            str(demo / "dev_gold.jsonl"),
            "--out",
            str(demo / "report"),
        )
        assert res.exit_code == 0, res.output
        assert (demo / "report" / "report.html").exists()
        assert (demo / "report" / "metrics.json").exists()
        assert "careful_reader" in flat(res)

    def test_evaluate_refuses_a_tampered_manifest(self, demo):
        run("freeze", "--config", str(demo / "eval.yaml"), "--out", str(demo / "frozen.json"))
        path = demo / "frozen.json"
        raw = json.loads(path.read_text())
        raw["config"]["bootstrap"]["seed"] = 999
        path.write_text(json.dumps(raw))
        res = run(
            "evaluate",
            "--frozen",
            str(path),
            "--pred",
            str(demo / "test_pred.jsonl"),
            "--gold",
            str(demo / "test_gold.jsonl"),
            "--index",
            str(demo / "test_index.csv"),
            "--out",
            str(demo / "report"),
        )
        assert res.exit_code == 1
        assert "modified after freezing" in flat(res)

    def test_evaluate_exits_nonzero_on_split_overlap(self, demo):
        """Passing the same gold as dev and test is an overlap by definition."""
        run("freeze", "--config", str(demo / "eval.yaml"), "--out", str(demo / "frozen.json"))
        res = run(
            "evaluate",
            "--frozen",
            str(demo / "frozen.json"),
            "--pred",
            str(demo / "test_pred.jsonl"),
            "--gold",
            str(demo / "test_gold.jsonl"),
            "--index",
            str(demo / "test_index.csv"),
            "--dev-gold",
            str(demo / "test_gold.jsonl"),
            "--out",
            str(demo / "report2"),
        )
        assert res.exit_code == 2
        assert "split overlap" in flat(res)

    def test_full_protocol_in_sequence(self, demo):
        assert (
            run(
                "tune",
                "--config",
                str(demo / "eval.yaml"),
                "--pred",
                str(demo / "dev_pred.jsonl"),
                "--gold",
                str(demo / "dev_gold.jsonl"),
                "--index",
                str(demo / "dev_index.csv"),
                "--out",
                str(demo / "tuned.yaml"),
            ).exit_code
            == 0
        )
        assert (
            run(
                "freeze", "--config", str(demo / "tuned.yaml"), "--out", str(demo / "frozen.json")
            ).exit_code
            == 0
        )
        res = run(
            "evaluate",
            "--frozen",
            str(demo / "frozen.json"),
            "--pred",
            str(demo / "test_pred.jsonl"),
            "--gold",
            str(demo / "test_gold.jsonl"),
            "--index",
            str(demo / "test_index.csv"),
            "--out",
            str(demo / "report"),
        )
        assert res.exit_code == 0, res.output


class TestRunCommand:
    def test_dev_run_requires_acknowledgement(self, demo):
        res = run(
            "run",
            "--config",
            str(demo / "eval.yaml"),
            "--pred",
            str(demo / "dev_pred.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--index",
            str(demo / "dev_index.csv"),
            "--out",
            str(demo / "r"),
        )
        assert res.exit_code == 1
        assert "--i-understand-this-is-dev" in flat(res)

    def test_dev_run_with_acknowledgement(self, demo):
        res = run(
            "run",
            "--config",
            str(demo / "eval.yaml"),
            "--pred",
            str(demo / "dev_pred.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--index",
            str(demo / "dev_index.csv"),
            "--out",
            str(demo / "r"),
            "--i-understand-this-is-dev",
        )
        assert res.exit_code == 0, res.output
        assert (demo / "r" / "report.html").exists()
        assert "in-sample" in flat(res)

    def test_run_refuses_a_test_config_even_with_the_flag(self, demo):
        cfg = demo / "test_split.yaml"
        cfg.write_text((demo / "eval.yaml").read_text().replace("split: dev", "split: test"))
        res = run(
            "run",
            "--config",
            str(cfg),
            "--pred",
            str(demo / "test_pred.jsonl"),
            "--gold",
            str(demo / "test_gold.jsonl"),
            "--index",
            str(demo / "test_index.csv"),
            "--out",
            str(demo / "r2"),
            "--i-understand-this-is-dev",
        )
        assert res.exit_code == 1
        assert "without a frozen config" in flat(res)

    def test_missing_index_is_a_clean_error(self, demo):
        res = run(
            "run",
            "--config",
            str(demo / "eval.yaml"),
            "--pred",
            str(demo / "dev_pred.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--out",
            str(demo / "r3"),
            "--i-understand-this-is-dev",
        )
        assert res.exit_code == 1
        assert "--index" in flat(res)
        assert "Traceback" not in res.output


class TestConsensus:
    def test_two_systems(self, demo):
        res = run(
            "consensus",
            "--systems",
            str(demo / "dev_pred_careful_reader.jsonl"),
            "--systems",
            str(demo / "dev_pred_negation_blind.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--config",
            str(demo / "eval.yaml"),
            "--index",
            str(demo / "dev_index.csv"),
            "--rule",
            "majority",
            "--out",
            str(demo / "cons"),
        )
        assert res.exit_code == 0, res.output
        assert (demo / "cons" / "consensus_units.csv").exists()
        assert (demo / "cons" / "review_queue.csv").exists()
        assert (demo / "cons" / "consensus.json").exists()
        assert "Consensus gate" in flat(res)

    def test_one_system_refused(self, demo):
        res = run(
            "consensus",
            "--systems",
            str(demo / "dev_pred_careful_reader.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--out",
            str(demo / "cons2"),
        )
        assert res.exit_code == 1
        assert "at least two" in flat(res)

    @pytest.mark.parametrize("rule", ["majority", "unanimous", "any"])
    def test_all_rules(self, demo, rule):
        res = run(
            "consensus",
            "--systems",
            str(demo / "dev_pred_careful_reader.jsonl"),
            "--systems",
            str(demo / "dev_pred_negation_blind.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--config",
            str(demo / "eval.yaml"),
            "--index",
            str(demo / "dev_index.csv"),
            "--rule",
            rule,
            "--out",
            str(demo / f"cons_{rule}"),
        )
        assert res.exit_code == 0, res.output


class TestNegatives:
    def test_materializes_negatives(self, demo, tmp_path):
        from clinval.io import read_predictions, write_predictions

        positives = [p for p in read_predictions(demo / "dev_pred.jsonl") if p.value]
        pos_path = tmp_path / "pos.jsonl"
        write_predictions(positives, pos_path)

        out = tmp_path / "filled.jsonl"
        res = run(
            "negatives",
            "--from",
            str(pos_path),
            "--notes",
            str(demo / "dev_notes.jsonl"),
            "--labels",
            "diarrhea,constipation",
            "--out",
            str(out),
        )
        assert res.exit_code == 0, res.output
        filled = read_predictions(out)
        assert len(filled) > len(positives)
        assert "explicit negatives added" in flat(res)


class TestAdjudicate:
    def test_collapses_two_annotators(self, demo, tmp_path):
        from clinval.io import read_annotations

        out = tmp_path / "adj.jsonl"
        res = run(
            "adjudicate",
            "--gold",
            str(demo / "dev_gold_two_annotators.jsonl"),
            "--out",
            str(out),
            "--rule",
            "agreement",
        )
        assert res.exit_code == 0, res.output
        recs = read_annotations(out)
        assert recs
        assert all(r.adjudicated for r in recs)
        assert "annotator" in flat(res)

    def test_reports_dropped_disagreements(self, demo, tmp_path):
        res = run(
            "adjudicate",
            "--gold",
            str(demo / "dev_gold_two_annotators.jsonl"),
            "--out",
            str(tmp_path / "adj.jsonl"),
        )
        assert "dropped" in flat(res)

    def test_unknown_rule_is_a_clean_error(self, demo, tmp_path):
        res = run(
            "adjudicate",
            "--gold",
            str(demo / "dev_gold_two_annotators.jsonl"),
            "--out",
            str(tmp_path / "adj.jsonl"),
            "--rule",
            "vibes",
        )
        assert res.exit_code == 1
        assert "unknown adjudication rule" in flat(res)


class TestValidate:
    def test_checks_each_input_type(self, demo):
        res = run(
            "validate",
            "--config",
            str(demo / "eval.yaml"),
            "--pred",
            str(demo / "dev_pred.jsonl"),
            "--gold",
            str(demo / "dev_gold.jsonl"),
            "--index",
            str(demo / "dev_index.csv"),
        )
        assert res.exit_code == 0, res.output
        assert flat(res).count("ok") >= 4

    def test_warns_on_unadjudicated_multi_rater_gold(self, demo):
        res = run("validate", "--gold", str(demo / "dev_gold_two_annotators.jsonl"))
        assert res.exit_code == 0
        assert "clinval adjudicate" in flat(res)

    def test_reports_the_offending_line(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            '{"system":"s","note_id":"N1","label":"l","value":true}\n{"system":"s","note_id":"N2"}\n'
        )
        res = run("validate", "--pred", str(bad))
        assert res.exit_code == 1
        assert ":2]" in flat(res)
        assert "Traceback" not in res.output

    def test_nothing_to_validate(self):
        res = run("validate")
        assert res.exit_code == 1
        assert "nothing to validate" in flat(res)

    def test_fhirground_output_validates(self):
        res = run(
            "validate",
            "--pred",
            str(FIXTURES / "fhirground_clinval_predictions.jsonl"),
            "--gold",
            str(FIXTURES / "fhirground_gold.jsonl"),
            "--index",
            str(FIXTURES / "fhirground_index.csv"),
        )
        assert res.exit_code == 0, res.output
        assert "fhirground-demo" in flat(res)


class TestModuleEntryPoint:
    def test_python_m_clinval(self):
        import subprocess
        import sys

        out = subprocess.run(
            [sys.executable, "-m", "clinval", "version"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        from clinval import __version__

        assert __version__ in out.stdout


class TestNoRealDataInCliOutput:
    def test_generated_dataset_has_only_placeholder_ids(self, demo):
        import re

        text = (demo / "dev_gold.jsonl").read_text()
        for match in re.finditer(r'"patient_id":"([^"]+)"', text):
            assert re.fullmatch(r"P\d{5}", match.group(1))

    def test_generator_is_the_only_data_source(self):
        """A guard on the claim the README makes."""
        data = generate(n_patients=3, seed=0)
        assert all(n["text"] for n in data.notes)
        assert all(str(p).startswith("P") for p in data.patients)
