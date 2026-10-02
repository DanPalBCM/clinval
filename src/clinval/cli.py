"""The ``clinval`` command line.

Commands mirror SPEC §5.1. The protocol commands (``tune``, ``freeze``,
``evaluate``) are the intended path; ``run`` is the single-shot convenience
that refuses test data without an explicit acknowledgement flag.

Every command exits non-zero with a readable message on bad input rather than a
traceback -- a harness that crashes with a stack trace on a malformed JSONL row
has failed at its own job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from clinval.exceptions import ClinvalError
from clinval.schemas.config import EvalConfig, Split, Unit

if TYPE_CHECKING:
    from clinval.consensus import ConsensusRule

app = typer.Typer(
    name="clinval",
    help="Validation harness for clinical LLM extraction.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

#: Metrics shown in the terminal summary, in priority order. The console is
#: narrow and the full metric set squeezes to unreadable; everything is in the
#: written report regardless.
_SUMMARY_METRICS = ("cohen_kappa", "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1")
_MAX_SUMMARY_METRICS = 4


def _fail(exc: Exception, code: int = 1) -> None:
    """Print a clean error and exit.

    The message is escaped before printing: clinval's record errors end with a
    ``[path:line]`` location, which rich would otherwise parse as a markup
    closing tag and discard the entire line.
    """
    err_console.print(f"[bold red]error:[/bold red] {escape(str(exc))}")
    raise typer.Exit(code)


def _load_config(path: Path | None) -> EvalConfig:
    return EvalConfig.from_yaml(path) if path else EvalConfig()


def _load_index(path: Path | None):
    from clinval.io import read_index_dates

    return read_index_dates(path) if path else None


def _needs_index(cfg: EvalConfig, index: Path | None) -> None:
    if cfg.unit is Unit.PATIENT_WINDOW and index is None:
        _fail(
            ClinvalError(
                "this config uses unit 'patient_window', which needs per-patient index "
                "dates. Pass --index path/to/index.csv, or set `unit: note` in the config."
            )
        )


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


@app.command()
def synth(
    out: Annotated[Path, typer.Option("--out", help="Directory to write the dataset into.")],
    n_patients: Annotated[int, typer.Option("--n-patients", help="Number of patients.")] = 200,
    labels: Annotated[
        str, typer.Option("--labels", help="Comma-separated label list.")
    ] = "diarrhea,constipation",
    months: Annotated[int, typer.Option("--months", help="Months of follow-up per patient.")] = 12,
    seed: Annotated[int, typer.Option("--seed", help="Random seed; output is deterministic.")] = 0,
    dev_fraction: Annotated[
        float, typer.Option("--dev-fraction", help="Share of patients assigned to dev.")
    ] = 0.5,
    no_split: Annotated[
        bool, typer.Option("--no-split", help="Write one undivided dataset instead of dev/test.")
    ] = False,
) -> None:
    """Generate a deterministic synthetic dataset.

    The split is **by patient**, never by note, so a tuned threshold is never
    tested on a patient it was chosen on.
    """
    from clinval.synth import generate

    label_list = tuple(x.strip() for x in labels.split(",") if x.strip())
    try:
        data = generate(n_patients=n_patients, months=months, labels=label_list, seed=seed)
    except (ValueError, ClinvalError) as exc:
        _fail(exc)
        return

    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if no_split:
        written.update(data.write(out))
    else:
        dev, test = data.split_by_patient(dev_fraction, seed=seed)
        written.update({f"dev_{k}": v for k, v in dev.write(out, prefix="dev_").items()})
        written.update({f"test_{k}": v for k, v in test.write(out, prefix="test_").items()})
        console.print(
            f"dev: {len(dev.patients)} patients, {len(dev.notes)} notes  |  "
            f"test: {len(test.patients)} patients, {len(test.notes)} notes"
        )

    cfg_path = out / "eval.yaml"
    cfg = EvalConfig.from_dict(
        {
            "split": "dev",
            "unit": "patient_window",
            "labels": list(label_list),
            "matching": {"span": {"mode": "overlap", "min_overlap": 0.5}},
            "window": {"size": "6M", "anchor": "index_date", "n_windows": 2},
            "bootstrap": {"n": 500, "cluster_by": "patient_id", "seed": 0},
            "cost": {"input_per_1k": 0.15, "output_per_1k": 0.60},
            "freeze": {
                "objective": "cohen_kappa",
                "grid": {"min_overlap": [0.3, 0.5, 0.7], "consensus": ["majority", "unanimous"]},
            },
        }
    )
    cfg.to_yaml(cfg_path)
    written["config"] = cfg_path

    table = Table(title=f"synthetic dataset (seed={seed})", show_lines=False)
    table.add_column("artifact")
    table.add_column("path")
    for name, path in sorted(written.items()):
        table.add_row(name, str(path))
    console.print(table)
    console.print(
        "[dim]All data is generated from a fixed sentence bank. No real record was "
        "involved at any point.[/dim]"
    )


# ---------------------------------------------------------------------------
# tune / freeze / evaluate
# ---------------------------------------------------------------------------


@app.command()
def tune(
    config: Annotated[Path, typer.Option("--config", help="Config YAML declaring freeze.grid.")],
    pred: Annotated[Path, typer.Option("--pred", help="Dev predictions JSONL.")],
    gold: Annotated[Path, typer.Option("--gold", help="Dev gold annotations JSONL.")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the tuned config.")],
    index: Annotated[
        Path | None, typer.Option("--index", help="Per-patient index dates CSV.")
    ] = None,
    system: Annotated[
        str | None, typer.Option("--system", help="Tune against one system instead of the mean.")
    ] = None,
) -> None:
    """Grid-search the declared parameters on the dev split."""
    from clinval.freeze import tune as run_tune
    from clinval.io import read_annotations, read_predictions

    try:
        cfg = _load_config(config)
        _needs_index(cfg, index)
        result = run_tune(
            cfg,
            read_predictions(pred),
            read_annotations(gold),
            index=_load_index(index),
            system=system,
        )
        result.config.to_yaml(out)
    except ClinvalError as exc:
        _fail(exc)
        return

    console.print(result.to_markdown())
    console.print(
        f"\n[bold green]chosen[/bold green] {result.best_params or '(no grid declared)'} "
        f"→ {result.objective}={result.best_score:.4f}"
    )
    console.print(f"tuned config written to [cyan]{out}[/cyan]")
    if result.is_flat:
        console.print(
            "[yellow]note:[/yellow] the objective varied by less than 0.005 across the "
            "whole grid, so this choice is effectively arbitrary on this dev set."
        )
    console.print(f"next: [cyan]clinval freeze --config {out} --out frozen.json[/cyan]")


@app.command()
def freeze(
    config: Annotated[Path, typer.Option("--config", help="Tuned config YAML.")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the freeze manifest.")],
) -> None:
    """Lock a config and record its SHA-256 hash."""
    from clinval.freeze import freeze as do_freeze
    from clinval.freeze import write_frozen

    try:
        frozen = do_freeze(_load_config(config))
        write_frozen(frozen, out)
    except ClinvalError as exc:
        _fail(exc)
        return

    console.print(f"[bold green]frozen[/bold green] → [cyan]{out}[/cyan]")
    console.print(f"  config hash: [bold]{frozen.config_hash}[/bold]")
    console.print(f"  frozen at:   {frozen.frozen_at.isoformat()}")
    console.print(f"  git commit:  {frozen.git_commit or 'not in a git repository'}")
    console.print(
        "\n[dim]`clinval evaluate` recomputes this hash and refuses to run if the config "
        "body was edited after freezing.[/dim]"
    )


@app.command()
def evaluate(
    frozen: Annotated[
        Path, typer.Option("--frozen", help="Freeze manifest from `clinval freeze`.")
    ],
    pred: Annotated[Path, typer.Option("--pred", help="Test predictions JSONL.")],
    gold: Annotated[Path, typer.Option("--gold", help="Test gold annotations JSONL.")],
    out: Annotated[Path, typer.Option("--out", help="Report output directory.")],
    index: Annotated[
        Path | None, typer.Option("--index", help="Per-patient index dates CSV.")
    ] = None,
    dev_gold: Annotated[
        Path | None,
        typer.Option("--dev-gold", help="Dev gold, to verify the splits are disjoint."),
    ] = None,
    raters: Annotated[
        Path | None,
        typer.Option("--raters", help="Multi-annotator JSONL, for inter-rater agreement."),
    ] = None,
) -> None:
    """Run a frozen config on held-out test data."""
    from clinval.freeze import check_split_disjoint, load_frozen
    from clinval.io import read_annotations
    from clinval.runner import run_eval

    try:
        manifest = load_frozen(frozen)
        _needs_index(manifest.config, index)
        golds = read_annotations(gold)
        report = run_eval(
            pred,
            golds,
            manifest.config,
            index=_load_index(index),
            frozen=manifest,
            annotator_records=raters,
        )
        if dev_gold is not None:
            report.split_check = check_split_disjoint(read_annotations(dev_gold), golds)
        written = report.write_all(out)
    except ClinvalError as exc:
        _fail(exc)
        return

    _print_summary(report, written)
    if report.split_check and not report.split_check.get("ok", True):
        err_console.print(
            "[bold red]split overlap:[/bold red] "
            f"{report.split_check.get('n_shared_patients', 0)} patient(s) and "
            f"{report.split_check.get('n_shared_notes', 0)} note(s) appear in both dev and "
            "test. The out-of-sample claim does not hold."
        )
        raise typer.Exit(2)


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", help="Config YAML.")],
    pred: Annotated[Path, typer.Option("--pred", help="Predictions JSONL.")],
    gold: Annotated[Path, typer.Option("--gold", help="Gold annotations JSONL.")],
    out: Annotated[Path, typer.Option("--out", help="Report output directory.")],
    index: Annotated[
        Path | None, typer.Option("--index", help="Per-patient index dates CSV.")
    ] = None,
    raters: Annotated[
        Path | None,
        typer.Option("--raters", help="Multi-annotator JSONL for inter-rater agreement."),
    ] = None,
    i_understand_this_is_dev: Annotated[
        bool,
        typer.Option(
            "--i-understand-this-is-dev",
            help="Acknowledge that an unfrozen run is an in-sample estimate.",
        ),
    ] = False,
) -> None:
    """Single-shot run without the freeze protocol.

    Refuses a config marked ``split: test``, and refuses a dev config unless
    ``--i-understand-this-is-dev`` is passed. The flag is deliberately
    unwieldy: an unfrozen number is not a validation result, and the command
    line should say so.
    """
    from clinval.freeze import evaluate_guard
    from clinval.runner import run_eval

    try:
        cfg = _load_config(config)
        _needs_index(cfg, index)
        evaluate_guard(cfg, frozen=None, acknowledge_dev=i_understand_this_is_dev)
        report = run_eval(pred, gold, cfg, index=_load_index(index), annotator_records=raters)
        written = report.write_all(out)
    except ClinvalError as exc:
        _fail(exc)
        return

    _print_summary(report, written)


# ---------------------------------------------------------------------------
# consensus / negatives / adjudicate / validate
# ---------------------------------------------------------------------------


@app.command()
def consensus(
    systems: Annotated[
        list[Path], typer.Option("--systems", help="Two or more prediction JSONL files.")
    ],
    gold: Annotated[Path, typer.Option("--gold", help="Gold annotations JSONL.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    rule: Annotated[str, typer.Option("--rule", help="majority, unanimous, or any.")] = "majority",
    config: Annotated[Path | None, typer.Option("--config", help="Config YAML.")] = None,
    index: Annotated[
        Path | None, typer.Option("--index", help="Per-patient index dates CSV.")
    ] = None,
) -> None:
    """Gate two or more systems against each other and route disagreements."""
    from clinval.consensus import CONSENSUS_RULES, consensus_gate
    from clinval.io import read_annotations, read_predictions

    if rule not in CONSENSUS_RULES:
        _fail(
            ClinvalError(
                f"unknown consensus rule {rule!r}; use one of {', '.join(CONSENSUS_RULES)}"
            )
        )
        return

    if len(systems) < 2:
        _fail(
            ClinvalError(
                f"consensus needs at least two --systems files, got {len(systems)}. "
                "A one-system consensus would report a review fraction of zero and mean "
                "nothing."
            )
        )
        return

    try:
        cfg = _load_config(config)
        _needs_index(cfg, index)
        golds = read_annotations(gold)
        frames = {}
        for path in systems:
            recs = read_predictions(path)
            name = recs[0].system if recs else path.stem
            frames[name] = recs
        result = consensus_gate(
            frames,
            golds,
            rule=cast("ConsensusRule", rule),
            config=cfg,
            index=_load_index(index),
        )
    except ClinvalError as exc:
        _fail(exc)
        return

    out.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(out / "consensus_units.csv", index=False)
    result.to_review_queue().to_csv(out / "review_queue.csv", index=False)
    (out / "consensus.md").write_text(result.to_markdown(), encoding="utf-8")

    from clinval.io import write_json

    write_json(result.as_dict(), out / "consensus.json")

    console.print(result.to_markdown())
    console.print(f"\nwritten to [cyan]{out}[/cyan]")


@app.command()
def negatives(
    from_pred: Annotated[Path, typer.Option("--from", help="Positives-only predictions JSONL.")],
    notes: Annotated[Path, typer.Option("--notes", help="Notes JSONL defining the note universe.")],
    out: Annotated[Path, typer.Option("--out", help="Output predictions JSONL.")],
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated target labels.")
    ] = None,
) -> None:
    """Materialize explicit negatives for (note, label) pairs with no prediction.

    clinval never reads silence as a negative, because it cannot tell a
    considered negative from an unprocessed note. This command is the opt-in
    for callers who know the note universe.
    """
    from clinval.io import materialize_negatives, read_predictions, write_predictions

    try:
        preds = read_predictions(from_pred)
        label_list = tuple(x.strip() for x in labels.split(",") if x.strip()) if labels else None
        filled = materialize_negatives(preds, labels=label_list, notes_path=notes)
        n = write_predictions(filled, out)
    except ClinvalError as exc:
        _fail(exc)
        return

    console.print(
        f"{len(preds)} input → [bold]{n}[/bold] output records "
        f"({n - len(preds)} explicit negatives added) → [cyan]{out}[/cyan]"
    )


@app.command()
def adjudicate(
    gold: Annotated[Path, typer.Option("--gold", help="Multi-annotator annotations JSONL.")],
    out: Annotated[Path, typer.Option("--out", help="Output single-gold JSONL.")],
    rule: Annotated[
        str, typer.Option("--rule", help="agreement, majority, or first.")
    ] = "agreement",
) -> None:
    """Collapse multi-annotator records into a single gold file.

    The default ``agreement`` rule drops pairs the annotators disagreed on: a
    disagreement is not a gold standard, and silently picking a side would hide
    the very uncertainty a validation study is meant to quantify.
    """
    from clinval.aggregate import adjudicate as do_adjudicate
    from clinval.io import read_annotations, write_annotations

    try:
        recs = read_annotations(gold)
        resolved = do_adjudicate(recs, rule=rule)
        n = write_annotations(resolved, out)
    except ClinvalError as exc:
        _fail(exc)
        return

    annotators = sorted({a.annotator for a in recs})
    pairs = len({(a.note_id, a.label) for a in recs})
    dropped = pairs - n
    console.print(
        f"{len(recs)} records from {len(annotators)} annotator(s) "
        f"({', '.join(annotators)}) over {pairs} (note, label) pairs"
    )
    console.print(f"→ [bold]{n}[/bold] adjudicated records written to [cyan]{out}[/cyan]")
    if dropped > 0:
        console.print(
            f"[yellow]{dropped} pair(s) dropped[/yellow] under rule '{rule}' because the "
            "annotators did not agree. They are not gold and are excluded rather than "
            "resolved by guessing."
        )


@app.command()
def validate(
    pred: Annotated[Path | None, typer.Option("--pred", help="Predictions JSONL to check.")] = None,
    gold: Annotated[Path | None, typer.Option("--gold", help="Annotations JSONL to check.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Config YAML to check.")] = None,
    index: Annotated[Path | None, typer.Option("--index", help="Index-date CSV to check.")] = None,
) -> None:
    """Check input files parse and are internally consistent, then stop.

    Useful as a fast pre-flight in a pipeline: it reports the first malformed
    row with its line number and exits non-zero, without running an evaluation.
    """
    from clinval.io import read_annotations, read_index_dates, read_predictions

    checked = 0
    try:
        if config is not None:
            cfg = EvalConfig.from_yaml(config)
            console.print(
                f"[green]ok[/green] config: unit={cfg.unit.value}, split={cfg.split.value}, "
                f"{len(cfg.metrics)} metric(s), hash {cfg.content_hash()[:12]}…"
            )
            checked += 1
        if pred is not None:
            recs = read_predictions(pred)
            systems = sorted({r.system for r in recs})
            console.print(
                f"[green]ok[/green] predictions: {len(recs)} records, "
                f"{len(systems)} system(s) ({', '.join(systems)}), "
                f"{len({r.label for r in recs})} label(s)"
            )
            checked += 1
        if gold is not None:
            recs_g = read_annotations(gold)
            raters = sorted({r.annotator for r in recs_g})
            n_adj = sum(1 for r in recs_g if r.adjudicated)
            console.print(
                f"[green]ok[/green] annotations: {len(recs_g)} records, "
                f"{len(raters)} annotator(s) ({', '.join(raters)}), {n_adj} adjudicated"
            )
            if len(raters) > 1 and n_adj == 0:
                console.print(
                    "[yellow]warning:[/yellow] multiple annotators and no adjudicated "
                    "records. Run `clinval adjudicate` before evaluating."
                )
            checked += 1
        if index is not None:
            idx = read_index_dates(index)
            console.print(
                f"[green]ok[/green] index dates: {len(idx)} patient(s), "
                f"{idx['index_date'].min().date()} to {idx['index_date'].max().date()}"
            )
            checked += 1
    except ClinvalError as exc:
        _fail(exc)
        return

    if checked == 0:
        _fail(
            ClinvalError("nothing to validate; pass at least one of --pred/--gold/--config/--index")
        )


@app.command()
def version() -> None:
    """Print the installed clinval version."""
    from clinval import __version__

    console.print(__version__)


# ---------------------------------------------------------------------------
# shared output
# ---------------------------------------------------------------------------


def _print_summary(report, written: dict[str, Path]) -> None:
    """Print the headline numbers and where the artifacts landed."""
    import math

    # Only a few headline metrics fit a terminal; the rest are in the report.
    shown = [m for m in _SUMMARY_METRICS if m in report.config.metrics][:_MAX_SUMMARY_METRICS]
    if not shown:
        shown = list(report.config.metrics)[:_MAX_SUMMARY_METRICS]
    omitted = [m for m in report.config.metrics if m not in shown]

    table = Table(title=f"clinval · {report.config.unit.value} units", show_lines=False)
    table.add_column("system", no_wrap=True, overflow="fold")
    table.add_column("n", justify="right")
    for metric in shown:
        table.add_column(metric.replace("_", " "), justify="right")

    for system in report.metrics.systems():
        rep = report.metrics[system]
        row = [system, str(rep.n_units)]
        for metric in shown:
            iv = rep.metrics.get(metric)
            row.append("n/a" if iv is None or math.isnan(iv.point) else f"{iv.point:.3f}")
        table.add_row(*row)
    console.print(table)
    if omitted:
        console.print(
            f"[dim]{len(omitted)} further metric(s) and all confidence intervals are in "
            f"the report: {', '.join(omitted)}[/dim]"
        )

    if report.errors.counts:
        mech = Table(title="errors by mechanism", show_lines=False)
        mech.add_column("mechanism")
        mech.add_column("n", justify="right")
        for name, n in sorted(report.errors.counts.items(), key=lambda kv: -kv[1]):
            mech.add_row(name, str(n))
        console.print(mech)

    for w in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    if report.frozen is None and report.config.split is Split.DEV:
        console.print(
            "[yellow]note:[/yellow] this run used no frozen config, so the numbers above "
            "are in-sample. Use tune → freeze → evaluate for an out-of-sample result."
        )

    console.print(f"\nreport: [cyan]{written.get('html')}[/cyan]")
    console.print(f"metrics: [cyan]{written.get('metrics_json')}[/cyan]")
    console.print(f"errors: [cyan]{written.get('errors_csv')}[/cyan]")


def main() -> None:
    """Entry point for ``python -m clinval``."""
    app()


if __name__ == "__main__":
    main()
