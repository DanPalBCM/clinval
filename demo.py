#!/usr/bin/env python
"""clinval demo: the full frozen-dev protocol on the toy data in this repository.

Run it with::

    uv sync
    uv run python demo.py

Everything it reads lives in ``examples/data/`` and was produced by
``clinval.synth``, a deterministic generator with a fixed sentence bank. No real
patient record was involved at any point, and nothing here reaches the network.

The demo walks the protocol in order and then does one extra thing: it tampers
with the frozen config and shows clinval refusing to run. That refusal is the
whole point of the package, so it seemed worth demonstrating rather than
asserting.

Output lands in ``demo_output/``. Open ``demo_output/report/report.html``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import clinval as cv

DATA = Path(__file__).parent / "examples" / "data"
OUT = Path(__file__).parent / "demo_output"

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)
if not sys.stdout.isatty():  # keep piped output clean
    GREEN = YELLOW = RED = DIM = BOLD = RESET = ""


def step(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}/6] {title}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def note(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def fmt(interval: cv.Interval) -> str:
    return interval.format(3)


def cell(value: object) -> str:
    """Render a frame cell, treating pandas' nan as absent.

    ``nan`` is truthy, so an ``or`` fallback chain would print the float.
    """
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def main() -> int:
    print(f"{BOLD}clinval {cv.__version__} — demo{RESET}")
    print(f"{DIM}Synthetic data only. No PHI, no network calls.{RESET}")

    if not DATA.exists():
        print(f"{RED}error:{RESET} {DATA} is missing. Regenerate it with:")
        print("  uv run clinval synth --out examples/data --n-patients 40 --seed 0")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # ------------------------------------------------------------------
    step(1, "Load the toy data")
    # ------------------------------------------------------------------
    dev_pred = cv.read_predictions(DATA / "dev_pred.jsonl")
    dev_gold = cv.read_annotations(DATA / "dev_gold.jsonl")
    dev_index = cv.read_index_dates(DATA / "dev_index.csv")

    test_pred = cv.read_predictions(DATA / "test_pred.jsonl")
    test_gold = cv.read_annotations(DATA / "test_gold.jsonl")
    test_index = cv.read_index_dates(DATA / "test_index.csv")

    raters = cv.read_annotations(DATA / "test_gold_two_annotators.jsonl")
    config = cv.EvalConfig.from_yaml(DATA / "eval.yaml")

    systems = sorted({p.system for p in test_pred})
    ok(
        f"dev:  {len(dev_index)} patients, {len({p.note_id for p in dev_pred})} notes, {len(dev_pred)} predictions"
    )
    ok(
        f"test: {len(test_index)} patients, {len({p.note_id for p in test_pred})} notes, {len(test_pred)} predictions"
    )
    ok(f"systems under evaluation: {', '.join(systems)}")
    note(f"unit = {config.unit.value}, window = {config.window.size} x {config.window.n_windows}")

    split = cv.check_split_disjoint(dev_gold, test_gold)
    if split["ok"]:
        ok("dev and test share no patients and no notes")
    else:
        warn(f"splits overlap on {split['n_shared_patients']} patient(s)")

    # ------------------------------------------------------------------
    step(2, "Tune the thresholds on dev")
    # ------------------------------------------------------------------
    tuning = cv.tune(config, dev_pred, dev_gold, index=dev_index)
    print()
    for line in tuning.to_markdown().splitlines():
        print(f"  {line}")
    print()
    ok(f"chose {tuning.best_params} at {tuning.objective} = {tuning.best_score:.4f}")
    if tuning.is_flat:
        warn("the objective barely moved across the grid, so this choice is near-arbitrary")
    for w in tuning.warnings:
        warn(w)

    # ------------------------------------------------------------------
    step(3, "Freeze the tuned config")
    # ------------------------------------------------------------------
    frozen = cv.freeze(tuning.config, tuning=tuning)
    frozen_path = cv.write_frozen(frozen, OUT / "frozen.json")
    ok(f"hash {frozen.config_hash[:24]}…")
    ok(
        f"split locked to '{frozen.config.split.value}', manifest at {frozen_path.relative_to(Path.cwd())}"
    )
    note(f"git commit: {frozen.git_commit or 'not in a git repository'}")

    # ------------------------------------------------------------------
    step(4, "Evaluate the frozen config on held-out test data")
    # ------------------------------------------------------------------
    manifest = cv.load_frozen(frozen_path)
    ok("frozen config verified against its hash")

    report = cv.run_eval(
        test_pred,
        test_gold,
        manifest.config,
        index=test_index,
        frozen=manifest,
        annotator_records=raters,
    )
    report.split_check = split
    written = report.write_all(OUT / "report")

    print()
    print(f"  {'system':<16} {'units':>6}  {'kappa':>20} {'sensitivity':>20} {'specificity':>20}")
    print(f"  {'-' * 16} {'-' * 6}  {'-' * 20} {'-' * 20} {'-' * 20}")
    for system in report.metrics.systems():
        rep = report.metrics[system]
        print(
            f"  {system:<16} {rep.n_units:>6}  "
            f"{fmt(rep.metrics['cohen_kappa']):>20} "
            f"{fmt(rep.metrics['sensitivity']):>20} "
            f"{fmt(rep.metrics['specificity']):>20}"
        )
    print()
    note("intervals: Wilson for proportions, clustered bootstrap for kappa (clustered by patient)")

    for w in report.warnings:
        warn(w)

    # The two mock systems are built with known, different pathologies. If the
    # harness cannot tell them apart, it is not working.
    kappas = {s: report.metrics[s].metrics["cohen_kappa"].point for s in report.metrics.systems()}
    if kappas.get("careful_reader", 0) > kappas.get("negation_blind", 1):
        ok(
            "careful_reader scores above negation_blind, as designed "
            f"({kappas['careful_reader']:.3f} vs {kappas['negation_blind']:.3f})"
        )

    comparison = next((c for c in report.comparisons if c["metric"] == "cohen_kappa"), None)
    if comparison and comparison["ci_low"] is not None:
        ok(
            f"paired bootstrap on {comparison['n_shared_units']} shared units: "
            f"difference {comparison['difference']:.3f} "
            f"(95% CI {comparison['ci_low']:.3f} to {comparison['ci_high']:.3f}, "
            f"p = {comparison['p_value']:.3f})"
        )

    if report.iaa.get("pairs"):
        for pair, vals in report.iaa["pairs"].items():
            if vals.get("cohen_kappa") is not None:
                ok(
                    f"human ceiling — annotators {pair} agreed at kappa "
                    f"{vals['cohen_kappa']:.3f} over {vals['n_units']} units"
                )

    print()
    print(f"  {BOLD}Why the disagreements happened{RESET}")
    mech = report.errors.by_mechanism()
    total = int(mech["n"].sum())
    for row in mech.itertuples(index=False):
        bar = "█" * min(40, round(row.n / max(total, 1) * 40))
        label = "none found" if row.n == 0 else str(row.n)
        print(f"    {row.mechanism:<22} {label:>11}  {bar}")
    note(f"{total} tagged disagreements; every tag carries the evidence that produced it")

    example = next(
        (
            r
            for m in ("misreading", "temporal_error", "annotation_threshold")
            for r in report.errors.sample(m, n=1).to_dict("records")
        ),
        None,
    )
    if example:
        # On a false negative the system quoted nothing, so the evidence to
        # show is the annotator's span.
        evidence = cell(example["span_text"]) or cell(example["gold_span_text"])
        source = "system quote" if cell(example["span_text"]) else "annotator quote"
        print()
        print(f"  {BOLD}One tagged error, in full{RESET}")
        print(
            f"    id        {example['error_id']}  ({example['error_type']}, {example['mechanism']})"
        )
        print(f"    system    {example['system']}  label {example['label']}")
        print(f"    evidence  {evidence[:90]!r}  ({source})")
        print(f"    why       {example['mechanism_reason']}")

    cost = report.cost.per_system
    print()
    print(f"  {BOLD}Cost{RESET}")
    for name in sorted(cost):
        c = cost[name]
        shown = "unknown" if c.cost is None else f"${c.cost:.4f}"
        print(f"    {name:<22} {shown:>10}   mean latency {c.latency_mean_s or 0:.2f}s")
    note("a missing token count is reported as unknown, never as zero")

    # ------------------------------------------------------------------
    step(5, "Gate the two systems against each other")
    # ------------------------------------------------------------------
    gate = cv.consensus_gate(
        {
            "careful_reader": cv.read_predictions(DATA / "test_pred_careful_reader.jsonl"),
            "negation_blind": cv.read_predictions(DATA / "test_pred_negation_blind.jsonl"),
        },
        test_gold,
        rule="majority",
        config=manifest.config,
        index=test_index,
    )
    ok(f"{gate.n_units} units; {gate.n_review} routed to human review ({gate.review_fraction:.1%})")
    avd = gate.agreed_vs_disagreed
    if avd.get("accuracy_lift") is not None:
        ok(
            f"accuracy on agreed units {avd['agreed']['accuracy']:.3f} "
            f"vs disagreed {avd['disagreed']['accuracy']:.3f} "
            f"(lift {avd['accuracy_lift']:+.3f})"
        )
        if avd["accuracy_lift"] < 0.05:
            warn("a small lift means the gate is barely separating reliable from unreliable units")
    gate.df.to_csv(OUT / "consensus_units.csv", index=False)
    gate.to_review_queue().to_csv(OUT / "review_queue.csv", index=False)
    ok("review queue written, most-contested first")

    # ------------------------------------------------------------------
    step(6, "Tamper with the frozen config and watch clinval refuse")
    # ------------------------------------------------------------------
    tampered = OUT / "frozen_tampered.json"
    raw = json.loads(frozen_path.read_text())
    original = raw["config"]["matching"]["span"]["min_overlap"]
    raw["config"]["matching"]["span"]["min_overlap"] = 0.01
    tampered.write_text(json.dumps(raw, indent=2))
    note(f"edited min_overlap from {original} to 0.01 in a copy of the manifest")

    try:
        cv.load_frozen(tampered)
    except cv.FrozenConfigError as exc:
        ok("clinval refused to load the edited manifest")
        for line in str(exc).splitlines():
            print(f"    {DIM}{line}{RESET}")
    else:
        print(f"  {RED}✗{RESET} the tamper check did not fire — this is a bug")
        return 1

    # ------------------------------------------------------------------
    print(f"\n{BOLD}Done.{RESET} Artifacts in {OUT.name}/:")
    for name, path in sorted(written.items()):
        print(f"  {name:<22} {path.relative_to(Path.cwd())}")
    print(
        f"\nOpen {BOLD}{(OUT / 'report' / 'report.html').relative_to(Path.cwd())}{RESET} in a browser."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
