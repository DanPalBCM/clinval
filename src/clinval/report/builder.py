"""Report generation: Markdown, HTML, figures, and CSV exports.

Static output only -- matplotlib PNGs and a self-contained HTML file with no
JavaScript, so a report can be attached to a submission, opened from a file
share, or archived without a server.

The report is opinionated about ordering. Caveats come before numbers: the
provenance block and any warnings are the first thing a reader meets, because
a coverage gap or an unfrozen config changes how every number below should be
read.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # no display needed; must precede pyplot
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from clinval.errors import MECHANISMS

if TYPE_CHECKING:
    from clinval.runner import EvalReport

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Colour-blind-safe qualitative palette (Okabe-Ito), used for systems.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442")

#: Mechanism bars use one hue so the eye compares lengths, not colours.
MECHANISM_COLOR = "#4C6EF5"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


def _fmt(x: Any, digits: int = 3) -> str:
    """Format a number, rendering an undefined value as ``n/a``."""
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        return f"{x:.{digits}f}"
    return str(x)


def _fmt_ci(iv: Any, digits: int = 3) -> str:
    """Format an interval's bounds, or ``n/a`` when undefined."""
    if iv is None:
        return "n/a"
    low, high = getattr(iv, "low", float("nan")), getattr(iv, "high", float("nan"))
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return f"{low:.{digits}f}-{high:.{digits}f}"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def plot_metrics(report: EvalReport, path: Path, *, dpi: int = 150) -> Path | None:
    """Grouped bars of each metric per system, with confidence intervals.

    Error bars are asymmetric because Wilson and percentile-bootstrap
    intervals both are; drawing a symmetric bar would misstate them.
    """
    systems = report.metrics.systems()
    if not systems:
        return None
    metrics = [m for m in report.config.metrics if m != "prevalence"]
    if not metrics:
        return None

    n_s = len(systems)
    width = 0.8 / n_s
    x = range(len(metrics))
    # Kappa and PABAK can go negative, but reserving [-1, 0] when nothing is
    # negative throws away half the canvas. Scale to the data instead.
    observed: list[float] = []
    for system in systems:
        for m in metrics:
            iv = report.metrics[system].metrics.get(m)
            if iv is None:
                continue
            for v in (iv.point, iv.low):
                if isinstance(v, float) and not math.isnan(v):
                    observed.append(v)
    floor = min(observed) if observed else 0.0
    ymin = 0.0 if floor >= 0.0 else max(-1.05, floor - 0.08)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(metrics)), 4.4))
    for i, system in enumerate(systems):
        rep = report.metrics[system]
        points, lows, highs = [], [], []
        for m in metrics:
            iv = rep.metrics.get(m)
            pt = float("nan") if iv is None else iv.point
            points.append(0.0 if math.isnan(pt) else pt)
            if iv is None or math.isnan(iv.low) or math.isnan(iv.high) or math.isnan(pt):
                lows.append(0.0)
                highs.append(0.0)
            else:
                lows.append(max(0.0, pt - iv.low))
                highs.append(max(0.0, iv.high - pt))
        offsets = [xi + i * width - 0.4 + width / 2 for xi in x]
        ax.bar(
            offsets,
            points,
            width=width * 0.92,
            label=system,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.errorbar(
            offsets,
            points,
            yerr=[lows, highs],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.1,
            capsize=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=20, ha="right")
    ax.set_ylim(ymin, 1.05)
    if ymin < 0.0:
        ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_ylabel("estimate")
    ax.set_title(f"Agreement by system ({report.config.unit.value} units)")
    ax.legend(frameon=False, ncols=min(n_s, 3), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_per_label(report: EvalReport, path: Path, *, dpi: int = 150) -> Path | None:
    """Sensitivity and PPV per label, per system, with prevalence annotated.

    Prevalence is printed on the axis labels because a sensitivity of 1.00 on a
    label with three positives is a different claim from the same number on
    three hundred.
    """
    systems = report.metrics.systems()
    if not systems:
        return None
    labels = sorted({label for s in systems for label in report.metrics[s].per_label})
    if not labels:
        return None

    shown = [m for m in ("sensitivity", "ppv", "specificity") if m in report.config.metrics]
    if not shown:
        return None

    fig, axes = plt.subplots(
        1, len(shown), figsize=(4.0 * len(shown), max(3.0, 0.42 * len(labels) + 1.6)), sharey=True
    )
    axes = [axes] if len(shown) == 1 else list(axes)

    counts: dict[str, int] = {}
    for label in labels:
        for s in systems:
            conf = report.metrics[s].per_label_confusion.get(label)
            if conf:
                counts[label] = conf.n_positive_gold
                break

    ypos = range(len(labels))
    height = 0.8 / len(systems)
    for ax, metric in zip(axes, shown, strict=True):
        for i, system in enumerate(systems):
            rep = report.metrics[system]
            vals = []
            for label in labels:
                iv = rep.per_label.get(label, {}).get(metric)
                vals.append(0.0 if iv is None or math.isnan(iv.point) else iv.point)
            ax.barh(
                [y + i * height - 0.4 + height / 2 for y in ypos],
                vals,
                height=height * 0.9,
                label=system,
                color=PALETTE[i % len(PALETTE)],
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xlim(0, 1.05)
        ax.set_title(metric.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    axes[0].set_yticks(list(ypos))
    axes[0].set_yticklabels(
        [f"{lab}\n(gold+ n={counts.get(lab, 0)})" for lab in labels], fontsize=8
    )
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Per-label performance", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_error_mechanisms(report: EvalReport, path: Path, *, dpi: int = 150) -> Path | None:
    """Horizontal bars of error counts by mechanism, split by system.

    Every mechanism in the taxonomy appears, including the zeros, so a reader
    can tell "none found" from "not checked".
    """
    df = report.errors.df
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.45 * len(MECHANISMS) + 1.4)))
    systems = report.metrics.systems() or ["all"]
    ypos = range(len(MECHANISMS))
    left = [0.0] * len(MECHANISMS)

    for i, system in enumerate(systems):
        if len(df):
            sub = df[df["system"] == system] if "system" in df.columns else df
            counts = [int((sub["mechanism"] == m).sum()) for m in MECHANISMS]
        else:
            counts = [0] * len(MECHANISMS)
        ax.barh(
            list(ypos),
            counts,
            left=left,
            label=system,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
            linewidth=0.5,
            height=0.68,
        )
        left = [a + b for a, b in zip(left, counts, strict=True)]

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([m.replace("_", " ") for m in MECHANISMS])
    ax.invert_yaxis()
    ax.set_xlabel("errors")
    ax.set_title("Disagreements by mechanism")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for y, total in zip(ypos, left, strict=True):
        if total == 0:
            ax.text(0.2, y, "none", va="center", fontsize=8, color="#777777")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_confusion(report: EvalReport, path: Path, *, dpi: int = 150) -> Path | None:
    """One 2x2 confusion panel per system, with counts printed in the cells."""
    systems = report.metrics.systems()
    if not systems:
        return None
    fig, axes = plt.subplots(1, len(systems), figsize=(3.1 * len(systems), 3.2))
    axes = [axes] if len(systems) == 1 else list(axes)
    for ax, system in zip(axes, systems, strict=True):
        c = report.metrics[system].confusion
        grid = [[c.tp, c.fp], [c.fn, c.tn]]
        vmax = max(1, c.n)
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=vmax)
        for r in range(2):
            for col in range(2):
                val = grid[r][col]
                ax.text(
                    col,
                    r,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if val > vmax * 0.55 else "#1a1a1a",
                )
        ax.set_xticks([0, 1], ["gold +", "gold -"])
        ax.set_yticks([0, 1], ["pred +", "pred -"])
        ax.set_title(system, fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Confusion matrices", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _context(report: EvalReport, digits: int, figures: dict[str, str]) -> dict[str, Any]:
    """Build the template context shared by Markdown and HTML."""
    from clinval import __version__

    systems = report.metrics.systems()
    rows: list[dict[str, Any]] = []
    for system in systems:
        rep = report.metrics[system]
        for metric in report.config.metrics:
            iv = rep.metrics.get(metric)
            rows.append(
                {
                    "system": system,
                    "metric": metric.replace("_", " "),
                    "point": _fmt(None if iv is None else iv.point, digits),
                    "ci": _fmt_ci(iv, digits),
                    "method": "" if iv is None else iv.method,
                }
            )

    per_label_rows: list[dict[str, Any]] = []
    for system in systems:
        rep = report.metrics[system]
        for label in sorted(rep.per_label):
            conf = rep.per_label_confusion.get(label)
            entry = {
                "system": system,
                "label": label,
                "n": conf.n if conf else 0,
                "gold_pos": conf.n_positive_gold if conf else 0,
                "tp": conf.tp if conf else 0,
                "fp": conf.fp if conf else 0,
                "fn": conf.fn if conf else 0,
                "tn": conf.tn if conf else 0,
            }
            for metric in report.config.metrics:
                iv = rep.per_label[label].get(metric)
                entry[metric] = _fmt(None if iv is None else iv.point, digits)
            per_label_rows.append(entry)

    mech_rows = [
        {
            "mechanism": m,
            "n": int(report.errors.counts.get(m, 0)),
            "share": (
                f"{report.errors.counts.get(m, 0) / len(report.errors):.1%}"
                if len(report.errors)
                else "0.0%"
            ),
        }
        for m in MECHANISMS
    ]

    def _text(*candidates: Any) -> str:
        """First non-empty, non-null candidate as a string.

        Pandas hands back ``nan`` for a missing object cell, and ``nan`` is
        truthy, so a plain ``or`` chain would pass the float straight through.
        """
        for c in candidates:
            if c is None:
                continue
            if isinstance(c, float) and math.isnan(c):
                continue
            s = str(c).strip()
            if s:
                return s
        return ""

    examples: list[dict[str, Any]] = []
    for m in MECHANISMS:
        sample = report.errors.sample(m, n=2)
        for rec in sample.to_dict("records"):
            examples.append(
                {
                    "mechanism": m,
                    "error_id": _text(rec.get("error_id")),
                    "system": _text(rec.get("system")),
                    "label": _text(rec.get("label")),
                    "error_type": _text(rec.get("error_type")),
                    "span_text": _text(rec.get("span_text"), rec.get("gold_span_text"))[:160],
                    "reason": _text(rec.get("mechanism_reason")),
                }
            )

    return {
        "version": __version__,
        "config": report.config,
        "config_hash": report.config_hash,
        "frozen": report.frozen,
        "meta": report.meta,
        "systems": systems,
        "metric_rows": rows,
        "metric_names": [m.replace("_", " ") for m in report.config.metrics],
        "metric_keys": list(report.config.metrics),
        "per_label_rows": per_label_rows,
        "mechanism_rows": mech_rows,
        "error_examples": examples,
        "n_errors": len(report.errors),
        "n_units": report.n_units,
        "coverage": report.aligned.coverage(),
        "warnings": report.warnings,
        "comparisons": report.comparisons,
        "iaa": report.iaa,
        "split_check": report.split_check,
        "cost_markdown": report.cost.to_markdown(),
        "cost": report.cost.as_dict(),
        "tuning": report.frozen.tuning_summary if report.frozen else None,
        "figures": figures,
        "fmt": _fmt,
        "digits": digits,
        "confusions": {s: report.metrics[s].confusion.as_dict() for s in systems},
    }


def render_markdown(
    report: EvalReport, *, digits: int = 3, figures: dict[str, str] | None = None
) -> str:
    """Render the Markdown report."""
    ctx = _context(report, digits, figures or {})
    return _env().get_template("report.md.j2").render(**ctx)


def render_html(
    report: EvalReport, *, digits: int = 3, figures: dict[str, str] | None = None
) -> str:
    """Render the HTML report."""
    ctx = _context(report, digits, figures or {})
    return _env().get_template("report.html.j2").render(**ctx)


def write_report(
    report: EvalReport,
    outdir: str | Path,
    *,
    digits: int = 3,
    figures: bool = True,
) -> dict[str, Path]:
    """Write every report artifact into ``outdir``.

    Produces ``report.html``, ``report.md``, ``metrics.json``, ``errors.csv``,
    ``confusion.csv``, ``units.csv``, and a ``figures/`` directory.

    Returns:
        A mapping of artifact name to path.
    """
    from clinval.errors import confusion_export

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    written: dict[str, Path] = {}
    figure_rel: dict[str, str] = {}

    if figures:
        figdir.mkdir(parents=True, exist_ok=True)
        for name, fn in (
            ("metrics", plot_metrics),
            ("per_label", plot_per_label),
            ("error_mechanisms", plot_error_mechanisms),
            ("confusion", plot_confusion),
        ):
            target = figdir / f"{name}.png"
            produced = fn(report, target)
            if produced is not None:
                written[f"figure_{name}"] = produced
                figure_rel[name] = f"figures/{name}.png"

    (out / "report.md").write_text(
        render_markdown(report, digits=digits, figures=figure_rel), encoding="utf-8"
    )
    written["markdown"] = out / "report.md"

    (out / "report.html").write_text(
        render_html(report, digits=digits, figures=figure_rel), encoding="utf-8"
    )
    written["html"] = out / "report.html"

    report.write_json(out / "metrics.json")
    written["metrics_json"] = out / "metrics.json"

    report.errors.to_csv(str(out / "errors.csv"))
    written["errors_csv"] = out / "errors.csv"

    confusion_export(report.windowed).to_csv(out / "confusion.csv", index=False)
    written["confusion_csv"] = out / "confusion.csv"

    report.windowed.df.to_csv(out / "units.csv", index=False)
    written["units_csv"] = out / "units.csv"

    report.metrics.to_dataframe().to_csv(out / "metrics.csv", index=False)
    written["metrics_csv"] = out / "metrics.csv"

    return written
