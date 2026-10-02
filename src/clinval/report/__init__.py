"""Report generation: static HTML, Markdown, figures, and CSV exports."""

from clinval.report.builder import (
    PALETTE,
    plot_confusion,
    plot_error_mechanisms,
    plot_metrics,
    plot_per_label,
    render_html,
    render_markdown,
    write_report,
)

__all__ = [
    "PALETTE",
    "plot_confusion",
    "plot_error_mechanisms",
    "plot_metrics",
    "plot_per_label",
    "render_html",
    "render_markdown",
    "write_report",
]
