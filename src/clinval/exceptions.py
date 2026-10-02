"""Exception hierarchy for clinval.

Every error clinval raises on bad user input carries enough context to find the
offending record: a file path and a line number where one exists.
"""

from __future__ import annotations


class ClinvalError(Exception):
    """Base class for every error raised by clinval."""


class RecordError(ClinvalError):
    """A single input record could not be parsed or validated."""

    def __init__(self, message: str, *, path: str | None = None, line: int | None = None) -> None:
        self.path = path
        self.line = line
        location = ""
        if path is not None:
            location = f" [{path}"
            location += f":{line}]" if line is not None else "]"
        super().__init__(f"{message}{location}")


class DuplicateRecordError(ClinvalError):
    """The same logical record appeared more than once.

    Predictions are keyed by ``(system, note_id, label)`` and annotations by
    ``(annotator, note_id, label)``. A repeat is an error rather than a
    silent last-one-wins, because which copy survives would change the metrics.
    """


class ConfigError(ClinvalError):
    """The evaluation configuration is missing, malformed, or self-contradictory."""


class FrozenConfigError(ClinvalError):
    """A frozen config failed verification, or a frozen config was required and absent."""


class SplitGuardError(ClinvalError):
    """An operation that may only touch dev data was pointed at test data.

    Raised by the frozen-dev protocol (:mod:`clinval.freeze`): tuning on test
    data, or evaluating test data with a config that was never frozen.
    """


class AggregationError(ClinvalError):
    """Note-level records could not be aggregated to the requested unit."""


class PluginError(ClinvalError):
    """An extractor plugin is unavailable or produced invalid predictions."""
