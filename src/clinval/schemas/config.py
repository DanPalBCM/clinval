"""Evaluation configuration and the freeze manifest.

The config is the whole experiment: what unit to score on, how to match spans
and codes, which metrics to report, how to bootstrap. It is deliberately a
single serializable object, because the frozen-dev protocol (SPEC §4.5) hashes
it and that hash is what makes a test-set result auditable.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clinval.exceptions import ConfigError

_DURATION_RE = re.compile(r"^(\d+)\s*([DWMY])$", re.IGNORECASE)

#: Metrics clinval knows how to compute. Anything else is a config error at
#: load time rather than a KeyError halfway through a bootstrap.
KNOWN_METRICS = (
    "cohen_kappa",
    "pabak",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "accuracy",
    "f1",
    "prevalence",
)


class Split(StrEnum):
    """Which half of the data a config is pointed at.

    This is the field the frozen-dev protocol guards on: tuning is legal only
    on ``dev``, and evaluating ``test`` requires a frozen config.
    """

    DEV = "dev"
    TEST = "test"


class SpanMode(StrEnum):
    """How two spans are compared."""

    EXACT = "exact"
    OVERLAP = "overlap"
    FUZZY = "fuzzy"


class Unit(StrEnum):
    """The row that metrics are computed over."""

    NOTE = "note"
    PATIENT = "patient"
    PATIENT_WINDOW = "patient_window"


class TieBreak(StrEnum):
    """Which note represents a window when several in it are positive."""

    EARLIEST = "earliest"
    LATEST = "latest"
    HIGHEST_CONFIDENCE = "highest_confidence"


class Duration(BaseModel):
    """A calendar duration such as ``6M`` or ``30D``.

    Months and years are calendar-based, not fixed multiples of 30 or 365 days,
    because a "6-month window" in a chart-review protocol means six calendar
    months. Day and week durations are exact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(gt=0)
    unit: Literal["D", "W", "M", "Y"]

    @classmethod
    def parse(cls, text: str | Duration) -> Duration:
        """Parse ``"6M"`` into ``Duration(n=6, unit="M")``.

        Raises:
            ConfigError: For any string clinval cannot turn into a positive
                duration, including a well-formed but zero-length one like
                ``"0M"``. The error type is uniform so a caller never has to
                catch a pydantic exception for bad user input.
        """
        if isinstance(text, Duration):
            return text
        raw = str(text).strip()
        m = _DURATION_RE.match(raw)
        if not m:
            raise ConfigError(
                f"cannot parse duration {text!r}; expected forms like '30D', '2W', '6M', '1Y'"
            )
        n = int(m.group(1))
        if n < 1:
            raise ConfigError(
                f"cannot parse duration {text!r}; the length must be at least 1, got {n}"
            )
        return cls(n=n, unit=m.group(2).upper())  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"{self.n}{self.unit}"

    def to_offset(self) -> Any:
        """Return an equivalent :class:`pandas.DateOffset`."""
        import pandas as pd

        mapping = {
            "D": {"days": self.n},
            "W": {"weeks": self.n},
            "M": {"months": self.n},
            "Y": {"years": self.n},
        }
        return pd.DateOffset(**mapping[self.unit])


class Window(BaseModel):
    """A time window anchored to a per-patient date.

    ``size`` is the window length; ``n_windows`` is how many consecutive windows
    to lay down starting at the anchor. The default of one window is the common
    "did this happen within 6 months of diagnosis" question.

    Window intervals are half-open ``[start, end)``. A note falling exactly on a
    boundary therefore lands in the *later* window, deterministically, with no
    double counting.
    """

    model_config = ConfigDict(extra="forbid")

    size: Duration = Field(default=Duration(n=6, unit="M"))
    anchor: Literal["index_date"] = Field(
        default="index_date",
        description="Only index_date is supported in v1; the column comes from the index file.",
    )
    n_windows: int = Field(default=1, gt=0, le=64)
    tie_break: TieBreak = TieBreak.EARLIEST
    include_prior: bool = Field(
        default=False,
        description="When true, also lay down one window immediately before the anchor.",
    )

    @field_validator("size", mode="before")
    @classmethod
    def _coerce_size(cls, v: object) -> object:
        return Duration.parse(v) if isinstance(v, str) else v


class OntologyMatching(BaseModel):
    """How codes in one terminology are compared."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    fuzzy: bool = Field(default=False, description="Allow fuzzy display-name matching.")
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    prefix_depth: int | None = Field(
        default=None,
        description="For hierarchical codes, compare only the first N characters (ICD-10 chapters).",
    )


class SpanMatching(BaseModel):
    """How predicted and annotated spans are compared."""

    model_config = ConfigDict(extra="forbid")

    mode: SpanMode = SpanMode.OVERLAP
    min_overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum Jaccard overlap for mode=overlap.",
    )
    fuzzy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    require_span: bool = Field(
        default=False,
        description="When true, a positive prediction without a span counts as unsupported.",
    )


#: Span-policy keys accepted at the top level of ``Matching`` as a shorthand,
#: so ``Matching(span="overlap", min_overlap=0.5)`` works as documented in
#: SPEC §5.2 without the caller spelling out the nested ``span`` mapping.
_SPAN_SHORTHAND_KEYS = ("min_overlap", "fuzzy_threshold", "require_span")


class Matching(BaseModel):
    """The full matching policy: spans plus per-terminology code rules.

    Accepts a flat shorthand for the span policy::

        Matching(span="overlap", min_overlap=0.5)

    which is equivalent to::

        Matching(span=SpanMatching(mode="overlap", min_overlap=0.5))
    """

    model_config = ConfigDict(extra="forbid")

    span: SpanMatching = Field(default_factory=SpanMatching)
    ontology: dict[str, OntologyMatching] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _lift_span_shorthand(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        span = out.get("span", {})
        if isinstance(span, str):
            span = {"mode": span}
        elif isinstance(span, SpanMatching):
            span = span.model_dump()
        elif isinstance(span, dict):
            span = dict(span)
        else:
            return data
        for key in _SPAN_SHORTHAND_KEYS:
            if key in out:
                span.setdefault(key, out.pop(key))
        out["span"] = span
        return out


class Bootstrap(BaseModel):
    """Clustered bootstrap settings.

    ``cluster_by`` names the column that defines a resampling cluster. Patient
    is the right default: two windows from the same patient are not independent,
    and resampling rows instead of patients produces intervals that are too
    narrow.
    """

    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=2000, ge=1, le=200_000)
    cluster_by: str = "patient_id"
    seed: int = 0
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class CostModel(BaseModel):
    """USD per 1,000 tokens, by direction."""

    model_config = ConfigDict(extra="forbid")

    input_per_1k: float = Field(default=0.0, ge=0.0)
    output_per_1k: float = Field(default=0.0, ge=0.0)
    currency: str = "USD"


class FreezeGrid(BaseModel):
    """The search space ``clinval tune`` is allowed to explore.

    Only parameters listed here may be tuned. Anything else in the config is
    fixed by hand, which keeps the dev protocol honest: the grid is declared
    before tuning, not chosen after seeing the results.
    """

    model_config = ConfigDict(extra="forbid")

    grid: dict[str, list[Any]] = Field(default_factory=dict)
    objective: str = Field(
        default="cohen_kappa",
        description="Metric maximized during tuning; must be in KNOWN_METRICS.",
    )

    @model_validator(mode="after")
    def _check_objective(self) -> FreezeGrid:
        if self.objective not in KNOWN_METRICS:
            raise ConfigError(
                f"unknown tuning objective {self.objective!r}; known: {', '.join(KNOWN_METRICS)}"
            )
        for key in self.grid:
            if key not in TUNABLE_PARAMETERS:
                raise ConfigError(
                    f"parameter {key!r} is not tunable; tunable: {', '.join(TUNABLE_PARAMETERS)}"
                )
        return self


#: Dotted paths into :class:`EvalConfig` that ``clinval tune`` may vary.
TUNABLE_PARAMETERS = (
    "min_overlap",
    "fuzzy_threshold",
    "span_mode",
    "consensus",
    "require_span",
)


class EvalConfig(BaseModel):
    """Everything needed to turn predictions plus annotations into metrics."""

    model_config = ConfigDict(extra="forbid")

    split: Split = Split.DEV
    unit: Unit = Unit.PATIENT_WINDOW
    labels: list[str] = Field(
        default_factory=list,
        description="Labels to score. Empty means every label present in the gold file.",
    )
    matching: Matching = Field(default_factory=Matching)
    window: Window = Field(default_factory=Window)
    metrics: list[str] = Field(default_factory=lambda: list(KNOWN_METRICS))
    bootstrap: Bootstrap = Field(default_factory=Bootstrap)
    cost: CostModel = Field(default_factory=CostModel)
    consensus: Literal["majority", "unanimous", "any"] = "majority"
    freeze: FreezeGrid = Field(default_factory=FreezeGrid)

    @field_validator("metrics")
    @classmethod
    def _check_metrics(cls, v: list[str]) -> list[str]:
        unknown = [m for m in v if m not in KNOWN_METRICS]
        if unknown:
            raise ConfigError(
                f"unknown metric(s) {unknown}; known metrics: {', '.join(KNOWN_METRICS)}"
            )
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalConfig:
        if self.unit is Unit.NOTE and self.window.n_windows != 1:
            # Not fatal, but it means something the user did not ask for: windows
            # are ignored entirely at note level.
            object.__setattr__(self, "window", self.window.model_copy(update={"n_windows": 1}))
        return self

    # -- serialization -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load a config from a YAML file, mapping SPEC §5.3 shorthands."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalConfig:
        """Build a config from a plain dict, accepting the documented shorthands."""
        data = dict(raw)
        # §5.3 writes the span rule as `matching.span: {mode: ..., min_overlap: ...}`
        # and the ontology rules as a mapping keyed by terminology name.
        matching = data.get("matching")
        if isinstance(matching, dict):
            matching = dict(matching)
            onto = matching.get("ontology")
            if isinstance(onto, dict):
                matching["ontology"] = {
                    k.upper(): (v if isinstance(v, dict) else {"enabled": bool(v)})
                    for k, v in onto.items()
                }
            data["matching"] = matching
        try:
            return cls.model_validate(data)
        except ConfigError:
            raise
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(f"invalid configuration: {exc}") from exc

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to YAML, round-trippable through :meth:`from_yaml`."""
        Path(path).write_text(yaml.safe_dump(self.to_canonical_dict(), sort_keys=True))

    def to_canonical_dict(self) -> dict[str, Any]:
        """A plain, sorted, JSON-safe dict -- the thing that gets hashed.

        Canonicalization matters: two configs that differ only in key order or
        in ``6M`` versus ``Duration(n=6, unit='M')`` must hash identically, or
        the freeze check would fire on a no-op edit.
        """
        return json.loads(self.model_dump_json())

    def content_hash(self) -> str:
        """SHA-256 over the canonical JSON form of this config."""
        blob = json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_overrides(self, **overrides: Any) -> EvalConfig:
        """Return a copy with tunable parameters replaced.

        Accepts the flat names in :data:`TUNABLE_PARAMETERS` and routes them to
        their nested homes, so a tuning grid does not need to know the schema
        layout.
        """
        data = self.model_dump()
        for key, value in overrides.items():
            if key == "min_overlap":
                data["matching"]["span"]["min_overlap"] = value
            elif key == "fuzzy_threshold":
                data["matching"]["span"]["fuzzy_threshold"] = value
            elif key == "span_mode":
                data["matching"]["span"]["mode"] = value
            elif key == "require_span":
                data["matching"]["span"]["require_span"] = value
            elif key == "consensus":
                data["consensus"] = value
            else:
                raise ConfigError(
                    f"parameter {key!r} is not tunable; tunable: {', '.join(TUNABLE_PARAMETERS)}"
                )
        return EvalConfig.model_validate(data)


class FrozenConfig(BaseModel):
    """A config plus the hash that proves it has not changed since freezing.

    Written by ``clinval freeze``, verified by ``clinval evaluate``. The hash
    covers the config only -- not the timestamp or commit -- so re-freezing an
    identical config is idempotent.
    """

    model_config = ConfigDict(extra="forbid")

    config: EvalConfig
    config_hash: str
    frozen_at: datetime
    git_commit: str | None = None
    clinval_version: str
    tuning_summary: dict[str, Any] | None = Field(
        default=None,
        description="What tune() searched and chose, for the audit trail.",
    )

    @classmethod
    def create(
        cls,
        config: EvalConfig,
        *,
        git_commit: str | None = None,
        tuning_summary: dict[str, Any] | None = None,
    ) -> FrozenConfig:
        """Freeze ``config``, computing its hash now."""
        from clinval import __version__

        return cls(
            config=config,
            config_hash=config.content_hash(),
            frozen_at=datetime.now(UTC),
            git_commit=git_commit,
            clinval_version=__version__,
            tuning_summary=tuning_summary,
        )

    def verify(self) -> None:
        """Raise :class:`FrozenConfigError` if the config no longer matches its hash."""
        from clinval.exceptions import FrozenConfigError

        actual = self.config.content_hash()
        if actual != self.config_hash:
            raise FrozenConfigError(
                "frozen config failed verification: the config body was modified after "
                f"freezing.\n  recorded hash: {self.config_hash}\n  actual hash:   {actual}\n"
                "Re-run `clinval freeze` if the change was intentional."
            )
