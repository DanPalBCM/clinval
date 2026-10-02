"""Shared fixtures. Every test uses synthetic data and makes no network calls."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from clinval.schemas.annotation import Annotation
from clinval.schemas.config import Bootstrap, EvalConfig, Unit, Window
from clinval.schemas.prediction import Code, Prediction, RunMeta, Span
from clinval.synth import generate


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the opt-in flag for tests that need a live model."""
    parser.addoption(
        "--run-live-llm",
        action="store_true",
        default=False,
        help="Run tests marked live_llm. These need network access and an API key.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``live_llm`` tests unless explicitly opted in."""
    if config.getoption("--run-live-llm"):
        return
    skip = pytest.mark.skip(reason="needs --run-live-llm (network + API key)")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# hand-built records, for tests that need exact control
# ---------------------------------------------------------------------------

BASE = datetime(2022, 1, 10, 9, 0)


def make_pred(
    note_id: str = "N1",
    label: str = "diarrhea",
    value: bool = True,
    *,
    system: str = "sysA",
    patient_id: str | None = "P1",
    span: tuple[int, int, str | None] | None = None,
    code: tuple[str, str] | None = None,
    note_datetime: datetime | None = BASE,
    confidence: float | None = None,
    tokens: tuple[int, int] | None = None,
    latency: float | None = None,
) -> Prediction:
    """Build one prediction with terse arguments."""
    run = None
    if tokens is not None or latency is not None:
        run = RunMeta(
            model="test-model",
            prompt_version="v1",
            tokens_in=tokens[0] if tokens else None,
            tokens_out=tokens[1] if tokens else None,
            latency_s=latency,
        )
    return Prediction(
        system=system,
        note_id=note_id,
        label=label,
        value=value,
        patient_id=patient_id,
        span=Span(start=span[0], end=span[1], text=span[2]) if span else None,
        code=Code(system=code[0], code=code[1]) if code else None,
        note_datetime=note_datetime,
        confidence=confidence,
        run=run,
    )


def make_gold(
    note_id: str = "N1",
    label: str = "diarrhea",
    value: bool = True,
    *,
    annotator: str = "adjudicated",
    patient_id: str | None = "P1",
    span: tuple[int, int, str | None] | None = None,
    code: tuple[str, str] | None = None,
    note_datetime: datetime | None = BASE,
    adjudicated: bool = True,
) -> Annotation:
    """Build one gold annotation with terse arguments."""
    return Annotation(
        annotator=annotator,
        note_id=note_id,
        label=label,
        value=value,
        patient_id=patient_id,
        span=Span(start=span[0], end=span[1], text=span[2]) if span else None,
        code=Code(system=code[0], code=code[1]) if code else None,
        note_datetime=note_datetime,
        adjudicated=adjudicated,
    )


@pytest.fixture
def note_config() -> EvalConfig:
    """Note-level config with fast bootstrap settings."""
    return EvalConfig(unit=Unit.NOTE, bootstrap=Bootstrap(n=50, seed=0))


@pytest.fixture
def window_config() -> EvalConfig:
    """Patient-window config with two 6-month windows."""
    return EvalConfig(
        unit=Unit.PATIENT_WINDOW,
        window=Window(size="6M", n_windows=2),
        bootstrap=Bootstrap(n=50, seed=0),
    )


@pytest.fixture
def index_one() -> pd.DataFrame:
    """A single patient anchored on 2022-01-01."""
    return pd.DataFrame({"patient_id": ["P1"], "index_date": [pd.Timestamp("2022-01-01")]})


@pytest.fixture(scope="session")
def small_dataset():
    """A small deterministic dataset, shared across tests that only read it."""
    return generate(n_patients=12, labels=("diarrhea", "constipation"), seed=11)


@pytest.fixture(scope="session")
def split_dataset(small_dataset):
    """A dev/test split of :func:`small_dataset`, split by patient."""
    return small_dataset.split_by_patient(0.5)


@pytest.fixture
def fast_config() -> EvalConfig:
    """A config with a small bootstrap, for tests where the CI value is irrelevant."""
    return EvalConfig(
        unit=Unit.PATIENT_WINDOW,
        window=Window(size="6M", n_windows=2),
        bootstrap=Bootstrap(n=25, seed=0),
    )
