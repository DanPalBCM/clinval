"""Deterministic synthetic data. The only data clinval ships or tests against."""

from clinval.synth.generator import (
    EPOCH,
    LABEL_CODES,
    SUPPORTED_LABELS,
    SyntheticDataset,
    generate,
)

__all__ = [
    "EPOCH",
    "LABEL_CODES",
    "SUPPORTED_LABELS",
    "SyntheticDataset",
    "generate",
]
