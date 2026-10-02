"""Extractor plugins. The core never calls a model; adapters live here."""

from clinval.plugins.base import ExtractorPlugin, Note, validate_predictions

__all__ = ["ExtractorPlugin", "Note", "validate_predictions"]
