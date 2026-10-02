"""Ontology code matching for ICD-10 and HPO.

clinval bundles a deliberately **minimal** map -- only the concepts used by the
synthetic generator, the tests, and the docs walkthrough. It is not a
terminology service and must not be mistaken for one: a real study supplies its
own crosswalk via :func:`load_ontology_map`.

Two codes can agree in three ways:

* **identical** after normalization (case, whitespace, ``HP:`` prefix form);
* **hierarchically**, when one is an ancestor of the other -- for ICD-10 this
  is a prefix test at a configured depth, since ``E11.9`` sits under ``E11``;
* **by display name**, fuzzily, when codes differ but the human-readable terms
  are the same concept. This is off by default because it is the weakest of
  the three.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from rapidfuzz import fuzz, process

from clinval.schemas.config import OntologyMatching
from clinval.schemas.prediction import Code


class ConceptEntry(TypedDict, total=False):
    """One row of a concept map.

    All keys are optional: a crosswalk-only entry may carry just ``icd10``,
    and a display-only entry just ``display``.
    """

    display: str
    synonyms: list[str]
    icd10: list[str]


_HPO_RE = re.compile(r"^HP[:_]?0*(\d{1,7})$", re.IGNORECASE)
_ICD10_RE = re.compile(r"^([A-Z])\s*(\d{2})\.?(\d*)$", re.IGNORECASE)

#: Minimal bundled concept map. Keys are normalized codes; values carry the
#: preferred display term and any synonyms used by the synthetic notes.
#: Scope is the gastrointestinal and endocrine concepts in the demo only.
BUNDLED_MAP: dict[str, ConceptEntry] = {
    "HP:0002014": {
        "display": "Diarrhea",
        "synonyms": [
            "diarrhoea",
            "loose stools",
            "watery stools",
            "frequent loose bowel movements",
        ],
        "icd10": ["K59.1", "R19.7"],
    },
    "HP:0002019": {
        "display": "Constipation",
        "synonyms": ["constipated", "hard stools", "infrequent bowel movements"],
        "icd10": ["K59.0"],
    },
    "HP:0002572": {
        "display": "Vomiting",
        "synonyms": ["emesis", "throwing up", "vomited"],
        "icd10": ["R11.10", "R11.2"],
    },
    "HP:0002018": {
        "display": "Nausea",
        "synonyms": ["nauseated", "queasy", "sick to stomach"],
        "icd10": ["R11.0"],
    },
    "HP:0002013": {
        "display": "Abdominal pain",
        "synonyms": ["stomach pain", "belly pain", "abdominal cramping"],
        "icd10": ["R10.9", "R10.84"],
    },
    "HP:0000819": {
        "display": "Diabetes mellitus",
        "synonyms": ["diabetes", "diabetic"],
        "icd10": ["E11.9", "E10.9"],
    },
}


def normalize_code(code: str, system: str | None = None) -> str:
    """Normalize a code string to clinval's canonical form.

    HPO identifiers become zero-padded ``HP:0002014``. ICD-10 codes become
    uppercase with the decimal point retained (``e119`` and ``E11.9`` both
    become ``E11.9``), because the point is the level separator and dropping it
    would make ``E1`` ``19`` indistinguishable from ``E11`` ``9``.
    """
    raw = (code or "").strip()
    if not raw:
        return ""
    sys_u = (system or "").upper()

    m = _HPO_RE.match(raw)
    if m and ("HPO" in sys_u or sys_u == "" or raw.upper().startswith("HP")):
        return f"HP:{int(m.group(1)):07d}"

    m = _ICD10_RE.match(raw.replace(" ", ""))
    if m and ("ICD" in sys_u or sys_u == ""):
        letter, pair, rest = m.group(1).upper(), m.group(2), m.group(3)
        return f"{letter}{pair}.{rest}" if rest else f"{letter}{pair}"

    return raw.upper()


def icd10_prefix(code: str, depth: int) -> str:
    """Truncate a normalized ICD-10 code to ``depth`` significant characters.

    Depth counts characters of the code with the decimal point removed, so
    depth 3 of ``E11.9`` is ``E11`` (the category) and depth 4 is ``E11.9``.
    """
    norm = normalize_code(code, "ICD-10")
    flat = norm.replace(".", "")
    truncated = flat[:depth]
    if len(truncated) > 3:
        return f"{truncated[:3]}.{truncated[3:]}"
    return truncated


@dataclass
class OntologyMap:
    """A concept map usable for code and display-name matching."""

    concepts: dict[str, ConceptEntry] = field(default_factory=dict)

    def display(self, code: str) -> str | None:
        """Preferred display term for a code, if known."""
        entry = self.concepts.get(normalize_code(code))
        return entry["display"] if entry and "display" in entry else None

    def synonyms(self, code: str) -> list[str]:
        """All surface forms for a code: its display term plus synonyms."""
        entry = self.concepts.get(normalize_code(code))
        if not entry:
            return []
        out = [entry["display"]] if entry.get("display") else []
        out.extend(str(syn) for syn in entry.get("synonyms", []))
        return [x for x in out if x]

    def crosswalk(self, code: str) -> set[str]:
        """Codes in other terminologies that the map says mean the same thing."""
        norm = normalize_code(code)
        out: set[str] = set()
        entry = self.concepts.get(norm)
        if entry:
            out.update(normalize_code(str(c), "ICD-10") for c in entry.get("icd10", []))
        # Reverse direction: an ICD-10 code pointing back at its HPO concept.
        for hpo, body in self.concepts.items():
            icds = {normalize_code(str(c), "ICD-10") for c in body.get("icd10", [])}
            if norm in icds:
                out.add(hpo)
        out.discard(norm)
        return out

    def lookup_by_text(self, text: str, *, threshold: float = 0.85) -> tuple[str, float] | None:
        """Find the best concept for a free-text term.

        Returns ``(code, score)`` for the highest-scoring surface form above
        ``threshold``, or ``None``. Used to grade an extractor that emits text
        where a code was expected.
        """
        candidates: dict[str, str] = {}
        for code in self.concepts:
            for syn in self.synonyms(code):
                candidates[syn.lower()] = code
        if not candidates or not text:
            return None
        hit = process.extractOne(text.lower(), list(candidates), scorer=fuzz.token_set_ratio)
        if hit is None:
            return None
        surface, score, _ = hit
        normalized = score / 100.0
        if normalized < threshold:
            return None
        return candidates[surface], normalized


def bundled_map() -> OntologyMap:
    """The minimal concept map shipped with clinval (demo scope only)."""
    return OntologyMap(concepts={normalize_code(k): v for k, v in BUNDLED_MAP.items()})


def load_ontology_map(path: str | Path) -> OntologyMap:
    """Load a concept map from JSON.

    Expected shape -- a mapping from code to an entry::

        {"HP:0002014": {"display": "Diarrhea",
                        "synonyms": ["loose stools"],
                        "icd10": ["K59.1"]}}
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("ontology map must be a JSON object keyed by code")
    return OntologyMap(
        concepts={normalize_code(k): cast("ConceptEntry", v) for k, v in data.items()}
    )


@dataclass(frozen=True, slots=True)
class CodeMatchResult:
    """Outcome of comparing two codes.

    ``how`` is one of ``identical``, ``crosswalk``, ``prefix``, ``fuzzy_display``,
    ``absent``, or ``mismatch`` -- the error taxonomy uses it to decide whether a
    disagreement is an ``ontology_mismatch`` or something else.
    """

    matched: bool
    how: str
    score: float = 0.0


def match_codes(
    pred: Code | None,
    gold: Code | None,
    policy: OntologyMatching | None = None,
    ontology: OntologyMap | None = None,
) -> CodeMatchResult:
    """Compare a predicted code with a gold code.

    A missing code on either side is reported as ``absent`` and counted as a
    match, because v1 treats codes as optional enrichment of a boolean label;
    requiring them is the caller's choice, expressed by filtering on ``how``.
    """
    policy = policy or OntologyMatching()
    if not policy.enabled:
        return CodeMatchResult(matched=True, how="disabled", score=1.0)
    if pred is None or gold is None:
        return CodeMatchResult(matched=True, how="absent", score=0.0)

    onto = ontology or bundled_map()
    p = normalize_code(pred.code, pred.system)
    g = normalize_code(gold.code, gold.system)

    if p == g:
        return CodeMatchResult(matched=True, how="identical", score=1.0)

    if g in onto.crosswalk(p) or p in onto.crosswalk(g):
        return CodeMatchResult(matched=True, how="crosswalk", score=1.0)

    if policy.prefix_depth is not None:
        same_system = (pred.system or "").upper() == (gold.system or "").upper()
        if same_system and icd10_prefix(p, policy.prefix_depth) == icd10_prefix(
            g, policy.prefix_depth
        ):
            return CodeMatchResult(matched=True, how="prefix", score=1.0)

    if policy.fuzzy:
        pd_, gd_ = pred.display or onto.display(p), gold.display or onto.display(g)
        if pd_ and gd_:
            # Case-folded: "Diarrhea" and "diarrhea" are the same concept, and
            # rapidfuzz would otherwise score them below a 0.85 threshold.
            score = fuzz.token_set_ratio(pd_.casefold(), gd_.casefold()) / 100.0
            if score >= policy.threshold:
                return CodeMatchResult(matched=True, how="fuzzy_display", score=score)
            return CodeMatchResult(matched=False, how="mismatch", score=score)

    return CodeMatchResult(matched=False, how="mismatch", score=0.0)
