"""Textual scene description and the privacy filter that guards it.

APL never ships the raw frame downstream: the deliverable is a *textual*
description built only from evidentiary predicates.  Two mechanisms keep that
description privacy-preserving:

1. the predicate inventory itself contains no identity attributes - predicates
   describe road state, vehicle state and signage, never who is in the frame;
2. :func:`sanitize` strips any residual personally identifiable information
   (plate strings, faces, names, phone numbers, ...) that a free-form caption
   or an upstream OCR module might have introduced, and reports what it
   removed so the privacy metrics can be computed on real counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# PII patterns
# --------------------------------------------------------------------------- #
PII_PATTERNS: Dict[str, re.Pattern] = {
    "license_plate": re.compile(
        r"\b(?=[A-Z0-9-]{5,9}\b)(?=.*[A-Z])(?=.*\d)[A-Z]{1,3}[- ]?\d{1,4}[- ]?[A-Z0-9]{0,3}\b"
    ),
    "plate_mention": re.compile(
        r"\b(?:license|licence|number)\s+plate\s+(?:reads|shows|is)?\s*[\"']?[A-Z0-9- ]{3,10}",
        re.IGNORECASE,
    ),
    "face": re.compile(
        r"\b(face|facial features|facial expression|beard|moustache|mustache|"
        r"hairstyle|skin tone|wearing glasses)\b",
        re.IGNORECASE,
    ),
    "identity": re.compile(
        r"\b(named|his name|her name|their name|mr\.|mrs\.|ms\.|driver's name)\b",
        re.IGNORECASE,
    ),
    "demographics": re.compile(
        r"\b(man|woman|boy|girl|elderly|teenager|caucasian|asian|black man|"
        r"white man|hispanic)\b",
        re.IGNORECASE,
    ),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "vin": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    "street_address": re.compile(
        r"\b\d{1,5}\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b"
    ),
}

REDACTION = {
    "license_plate": "[plate redacted]",
    "plate_mention": "[plate redacted]",
    "face": "person",
    "identity": "a road user",
    "demographics": "a pedestrian",
    "phone": "[contact redacted]",
    "email": "[contact redacted]",
    "vin": "[vin redacted]",
    "street_address": "[location redacted]",
}


@dataclass
class SanitizationReport:
    text: str
    removed: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def num_removed(self) -> int:
        return sum(len(v) for v in self.removed.values())

    @property
    def clean(self) -> bool:
        return self.num_removed == 0


def sanitize(text: str, patterns: Optional[Dict[str, re.Pattern]] = None
             ) -> SanitizationReport:
    """Redact PII from a generated description."""
    patterns = patterns or PII_PATTERNS
    removed: Dict[str, List[str]] = {}
    out = text
    for name, pattern in patterns.items():
        found = pattern.findall(out)
        if found:
            removed[name] = [
                f if isinstance(f, str) else " ".join(x for x in f if x)
                for f in found
            ]
            out = pattern.sub(REDACTION.get(name, "[redacted]"), out)
    return SanitizationReport(text=out, removed=removed)


def detect_pii(text: str, patterns: Optional[Dict[str, re.Pattern]] = None
               ) -> Dict[str, List[str]]:
    return sanitize(text, patterns).removed


# --------------------------------------------------------------------------- #
# description composition
# --------------------------------------------------------------------------- #
@dataclass
class SceneDescription:
    """The textual representation ``t = f_theta(x)`` handed to downstream tasks."""

    text: str
    observed: List[str] = field(default_factory=list)
    uncertain: List[str] = field(default_factory=list)
    absent: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    sanitization: Optional[SanitizationReport] = None
    #: the affirmative part of ``text`` only - what the scene *does* show.
    #: Semantic Utility is measured on this, so that listing a predicate as
    #: "not observed" cannot be mistaken for retaining its evidence.
    positive_text: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


class DescriptionComposer:
    """Builds a readable description out of the structured evidence."""

    def __init__(self, bank, observed_threshold: float = 0.6,
                 absent_threshold: float = 0.35):
        self.bank = bank
        self.observed_threshold = observed_threshold
        self.absent_threshold = absent_threshold

    # ------------------------------------------------------------------ #
    def compose(self, description, engine_output=None,
                index: int = 0) -> SceneDescription:
        satisfaction = description.satisfaction[index].detach()
        observed, uncertain, absent = [], [], []
        for i, predicate in enumerate(self.bank.predicates):
            value = float(satisfaction[i])
            if value >= self.observed_threshold:
                observed.append(predicate.text)
            elif value <= self.absent_threshold:
                absent.append(predicate.text)
            else:
                uncertain.append(predicate.text)

        sentences: List[str] = []
        positive: List[str] = []
        if observed:
            sentences.append("The scene shows that " + _join(observed) + ".")
            positive.append(sentences[-1])
        else:
            sentences.append(
                "No evidentiary condition of the scenario bank is clearly "
                "observable in the scene."
            )
        if uncertain:
            sentences.append(
                "The following conditions are only partially visible: "
                + _join(uncertain) + "."
            )
            positive.append(sentences[-1])
        if absent:
            sentences.append("Not observed: " + _join(absent[:6]) + ".")

        violations: List[str] = []
        if engine_output is not None:
            violations = list(engine_output.violations)
            if violations:
                sentences.append(
                    "Rule evaluation reports "
                    + _join([v.replace("_", " ") for v in violations])
                    + "."
                )
            elif engine_output.scene_label == "U":
                sentences.append(
                    "Rule evaluation reports insufficient evidence for any "
                    "scenario in the bank."
                )
            else:
                sentences.append("Rule evaluation reports no violation.")

        report = sanitize(" ".join(sentences))
        positive_report = sanitize(" ".join(positive))
        return SceneDescription(
            text=report.text,
            positive_text=positive_report.text,
            observed=observed,
            uncertain=uncertain,
            absent=absent,
            violations=violations,
            sanitization=report,
        )


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
