from __future__ import annotations

import importlib.resources as resources
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import yaml

from .policies import MaskingPolicy

_REGEX_RULES = (
    ("EMAIL", r"\S+@\S+"),
    ("ADDRESS", r"\b\d+\s+[A-Z][a-zA-Z]+(\s+[A-Z][a-zA-Z]+)*\s+(Road|Street|Avenue|Lane|Marg)\b"),
    ("PHONE", r"\b\d{10}\b"),
    ("AADHAAR", r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    ("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    ("HEALTH_ID", r"\b\d{8,16}\b"),
    ("PATIENT_ID", r"\b(?:MRN|UHID|PID)\s*:?[\s-]*\d{4,14}\b"),
    ("DOB", r"\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b"),
)

_SPACY_LABELS = ("PERSON", "GPE", "LOC", "FAC", "ORG")
_PLACEHOLDERS = {
    "PERSON": "[NAME_MASKED]",
    "GPE": "[ADDRESS_MASKED]",
    "LOC": "[ADDRESS_MASKED]",
    "FAC": "[ADDRESS_MASKED]",
    "ORG": "[ORG_MASKED]",
    "EMAIL": "[EMAIL_MASKED]",
    "PHONE": "[PHONE_MASKED]",
    "AADHAAR": "[AADHAAR_MASKED]",
    "PAN": "[PAN_MASKED]",
    "HEALTH_ID": "[HEALTH_ID_MASKED]",
    "PATIENT_ID": "[PATIENT_ID_MASKED]",
    "DOB": "[DOB_MASKED]",
    "ADDRESS": "[ADDRESS_MASKED]",
}

_DEFAULT_DETECTORS = None
_DOMAIN_CONFIG_PACKAGE = "secure_rag.domain_configs"
_nlp = None


@dataclass(frozen=True)
class DetectedEntity:
    start: int
    end: int
    text: str
    label: str
    source: str
    priority: int
    order: int = 0


class EntityDetector(ABC):
    name = "detector"
    priority = 0

    @abstractmethod
    def detect(self, text: str) -> Iterable[DetectedEntity]:
        raise NotImplementedError


class RegexDetector(EntityDetector):
    name = "regex"
    priority = 0

    def __init__(self, rules: Optional[Sequence[Tuple[str, str]]] = None):
        self.rules = tuple(rules or _REGEX_RULES)

    def detect(self, text: str) -> Iterable[DetectedEntity]:
        for rule_index, (label, pattern) in enumerate(self.rules):
            for match in re.finditer(pattern, text):
                yield DetectedEntity(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    label=label,
                    source=self.name,
                    priority=self.priority,
                    order=rule_index,
                )


class SpaCyDetector(EntityDetector):
    name = "spacy"
    priority = 2

    def __init__(self, labels: Optional[Sequence[str]] = None):
        self.labels = tuple(labels or _SPACY_LABELS)

    def detect(self, text: str) -> Iterable[DetectedEntity]:
        try:
            nlp = _get_nlp()
            doc = nlp(text)
        except Exception:
            return []

        for index, ent in enumerate(doc.ents):
            if ent.label_ not in self.labels:
                continue
            yield DetectedEntity(
                start=ent.start_char,
                end=ent.end_char,
                text=ent.text,
                label=ent.label_,
                source=self.name,
                priority=self.priority,
                order=index,
            )


class DomainDetector(EntityDetector):
    name = "domain"
    priority = 1

    def __init__(self, source: Union[str, Path]):
        self.source = str(source)
        config = _load_domain_config(source)
        self.config_name = str(config.get("name") or Path(str(source)).stem or "domain")
        self._term_rules = []
        self._pattern_rules = []

        terms = config.get("terms") or {}
        if not isinstance(terms, dict):
            raise ValueError(f"{source}: 'terms' must be a mapping of label -> list[str]")
        for label, values in terms.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{source}: domain labels must be non-empty strings")
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError(f"{source}: 'terms.{label}' must be a list of non-empty strings")
            for value in values:
                pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE)
                self._term_rules.append((str(label).upper(), pattern))

        patterns = config.get("patterns") or {}
        if not isinstance(patterns, dict):
            raise ValueError(f"{source}: 'patterns' must be a mapping of label -> list[str]")
        for label, values in patterns.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{source}: pattern labels must be non-empty strings")
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError(f"{source}: 'patterns.{label}' must be a list of non-empty strings")
            for value in values:
                self._pattern_rules.append((str(label).upper(), re.compile(value, re.IGNORECASE)))

    def detect(self, text: str) -> Iterable[DetectedEntity]:
        for term_index, (label, pattern) in enumerate(self._term_rules):
            for match in pattern.finditer(text):
                yield DetectedEntity(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    label=label,
                    source=f"{self.name}:{self.config_name}",
                    priority=self.priority,
                    order=term_index,
                )

        for pattern_index, (label, pattern) in enumerate(self._pattern_rules):
            for match in pattern.finditer(text):
                yield DetectedEntity(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    label=label,
                    source=f"{self.name}:{self.config_name}",
                    priority=self.priority,
                    order=10_000 + pattern_index,
                )


class MedicalDetector(DomainDetector):
    def __init__(self):
        super().__init__("medical")


def _get_nlp():
    global _nlp
    if _nlp is None:
        import os

        import spacy

        model = os.getenv("SPACY_MODEL", "en_core_web_sm")
        _nlp = spacy.load(model)
    return _nlp


def _load_domain_config(source: Union[str, Path]) -> dict:
    path = Path(str(source))
    if path.suffix.lower() in {".yaml", ".yml"} and path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data or {}

    resource = resources.files(_DOMAIN_CONFIG_PACKAGE).joinpath(f"{source}.yaml")
    if resource.is_file():
        with resource.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data or {}

    raise FileNotFoundError(f"domain detector config not found: {source}")


def load_detector_stack(profile: Optional[str] = None) -> Tuple[EntityDetector, ...]:
    if profile is None:
        return _default_detectors()

    return (RegexDetector(), DomainDetector(profile), SpaCyDetector())


def detect_entities(
    text: str,
    detectors: Optional[Sequence[EntityDetector]] = None,
) -> List[DetectedEntity]:
    stack = tuple(detectors or _default_detectors())
    candidates: List[DetectedEntity] = []
    for detector in stack:
        try:
            candidates.extend(detector.detect(text))
        except Exception:
            continue
    return _resolve_overlaps(candidates)


def apply_detections(
    text: str,
    candidates: Sequence[DetectedEntity],
    policy: MaskingPolicy,
) -> str:
    selected = _resolve_overlaps(candidates)
    if not selected:
        return text

    output = []
    cursor = 0
    for candidate in selected:
        if candidate.start < cursor:
            continue
        output.append(text[cursor:candidate.start])
        span_text = text[candidate.start:candidate.end]
        if policy.should_mask(candidate.label, span_text):
            output.append(_placeholder_for(candidate.label))
        else:
            output.append(span_text)
        cursor = candidate.end

    output.append(text[cursor:])
    return "".join(output)


def _default_detectors() -> Tuple[EntityDetector, ...]:
    global _DEFAULT_DETECTORS
    if _DEFAULT_DETECTORS is None:
        _DEFAULT_DETECTORS = (RegexDetector(), SpaCyDetector())
    return _DEFAULT_DETECTORS


def _resolve_overlaps(candidates: Sequence[DetectedEntity]) -> List[DetectedEntity]:
    ordered = sorted(
        candidates,
        key=lambda ent: (
            ent.priority,
            ent.order,
            -(ent.end - ent.start),
            ent.start,
            ent.end,
            ent.label,
            ent.source,
            ent.text.lower(),
        ),
    )

    chosen: List[DetectedEntity] = []
    for candidate in ordered:
        if any(candidate.start < existing.end and existing.start < candidate.end for existing in chosen):
            continue
        chosen.append(candidate)

    return sorted(
        chosen,
        key=lambda ent: (ent.start, ent.end, ent.priority, ent.order, ent.label, ent.source),
    )


def _placeholder_for(label: str) -> str:
    return _PLACEHOLDERS.get(str(label).upper(), f"[{str(label).upper()}_MASKED]")
