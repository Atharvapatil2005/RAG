import re
from typing import Dict, Iterable, Mapping, Tuple

from .base import MaskingPolicy

DECISION_MASK = "mask"
DECISION_KEEP = "keep"


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


class ConfigurablePolicy(MaskingPolicy):
    """Generic, domain-agnostic policy driven purely by configuration.

    The engine knows only abstract concepts: entity labels, entity text,
    configured decisions (MASK/KEEP) and a configured preserved vocabulary.
    All domain knowledge lives in the configuration that constructed it;
    this module contains none.

    Decision order (privacy is the default):

        1. entity text matches a preserved term      -> KEEP
        2. entity label has an explicit decision     -> that decision
        3. anything else (unknown / ambiguous)       -> MASK
    """

    def __init__(
        self,
        entity_decisions: Mapping[str, str] = None,
        preserved_terms: Iterable[str] = None,
        preserved_categories: Iterable[str] = None,
        name: str = "configurable",
    ):
        self.name = name
        self.entity_decisions: Dict[str, str] = {
            str(label).upper(): str(decision).lower()
            for label, decision in (entity_decisions or {}).items()
        }
        normalized_terms = []
        for term in preserved_terms or ():
            normalized = _normalize(term)
            if normalized and normalized not in normalized_terms:
                normalized_terms.append(normalized)
        self.preserved_terms: Tuple[str, ...] = tuple(normalized_terms)
        # (?<!\w)/(?!\w): phrase match on word boundaries so multi-word
        # vocabulary entries ("main street") also match inflected spans.
        self._term_patterns = tuple(
            re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")
            for term in self.preserved_terms
        )
        self.preserved_categories: Tuple[str, ...] = tuple(
            dict.fromkeys(_normalize(c) for c in preserved_categories or () if _normalize(c))
        )

    def should_mask(self, entity_label: str, entity_text: str) -> bool:
        text_norm = _normalize(entity_text)
        if text_norm:
            for pattern in self._term_patterns:
                if pattern.search(text_norm):
                    return False  # explicitly preserved by configuration
        label_norm = _normalize(entity_label)
        if label_norm and label_norm in self.preserved_categories:
            return False  # preserved category wins for domain-aware detectors
        decision = self.entity_decisions.get(str(entity_label).upper())
        if decision == DECISION_KEEP:
            return False  # explicitly kept by configuration
        return True  # explicit mask, unlisted, unknown or ambiguous -> MASK

    def __repr__(self) -> str:
        return (
            f"ConfigurablePolicy(name={self.name!r}, "
            f"entity_decisions={self.entity_decisions!r}, "
            f"preserved_terms={len(self.preserved_terms)}, "
            f"preserved_categories={len(self.preserved_categories)})"
        )
