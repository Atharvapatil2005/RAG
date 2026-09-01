from typing import Optional, Sequence

from .policies import DefaultPolicy, MaskingPolicy
from .detection import (
    EntityDetector,
    RegexDetector,
    _default_detectors,
    apply_detections,
    detect_entities,
)


def mask_text(
    text: str,
    policy: Optional[MaskingPolicy] = None,
    detectors: Optional[Sequence[EntityDetector]] = None,
) -> str:
    if policy is None:
        policy = DefaultPolicy()
    stack = _default_detectors() if detectors is None else tuple(detectors)

    regex_detectors = tuple(detector for detector in stack if isinstance(detector, RegexDetector))
    later_detectors = tuple(detector for detector in stack if not isinstance(detector, RegexDetector))

    working = text
    if regex_detectors:
        regex_candidates = detect_entities(working, detectors=regex_detectors)
        working = apply_detections(working, regex_candidates, policy)

    if later_detectors:
        later_candidates = detect_entities(working, detectors=later_detectors)
        working = apply_detections(working, later_candidates, policy)

    return working
