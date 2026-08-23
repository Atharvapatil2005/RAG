from .base import MaskingPolicy


class DefaultPolicy(MaskingPolicy):
    """Masks every detected entity, reproducing the historical mask_text() behavior."""

    def should_mask(self, entity_label: str, entity_text: str) -> bool:
        return True
