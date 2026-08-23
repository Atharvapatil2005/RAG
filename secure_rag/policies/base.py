import abc


class MaskingPolicy(abc.ABC):
    """Decides whether a detected entity should be masked.

    The policy layer answers only "MASK or KEEP?" for entities that the
    detectors (regex rules / NER) have already found. How an entity is
    masked (which placeholder) remains the masker's responsibility.
    """

    @abc.abstractmethod
    def should_mask(self, entity_label: str, entity_text: str) -> bool:
        """Return True if the detected entity should be replaced.

        entity_label: detector label for the entity (e.g. a NER label such
            as "PERSON" or a regex rule label such as "EMAIL").
        entity_text: the exact text span that was detected.
        """
        raise NotImplementedError
