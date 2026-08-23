import pytest

from secure_rag.masker import mask_text
from secure_rag.policies import DefaultPolicy, MaskingPolicy


class KeepLabelPolicy(MaskingPolicy):
    """Test policy that KEEP-masks everything except one configurable label."""

    def __init__(self, keep_label: str):
        self.keep_label = keep_label

    def should_mask(self, entity_label, entity_text) -> bool:
        return entity_label != self.keep_label


class TestDefaultPolicy:
    def test_masks_entity_current_implementation_masks(self):
        policy = DefaultPolicy()
        assert policy.should_mask("PERSON", "Rajesh Kumar") is True
        assert policy.should_mask("ORG", "Apollo Hospital") is True
        assert policy.should_mask("GPE", "Mumbai") is True
        assert policy.should_mask("PHONE", "9876543210") is True
        assert policy.should_mask("AADHAAR", "123456789012") is True
        assert policy.should_mask("PAN", "ABCDE1234F") is True

    def test_is_a_masking_policy(self):
        assert isinstance(DefaultPolicy(), MaskingPolicy)

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            MaskingPolicy()


class TestBackwardCompatibility:
    @pytest.mark.parametrize(
        "text",
        [
            "contact dr. at doctor@hospital.com",
            "call 9876543210",
            "Aadhaar: 1234 5678 9012",
            "PAN: ABCDE1234F",
            "Medical ID: MRN1002",
            "DOB: 15/08/1990",
            "",
            "Patient presents with fever and cough.",
        ],
    )
    def test_default_call_matches_explicit_default_policy(self, text):
        assert mask_text(text) == mask_text(text, policy=DefaultPolicy())

    def test_plain_call_still_works(self):
        result = mask_text("call 9876543210")
        assert "[PHONE_MASKED]" in result
        assert "9876543210" not in result

    def test_explicit_default_policy_call_works(self):
        result = mask_text(
            "email patient@email.com from 9876543210",
            policy=DefaultPolicy(),
        )
        assert "[EMAIL_MASKED]" in result
        assert "[PHONE_MASKED]" in result

    @pytest.mark.slow
    def test_ner_output_identical_with_and_without_policy_arg(self):
        text = "Rajesh Kumar visited Apollo Hospital in Mumbai."
        assert mask_text(text) == mask_text(text, policy=DefaultPolicy())
        assert "[NAME_MASKED]" in mask_text(text)


class TestCustomPolicy:
    def test_custom_policy_keeps_regex_entity_default_would_mask(self):
        text = "email a@b.com about appointment 9876543210"
        kept = mask_text(text, policy=KeepLabelPolicy(keep_label="EMAIL"))
        assert "a@b.com" in kept             # EMAIL kept by custom policy
        assert "[EMAIL_MASKED]" not in kept
        assert "[PHONE_MASKED]" in kept      # unrelated detection still masked

    def test_custom_policy_output_differs_from_default(self):
        text = "email a@b.com"
        default_out = mask_text(text)
        kept_out = mask_text(text, policy=KeepLabelPolicy(keep_label="EMAIL"))
        assert default_out != kept_out
        assert "[EMAIL_MASKED]" in default_out
        assert kept_out == "email a@b.com"

    @pytest.mark.slow
    def test_custom_policy_keeps_ner_person_default_would_mask(self):
        text = "Rajesh Kumar visited the clinic."
        baseline = mask_text(text)
        assert "[NAME_MASKED]" in baseline   # DefaultPolicy masks PERSON

        kept = mask_text(text, policy=KeepLabelPolicy(keep_label="PERSON"))
        assert "Rajesh Kumar" in kept        # custom policy keeps PERSON
        assert kept != baseline
