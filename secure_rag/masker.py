import re
import os
from typing import Optional

from .policies import DefaultPolicy, MaskingPolicy

_nlp = None


def _get_nlp():#module based caching for spacy pipeline 
                #lazily loads the spacy NER model only when it is needed
    global _nlp
    if _nlp is None:
        import spacy
        model = os.getenv("SPACY_MODEL", "en_core_web_sm")
        _nlp = spacy.load(model)
    return _nlp


#regex rules: (policy label, pattern, placeholder) applied in this exact order
_REGEX_RULES = (
    ("EMAIL", r'\S+@\S+', '[EMAIL_MASKED]'),
    ("ADDRESS", r'\b\d+\s+[A-Z][a-zA-Z]+(\s+[A-Z][a-zA-Z]+)*\s+(Road|Street|Avenue|Lane|Marg)\b', '[ADDRESS_MASKED]'),
    ("PHONE", r'\b\d{10}\b', '[PHONE_MASKED]'),
    ("AADHAAR", r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[AADHAAR_MASKED]'),
    ("PAN", r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN_MASKED]'),
    ("HEALTH_ID", r'\b\d{8,16}\b', '[HEALTH_ID_MASKED]'),
    ("PATIENT_ID", r'\b(?:MRN|UHID|PID)\s*:?[\s-]*\d{4,14}\b', '[PATIENT_ID_MASKED]'),
    ("DOB", r'\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b', '[DOB_MASKED]'),
)

#NER labels considered maskable candidates, mapped to their placeholders
_NER_LABEL_PLACEHOLDERS = {
    "PERSON": "[NAME_MASKED]",
    "GPE": "[ADDRESS_MASKED]",
    "LOC": "[ADDRESS_MASKED]",
    "FAC": "[ADDRESS_MASKED]",
    "ORG": "[ORG_MASKED]",
}


def mask_text(text: str, policy: Optional[MaskingPolicy] = None) -> str:
    #function works in two stages -> 1.Regex masking 2.Spacy NER matching
    #the policy decides MASK/KEEP per detected entity; DefaultPolicy() keeps historical behavior
    if policy is None:
        policy = DefaultPolicy()

    for label, pattern, placeholder in _REGEX_RULES:
        text = re.sub(
            pattern,
            lambda m, label=label, placeholder=placeholder: (
                placeholder if policy.should_mask(label, m.group(0)) else m.group(0)
            ),
            text,
        )

    try:
        nlp = _get_nlp()    #after regex pattern mathing fn tries Spacy NER matching
        doc = nlp(text)
        for ent in doc.ents:
            placeholder = _NER_LABEL_PLACEHOLDERS.get(ent.label_)
            if placeholder is None:
                continue
            if not policy.should_mask(ent.label_, ent.text):
                continue
            text = text.replace(ent.text, placeholder, 1)
    except Exception:
        pass

    return text
