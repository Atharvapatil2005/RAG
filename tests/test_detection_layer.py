from secure_rag.detection import DetectedEntity, MedicalDetector, apply_detections, detect_entities, load_detector_stack
from secure_rag.masker import mask_text
from secure_rag.policies import load_policy


class TestMedicalDetector:
    def test_case_insensitive_medication_and_dosage(self):
        detector = MedicalDetector()
        entities = list(detector.detect("amlodipine 5mg, AMLODIPINE 5MG, Amlodipine 5mg"))

        medication_hits = [entity for entity in entities if entity.label == "MEDICATION"]
        dosage_hits = [entity for entity in entities if entity.label == "DOSAGE"]

        assert len(medication_hits) == 3
        assert len(dosage_hits) == 3
        assert all(hit.text.lower() == "amlodipine" for hit in medication_hits)
        assert all(hit.text.lower() == "5mg" for hit in dosage_hits)

    def test_multi_word_diagnosis_terms(self):
        detector = MedicalDetector()
        entities = list(
            detector.detect(
                "Type 2 Diabetes, Chronic Kidney Disease, Dengue Fever, Hypertension"
            )
        )

        labels = {(entity.label, entity.text) for entity in entities}

        assert ("DIAGNOSIS", "Type 2 Diabetes") in labels
        assert ("DIAGNOSIS", "Chronic Kidney Disease") in labels
        assert ("DIAGNOSIS", "Dengue Fever") in labels
        assert ("DIAGNOSIS", "Hypertension") in labels

    def test_false_substring_matches_do_not_fire(self):
        detector = MedicalDetector()
        entities = list(detector.detect("Amlodipinex and hypertensive symptoms"))
        assert entities == []


class TestDetectionIntegration:
    def test_domain_detector_stack_returns_medical_terms(self):
        entities = detect_entities("Metformin 500mg for Type 2 Diabetes.", detectors=load_detector_stack("medical"))

        labels = {(entity.label, entity.text) for entity in entities}
        assert ("MEDICATION", "Metformin") in labels
        assert ("DOSAGE", "500mg") in labels
        assert ("DIAGNOSIS", "Type 2 Diabetes") in labels

    def test_overlap_resolution_keeps_clinical_terms_and_masks_pii(self):
        text = "Nakul Rattan was prescribed Amlodipine 5mg for Hypertension."
        candidates = [
            DetectedEntity(0, 12, "Nakul Rattan", "PERSON", "spacy", 2),
            DetectedEntity(28, 38, "Amlodipine", "MEDICATION", "domain:medical", 1),
            DetectedEntity(39, 42, "5mg", "DOSAGE", "domain:medical", 1),
            DetectedEntity(28, 42, "Amlodipine 5mg", "PRODUCT", "spacy", 2),
            DetectedEntity(47, 59, "Hypertension", "DIAGNOSIS", "domain:medical", 1),
        ]

        result = apply_detections(text, candidates, load_policy("medical"))

        assert result == "[NAME_MASKED] was prescribed Amlodipine 5mg for Hypertension."

    def test_mask_text_with_medical_detector_stack_preserves_clinical_terms(self):
        text = "Patient was prescribed Metformin 500mg for Type 2 Diabetes."
        result = mask_text(text, policy=load_policy("medical"), detectors=load_detector_stack("medical"))

        assert "Metformin 500mg" in result
        assert "Type 2 Diabetes" in result
        assert "[NAME_MASKED]" not in result

