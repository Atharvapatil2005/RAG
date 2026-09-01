import copy
import json
import pathlib
import pytest

from benchmarks.retrieval.ground_truth import (
    GROUND_TRUTH_PATH_V2,
    GROUND_TRUTH_VERSION_V2,
    generate_ground_truth_v2,
    validate_ground_truth,
    compute_statistics,
    _load_mrn_records_raw,
    AGGREGATE_QUERIES_V2,
)


def _load_v2():
    return json.loads(pathlib.Path(GROUND_TRUTH_PATH_V2).read_text())


def test_v2_file_exists():
    assert pathlib.Path(GROUND_TRUTH_PATH_V2).exists()


def test_v2_preserves_v1():
    v1_path = pathlib.Path("benchmarks/retrieval/ground_truth_v1.json")
    assert v1_path.exists()
    v1 = json.loads(v1_path.read_text())
    assert v1["version"] == "v1"


def test_v2_version_and_schema():
    gt = _load_v2()
    assert gt["version"] == GROUND_TRUTH_VERSION_V2
    assert "relevant_records" in gt["schema"]
    assert "list[str]" in gt["schema"]["relevant_records"]


def test_single_record_query_remains_list_of_one():
    gt = _load_v2()
    singles = [q for q in gt["queries"] if q["qid"] == "MRN1001_Q1"]
    assert len(singles) == 1
    assert singles[0]["relevant_records"] == ["MRN1001"]


def test_multi_record_amlodipine():
    gt = _load_v2()
    q = next(x for x in gt["queries"] if x["qid"] == "AGG_AMLODIPINE_5MG")
    assert q["question"] == "Which patients were prescribed Amlodipine 5mg?"
    assert q["relevant_records"] == ["MRN1001", "MRN1005", "MRN1021", "MRN1025", "MRN1026", "MRN1052", "MRN1054", "MRN1059", "MRN1066", "MRN1070", "MRN1074", "MRN1085", "MRN1092", "MRN1107", "MRN1111", "MRN1118", "MRN1119"]
    assert len(q["relevant_records"]) == 17


def test_multi_record_metformin():
    gt = _load_v2()
    q = next(x for x in gt["queries"] if x["qid"] == "AGG_METFORMIN_500MG")
    assert q["question"] == "Which patients were prescribed Metformin 500mg?"
    assert len(q["relevant_records"]) == 7
    assert q["relevant_records"] == ["MRN1012", "MRN1047", "MRN1055", "MRN1068", "MRN1080", "MRN1084", "MRN1113"]


def test_multi_record_paracetamol():
    gt = _load_v2()
    q = next(x for x in gt["queries"] if x["qid"] == "AGG_PARACETAMOL_650MG")
    assert q["question"] == "Who received Paracetamol 650mg?"
    assert len(q["relevant_records"]) == 20


def test_multi_record_hypertension():
    gt = _load_v2()
    q = next(x for x in gt["queries"] if x["qid"] == "AGG_HYPERTENSION")
    assert q["question"] == "Which patients have Hypertension?"
    assert len(q["relevant_records"]) == 16
    assert q["relevant_records"] == ["MRN1001", "MRN1005", "MRN1021", "MRN1025", "MRN1026", "MRN1052", "MRN1059", "MRN1066", "MRN1070", "MRN1074", "MRN1085", "MRN1104", "MRN1107", "MRN1111", "MRN1118", "MRN1119"]


def test_multi_record_t2d_hypertension():
    gt = _load_v2()
    q = next(x for x in gt["queries"] if x["qid"] == "AGG_T2D_HYPERTENSION")
    assert q["question"] == "Which patients are being treated for both Type 2 Diabetes and Hypertension?"
    assert q["relevant_records"] == ["MRN1104"]


def test_zero_relevant_records_valid_schema_but_flagged():
    gt = generate_ground_truth_v2()
    gt_zero = copy.deepcopy(gt)
    gt_zero["queries"].append({
        "qid": "TEST_ZERO",
        "question": "Which patients were prescribed UnknownDrug 999mg?",
        "category": "general",
        "subcategory": "factual_hospital",
        "relevant_records": [],
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
    })
    issues = validate_ground_truth(gt_zero)
    assert not any("FAIL" in i for i in issues)
    stats = compute_statistics(gt_zero, _load_mrn_records_raw(), set(_load_mrn_records_raw().keys()))
    assert stats["zero_record_queries"] == 1


def test_duplicate_relevant_record_ids_detected():
    gt = generate_ground_truth_v2()
    gt_dup = copy.deepcopy(gt)
    gt_dup["queries"].append({
        "qid": "TEST_DUP",
        "question": "dup",
        "category": "general",
        "subcategory": "factual_hospital",
        "relevant_records": ["MRN1001", "MRN1001"],
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
    })
    issues = validate_ground_truth(gt_dup)
    assert any("duplicate" in i.lower() for i in issues)


def test_nonexistent_mrn_detected():
    gt = generate_ground_truth_v2()
    gt_bad = copy.deepcopy(gt)
    gt_bad["queries"].append({
        "qid": "TEST_BAD_MRN",
        "question": "bad",
        "category": "general",
        "subcategory": "factual_hospital",
        "relevant_records": ["MRN9999"],
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
    })
    issues = validate_ground_truth(gt_bad)
    assert any("non-existent" in i or "nonexistent" in i.lower() for i in issues)


def test_invalid_schema_missing_field():
    gt = generate_ground_truth_v2()
    gt_inv = copy.deepcopy(gt)
    bad = copy.deepcopy(gt_inv["queries"][0])
    del bad["relevant_records"]
    gt_inv["queries"].append({"qid": "TEST_MISSING", "question": "x", "category": "general", "subcategory": "factual_hospital", "phi_in_answer": False, "expected_behaviour": "record_retrieval"})
    issues = validate_ground_truth(gt_inv)
    assert any("missing field" in i.lower() for i in issues)


def test_invalid_schema_wrong_type():
    gt = generate_ground_truth_v2()
    gt_inv = copy.deepcopy(gt)
    gt_inv["queries"].append({
        "qid": "TEST_WRONG_TYPE",
        "question": "x",
        "category": "general",
        "subcategory": "factual_hospital",
        "relevant_records": "MRN1001",
        "phi_in_answer": False,
        "expected_behaviour": "record_retrieval",
    })
    issues = validate_ground_truth(gt_inv)
    assert any("not a list" in i.lower() for i in issues)


def test_backward_compatible_metadata_preserved():
    gt = _load_v2()
    for q in gt["queries"]:
        for field in ("qid", "question", "category", "subcategory", "relevant_records", "phi_in_answer", "expected_behaviour"):
            assert field in q, f"missing {field} in {q['qid']}"
        assert isinstance(q["relevant_records"], list)
        assert isinstance(q["phi_in_answer"], bool)
        assert isinstance(q["qid"], str)


def test_validation_against_sample_data_independently():
    txt = pathlib.Path("data/sample_patient_data.txt").read_text()
    for agg in AGGREGATE_QUERIES_V2:
        term_map = {
            "AGG_AMLODIPINE_5MG": "Amlodipine 5mg",
            "AGG_METFORMIN_500MG": "Metformin 500mg",
            "AGG_PARACETAMOL_650MG": "Paracetamol 650mg",
            "AGG_HYPERTENSION": "Hypertension",
            "AGG_T2D_HYPERTENSION": None,
        }
        qid = agg["qid"]
        if qid == "AGG_T2D_HYPERTENSION":
            expected = sorted([line for line in txt.split("Medical ID: ") if "Type 2 Diabetes" in line and "Hypertension" in line])
            import re
            mrns = re.findall(r"MRN\d+", "\n".join(expected))
            assert sorted(agg["relevant_records"]) == sorted(set(mrns))
        else:
            term = term_map[qid]
            count = txt.lower().count(term.lower())
            assert len(agg["relevant_records"]) > 0
            for rid in agg["relevant_records"]:
                assert rid in txt
