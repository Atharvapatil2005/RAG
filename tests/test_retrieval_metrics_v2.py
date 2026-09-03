import json
import pathlib
import pytest

from benchmarks.retrieval.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    compute_per_query_metrics,
    compute_metrics,
)

def _mk_retrieved(record_ids):
    return [{"record_id": rid, "relevant": False, "rank": i, "chunk_index": i, "score": 0.0} for i, rid in enumerate(record_ids)]


def test_single_relevant_record():
    retrieved = _mk_retrieved(["A", "B", "C"])
    s = {"A"}
    assert hit_rate_at_k(retrieved, 1, s) == 1
    assert hit_rate_at_k(retrieved, 1, {"B"}) == 0
    assert precision_at_k(retrieved, 1, s) == 1.0
    assert precision_at_k(retrieved, 2, s) == 0.5
    assert recall_at_k(retrieved, 1, relevant_set=s) == 1.0
    assert mrr_at_k(retrieved, 3, s) == 1.0
    # not in top-k
    assert hit_rate_at_k(_mk_retrieved(["B", "C"]), 1, s) == 0
    assert recall_at_k(_mk_retrieved(["B", "C"]), 2, relevant_set=s) == 0.0


def test_multiple_relevant_records():
    retrieved = _mk_retrieved(["MRN1", "MRN2", "X", "MRN3"])
    s = {"MRN1", "MRN2", "MRN3", "MRN4"}
    assert hit_rate_at_k(retrieved, 4, s) == 1
    assert recall_at_k(retrieved, 2, relevant_set=s) == 0.5
    assert recall_at_k(retrieved, 4, relevant_set=s) == 0.75
    assert precision_at_k(retrieved, 4, s) == 0.75
    assert precision_at_k(retrieved, 2, s) == 1.0
    assert mrr_at_k(retrieved, 4, s) == 1.0
    # 6 of 17 in top-10 example from spec
    retrieved10 = _mk_retrieved([f"MRN{i}" for i in range(6)] + [f"X{i}" for i in range(4)])
    s17 = {f"MRN{i}" for i in range(17)}
    assert recall_at_k(retrieved10, 10, relevant_set=s17) == pytest.approx(6/17)
    assert precision_at_k(retrieved10, 10, s17) == pytest.approx(0.6)
    assert hit_rate_at_k(retrieved10, 10, s17) == 1


def test_zero_relevant_records():
    retrieved = _mk_retrieved(["A", "B"])
    s = set()
    assert hit_rate_at_k(retrieved, 2, s) == 0
    assert precision_at_k(retrieved, 2, s) == 0.0
    assert recall_at_k(retrieved, 2, relevant_set=s) == 0.0
    assert mrr_at_k(retrieved, 2, s) == 0.0
    # empty retrieved as well
    assert hit_rate_at_k([], 5, s) == 0


def test_duplicate_retrieved_records():
    retrieved = _mk_retrieved(["A", "A", "B", "A"])
    s = {"A", "B"}
    assert recall_at_k(retrieved, 4, relevant_set=s) == 1.0
    assert precision_at_k(retrieved, 4, s) == 0.5
    assert hit_rate_at_k(retrieved, 4, s) == 1
    # MRR should use first unique occurrence
    assert mrr_at_k(retrieved, 4, s) == 1.0
    retrieved2 = _mk_retrieved(["X", "A", "A", "B"])
    assert mrr_at_k(retrieved2, 4, s) == 0.5


def test_duplicate_ground_truth_dedup():
    retrieved = _mk_retrieved(["A", "B"])
    s = {"A"}
    assert recall_at_k(retrieved, 2, relevant_set=s) == 1.0


def test_nonexistent_record_ids():
    retrieved = _mk_retrieved(["X", "Y", "Z"])
    s = {"A", "B"}
    assert recall_at_k(retrieved, 3, relevant_set=s) == 0.0
    assert precision_at_k(retrieved, 3, s) == 0.0
    assert hit_rate_at_k(retrieved, 3, s) == 0
    assert mrr_at_k(retrieved, 3, s) == 0.0


def test_recall_calculation():
    retrieved = _mk_retrieved(["A", "X", "B", "Y", "C"])
    s = {"A", "B", "C", "D"}
    assert recall_at_k(retrieved, 3, relevant_set=s) == pytest.approx(0.5)
    assert recall_at_k(retrieved, 5, relevant_set=s) == pytest.approx(0.75)


def test_precision_calculation():
    retrieved = _mk_retrieved(["A", "A", "X", "B"])
    s = {"A", "B"}
    assert precision_at_k(retrieved, 4, s) == pytest.approx(0.5)
    assert precision_at_k(retrieved, 2, s) == pytest.approx(0.5)


def test_hit_rate():
    assert hit_rate_at_k(_mk_retrieved(["A"]), 1, {"A"}) == 1
    assert hit_rate_at_k(_mk_retrieved(["B"]), 1, {"A"}) == 0
    assert hit_rate_at_k(_mk_retrieved(["B", "A"]), 1, {"A"}) == 0
    assert hit_rate_at_k(_mk_retrieved(["B", "A"]), 2, {"A"}) == 1


def test_mrr():
    assert mrr_at_k(_mk_retrieved(["A", "B", "C"]), 3, {"A"}) == 1.0
    assert mrr_at_k(_mk_retrieved(["B", "A", "C"]), 3, {"A"}) == 0.5
    assert mrr_at_k(_mk_retrieved(["B", "C", "A"]), 3, {"A"}) == pytest.approx(1/3)
    assert mrr_at_k(_mk_retrieved(["B", "C"]), 2, {"A"}) == 0.0
    assert mrr_at_k(_mk_retrieved(["B", "A", "A"]), 3, {"A"}) == 0.5


def test_k_greater_than_index_size():
    retrieved = _mk_retrieved(["A", "B"])
    s = {"A"}
    assert precision_at_k(retrieved, 50, s) == pytest.approx(0.5)
    assert recall_at_k(retrieved, 50, relevant_set=s) == 1.0
    assert hit_rate_at_k(retrieved, 50, s) == 1
    assert mrr_at_k(retrieved, 50, s) == 1.0
    # k larger than available still safe for empty relevant
    assert recall_at_k(retrieved, 50, relevant_set=set()) == 0.0


def test_backward_compat_v1_artifact():
    path = pathlib.Path("benchmarks/retrieval/_archive_v1/retrieval_results_v1.json")
    if not path.exists():
        pytest.skip("v1 artifact missing")
    rr = json.loads(path.read_text())
    m = compute_metrics(rr)
    assert m["version"] == "v1"
    assert set(m["k_values"]) == {1, 3, 5, 10}


def test_v2_aggregate_queries():
    path = pathlib.Path("benchmarks/retrieval/metrics_v2.json")
    if not path.exists():
        pytest.skip("metrics_v2 not generated")
    m = json.loads(path.read_text())
    assert m["version"] == "v2"
    assert set(m["k_values"]) == {1, 3, 5, 10, 20, 30, 50}
    for qid in ["AGG_AMLODIPINE_5MG", "AGG_METFORMIN_500MG", "AGG_PARACETAMOL_650MG", "AGG_HYPERTENSION", "AGG_T2D_HYPERTENSION"]:
        assert qid in m["per_query"]
    pq = m["per_query"]["AGG_AMLODIPINE_5MG"]
    assert pq["num_relevant"] == 17
    pq2 = m["per_query"]["AGG_PARACETAMOL_650MG"]
    assert pq2["num_relevant"] == 20
    pq3 = m["per_query"]["AGG_HYPERTENSION"]
    assert pq3["num_relevant"] == 16


def test_record_level_dedup_not_chunk_count():
    rr = {
        "version": "v2",
        "runner_version": "3",
        "configs": {"baseline_a": {}, "baseline_b": {}, "secure_rag": {}},
        "k_values": [1, 3, 5],
        "max_k": 5,
        "queries": [
            {
                "qid": "Q1",
                "question": "q",
                "category": "general",
                "subcategory": "factual_hospital",
                "ground_truth_records": ["R1", "R2"],
                "expected_behaviour": "multi_record_retrieval",
                "results": {
                    "baseline_a": {"retrieved": _mk_retrieved(["R1", "R1", "R1"])},
                    "baseline_b": {"retrieved": _mk_retrieved(["R1", "R1", "R1"])},
                    "secure_rag": {"retrieved": _mk_retrieved(["R1", "R1", "R1"])},
                },
            }
        ],
    }
    m = compute_metrics(rr, version="v2")
    pq = m["per_query"]["Q1"]
    assert pq["baseline_a"]["k_3"]["relevant_retrieved"] == 1
    assert pq["baseline_a"]["k_3"]["recall"] == 0.5
    assert pq["baseline_a"]["k_3"]["precision"] == pytest.approx(1/3)
