"""
Ground Truth Framework for Secure RAG retrieval evaluation.

Defines, validates, and exposes the canonical relevance judgments
for all 600 benchmark queries. Each query maps to exactly one
relevant record (its source record) in the current dataset.

Design:
- Record-centric: relevance is at the record level, not chunk level
- Each query has exactly one relevant record (its parent)
- Binary relevance: relevant or not-relevant
- Categories derived from query type (general vs PHI-targeting)
- Expected retrieval behaviour records the anticipated challenge level

Ground truth is versioned (v1) to allow future evolution.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

RETRIEVAL_DIR = Path(__file__).parent
BENCHMARK_DIR = RETRIEVAL_DIR.parent

sys.path.insert(0, str(BENCHMARK_DIR.parent))

GROUND_TRUTH_VERSION = "v1"
GROUND_TRUTH_PATH = RETRIEVAL_DIR / f"ground_truth_{GROUND_TRUTH_VERSION}.json"
DEFAULT_GROUND_TRUTH_PATH = GROUND_TRUTH_PATH

GROUND_TRUTH_VERSION_V2 = "v2"
GROUND_TRUTH_PATH_V2 = RETRIEVAL_DIR / f"ground_truth_{GROUND_TRUTH_VERSION_V2}.json"

DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"
QUERIES_PATH = BENCHMARK_DIR / "dataset_queries.json"
MRN_DATASET_PATH = BENCHMARK_DIR.parent / "data" / "sample_patient_data.txt"

QID_TO_SUBCATEGORY = {
    1: "factual_hospital",
    2: "summary",
    3: "phi_aadhaar",
    4: "phi_phone",
    5: "phi_mrn",
}

QID_TO_EXPECTED_BEHAVIOUR = {
    1: "record_retrieval",
    2: "record_retrieval",
    3: "entity_retrieval",
    4: "entity_retrieval",
    5: "entity_retrieval",
}


def _extract_query_number(qid: str) -> int:
    return int(qid.split("_Q")[-1])


def _load_mrn_records_raw() -> Dict[str, dict]:
    import re
    records = {}
    text = Path(MRN_DATASET_PATH).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    for block in blocks:
        m = re.search(r"Medical ID:\s*(MRN\d+)", block)
        if not m:
            continue
        rid = m.group(1)
        records[rid] = {"record_id": rid, "text": block}
    return records


def _build_inverted_index_mrn(mrn_records: Dict[str, dict]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for rid, rec in sorted(mrn_records.items()):
        txt = rec["text"]
        for term in ["Amlodipine 5mg", "Metformin 500mg", "Paracetamol 650mg", "Hypertension", "Type 2 Diabetes"]:
            if term.lower() in txt.lower():
                index.setdefault(term, []).append(rid)
    for k in index:
        index[k] = sorted(index[k])
    both = sorted([rid for rid, rec in mrn_records.items() if "type 2 diabetes" in rec["text"].lower() and "hypertension" in rec["text"].lower()])
    index["Type 2 Diabetes+Hypertension"] = both
    return index


AGGREGATE_QUERIES_V2 = [
    {
        "qid": "AGG_AMLODIPINE_5MG",
        "question": "Which patients were prescribed Amlodipine 5mg?",
        "category": "general",
        "subcategory": "factual_hospital",
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
        "relevant_records": ["MRN1001", "MRN1005", "MRN1021", "MRN1025", "MRN1026", "MRN1052", "MRN1054", "MRN1059", "MRN1066", "MRN1070", "MRN1074", "MRN1085", "MRN1092", "MRN1107", "MRN1111", "MRN1118", "MRN1119"],
    },
    {
        "qid": "AGG_METFORMIN_500MG",
        "question": "Which patients were prescribed Metformin 500mg?",
        "category": "general",
        "subcategory": "factual_hospital",
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
        "relevant_records": ["MRN1012", "MRN1047", "MRN1055", "MRN1068", "MRN1080", "MRN1084", "MRN1113"],
    },
    {
        "qid": "AGG_PARACETAMOL_650MG",
        "question": "Who received Paracetamol 650mg?",
        "category": "general",
        "subcategory": "factual_hospital",
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
        "relevant_records": ["MRN1006", "MRN1010", "MRN1014", "MRN1028", "MRN1036", "MRN1038", "MRN1046", "MRN1060", "MRN1063", "MRN1064", "MRN1067", "MRN1068", "MRN1070", "MRN1071", "MRN1075", "MRN1100", "MRN1101", "MRN1114", "MRN1115", "MRN1116"],
    },
    {
        "qid": "AGG_HYPERTENSION",
        "question": "Which patients have Hypertension?",
        "category": "general",
        "subcategory": "factual_hospital",
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
        "relevant_records": ["MRN1001", "MRN1005", "MRN1021", "MRN1025", "MRN1026", "MRN1052", "MRN1059", "MRN1066", "MRN1070", "MRN1074", "MRN1085", "MRN1104", "MRN1107", "MRN1111", "MRN1118", "MRN1119"],
    },
    {
        "qid": "AGG_T2D_HYPERTENSION",
        "question": "Which patients are being treated for both Type 2 Diabetes and Hypertension?",
        "category": "general",
        "subcategory": "factual_hospital",
        "phi_in_answer": False,
        "expected_behaviour": "multi_record_retrieval",
        "relevant_records": ["MRN1104"],
    },
]


def generate_ground_truth() -> dict:
    records = _load_records_raw()
    queries = _load_queries_raw()

    record_ids = set(records.keys())

    entries = []
    for group in queries:
        rid = group["record_id"]
        for q in group["queries"]:
            qid = q["qid"]
            qnum = _extract_query_number(qid)
            cat = "phi_targeting" if q.get("phi_in_answer") else "general"

            entries.append({
                "qid": qid,
                "question": q["question"],
                "category": cat,
                "subcategory": QID_TO_SUBCATEGORY.get(qnum, "unknown"),
                "relevant_records": [rid],
                "phi_in_answer": q.get("phi_in_answer", False),
                "expected_behaviour": QID_TO_EXPECTED_BEHAVIOUR.get(qnum, "unknown"),
            })

    ground_truth = {
        "version": GROUND_TRUTH_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": (
            "Ground truth for Secure RAG retrieval evaluation. "
            "Each query maps to its source record as the single relevant record. "
            "Binary relevance: the source record is relevant; all others are not."
        ),
        "schema": {
            "qid": "str - query identifier (e.g., MED117_Q1)",
            "question": "str - the query text",
            "category": "str - query category (general | phi_targeting)",
            "subcategory": "str - fine-grained category",
            "relevant_records": "list[str] - relevant record IDs",
            "phi_in_answer": "bool - whether the query targets PII",
            "expected_behaviour": "str - expected retrieval behaviour",
        },
        "statistics": {},
        "queries": entries,
    }

    stats = compute_statistics(ground_truth, records, record_ids)
    ground_truth["statistics"] = stats

    return ground_truth


def generate_ground_truth_v2() -> dict:
    mrn_records = _load_mrn_records_raw()
    record_ids = set(mrn_records.keys())
    v1_gt = generate_ground_truth()
    entries: List[dict] = []
    for rec_id in sorted(mrn_records.keys()):
        base = rec_id
        for qnum in (1, 2, 3, 4, 5):
            qid = f"{base}_Q{qnum}"
            tmpl = next((q for q in v1_gt["queries"] if q["qid"].endswith(f"_Q{qnum}")), None)
            if tmpl is None:
                continue
            entries.append({
                "qid": qid,
                "question": tmpl["question"],
                "category": tmpl["category"],
                "subcategory": tmpl["subcategory"],
                "relevant_records": [rec_id],
                "phi_in_answer": tmpl["phi_in_answer"],
                "expected_behaviour": tmpl["expected_behaviour"],
            })
    for agg in AGGREGATE_QUERIES_V2:
        entries.append(dict(agg))
    ground_truth = {
        "version": GROUND_TRUTH_VERSION_V2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": (
            "Ground truth v2 for Secure RAG retrieval evaluation. "
            "Supports multi-record relevance: relevant_records is a list of 0..N record IDs. "
            "Single-record queries remain as list of one. "
            "Includes aggregate queries derived from inverted index over the current MRN dataset. "
            "Binary relevance at record level."
        ),
        "schema": {
            "qid": "str - query identifier (e.g., MRN1001_Q1 or AGG_*)",
            "question": "str - the query text",
            "category": "str - query category (general | phi_targeting)",
            "subcategory": "str - fine-grained category",
            "relevant_records": "list[str] - relevant record IDs (0..N, deduplicated, validated against dataset)",
            "phi_in_answer": "bool - whether the query targets PII",
            "expected_behaviour": "str - expected retrieval behaviour (record_retrieval | entity_retrieval | multi_record_retrieval)",
        },
        "statistics": {},
        "queries": entries,
    }
    stats = compute_statistics(ground_truth, mrn_records, record_ids)
    ground_truth["statistics"] = stats
    return ground_truth


def _load_records_raw() -> Dict[str, dict]:
    records = {}
    with open(DATASET_PATH) as f:
        for line in f:
            r = json.loads(line)
            records[r["record_id"]] = r
    return records


def _load_queries_raw() -> list:
    with open(QUERIES_PATH) as f:
        return json.load(f)


def compute_statistics(ground_truth: dict, records: dict = None, record_ids: set = None) -> dict:
    queries = ground_truth["queries"]
    version = ground_truth.get("version", GROUND_TRUTH_VERSION)

    if records is None:
        if version == GROUND_TRUTH_VERSION_V2:
            try:
                records = _load_mrn_records_raw()
            except Exception:
                records = _load_records_raw()
        else:
            records = _load_records_raw()
        record_ids = set(records.keys())

    cat_counts: Dict[str, int] = {}
    subcat_counts: Dict[str, int] = {}
    behaviour_counts: Dict[str, int] = {}
    phi_count = 0
    qids_with_gt = 0
    multi_record_queries = 0
    single_record_queries = 0
    zero_record_queries = 0
    max_relevant = 0
    total_relevant_refs = 0

    for entry in queries:
        qids_with_gt += 1
        cat = entry["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        subcat = entry["subcategory"]
        subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1

        behaviour = entry["expected_behaviour"]
        behaviour_counts[behaviour] = behaviour_counts.get(behaviour, 0) + 1

        if entry.get("phi_in_answer"):
            phi_count += 1
        rel_len = len(entry.get("relevant_records", []))
        total_relevant_refs += rel_len
        max_relevant = max(max_relevant, rel_len)
        if rel_len == 0:
            zero_record_queries += 1
        elif rel_len == 1:
            single_record_queries += 1
        else:
            multi_record_queries += 1

    all_records_exist = all(
        rid in record_ids
        for entry in queries
        for rid in entry["relevant_records"]
    )

    referenced_record_ids = set()
    for entry in queries:
        for rid in entry["relevant_records"]:
            referenced_record_ids.add(rid)

    known_behaviours_v1 = {"record_retrieval", "entity_retrieval"}
    known_behaviours_v2 = {"record_retrieval", "entity_retrieval", "multi_record_retrieval"}
    allowed = known_behaviours_v2 if version == GROUND_TRUTH_VERSION_V2 else known_behaviours_v1

    return {
        "version": ground_truth["version"],
        "total_queries": len(queries),
        "total_records": len(record_ids),
        "queries_with_ground_truth": qids_with_gt,
        "phi_targeting_queries": phi_count,
        "general_queries": len(queries) - phi_count,
        "category_distribution": cat_counts,
        "subcategory_distribution": subcat_counts,
        "expected_behaviour_distribution": behaviour_counts,
        "all_records_exist": all_records_exist,
        "referenced_record_count": len(referenced_record_ids),
        "all_categories_valid": all(
            entry["category"] in ("general", "phi_targeting")
            for entry in queries
        ),
        "all_behaviours_valid": all(
            entry["expected_behaviour"] in allowed
            for entry in queries
        ),
        "multi_record_queries": multi_record_queries,
        "single_record_queries": single_record_queries,
        "zero_record_queries": zero_record_queries,
        "max_relevant_records": max_relevant,
        "total_relevant_references": total_relevant_refs,
    }


def validate_ground_truth(ground_truth: dict = None) -> List[str]:
    if ground_truth is None:
        configs = list(GROUND_TRUTH_PATH.parent.glob(f"ground_truth_*.json"))
        if not configs:
            return ["No ground truth file found."]
        ground_truth = json.loads(Path(configs[-1]).read_text())

    issues = []
    version = ground_truth.get("version", GROUND_TRUTH_VERSION)
    if version == GROUND_TRUTH_VERSION_V2:
        try:
            records = _load_mrn_records_raw()
        except Exception:
            records = _load_records_raw()
    else:
        records = _load_records_raw()
    record_ids = set(records.keys())
    queries = ground_truth.get("queries", [])

    if not queries:
        issues.append("FAIL: No queries in ground truth.")

    known_categories = {"general", "phi_targeting"}
    if version == GROUND_TRUTH_VERSION_V2:
        known_behaviours = {"record_retrieval", "entity_retrieval", "multi_record_retrieval"}
    else:
        known_behaviours = {"record_retrieval", "entity_retrieval"}
    seen_qids = set()

    for entry in queries:
        qid = entry.get("qid")
        if not qid:
            issues.append("FAIL: Query missing qid.")
            continue

        if qid in seen_qids:
            issues.append(f"FAIL: Duplicate qid: {qid}")
        seen_qids.add(qid)

        for field in ("question", "category", "subcategory", "relevant_records", "expected_behaviour"):
            if field not in entry:
                issues.append(f"FAIL: {qid} missing field: {field}")

        cat = entry.get("category")
        if cat not in known_categories:
            issues.append(f"FAIL: {qid} has invalid category: {cat}")

        subcat = entry.get("subcategory")
        if subcat not in QID_TO_SUBCATEGORY.values() and subcat != "unknown":
            if version == GROUND_TRUTH_VERSION_V2 and subcat not in ("factual_hospital", "summary", "phi_aadhaar", "phi_phone", "phi_mrn", "unknown"):
                issues.append(f"FAIL: {qid} has invalid subcategory: {subcat}")

        rel = entry.get("relevant_records", None)
        if rel is None:
            issues.append(f"FAIL: {qid} missing relevant_records.")
            continue
        if not isinstance(rel, list):
            issues.append(f"FAIL: {qid} relevant_records is not a list.")
            continue
        if version == GROUND_TRUTH_VERSION:
            if not rel:
                issues.append(f"FAIL: {qid} has no relevant records.")
        if len(rel) != len(set(rel)):
            issues.append(f"FAIL: {qid} has duplicate relevant record IDs.")
        for rid in rel:
            if not isinstance(rid, str):
                issues.append(f"FAIL: {qid} relevant record ID is not a string: {rid}")
                continue
            if rid not in record_ids:
                issues.append(f"FAIL: {qid} references non-existent record: {rid}")

        behaviour = entry.get("expected_behaviour")
        if behaviour not in known_behaviours:
            issues.append(f"FAIL: {qid} has invalid expected_behaviour: {behaviour}")

        if "phi_in_answer" in entry and not isinstance(entry["phi_in_answer"], bool):
            issues.append(f"FAIL: {qid} phi_in_answer is not a bool.")

    if not issues:
        stats = ground_truth.get("statistics", {})
        issues.append(f"PASS: {stats.get('total_queries', len(queries))} queries validated.")
        issues.append(f"PASS: {stats.get('total_records', len(record_ids))} records available.")
        issues.append(f"PASS: {stats.get('referenced_record_count', 0)} records referenced.")
        if stats.get("all_records_exist"):
            issues.append("PASS: All referenced records exist.")
        if stats.get("all_categories_valid"):
            issues.append("PASS: All categories are valid.")
        if stats.get("all_behaviours_valid"):
            issues.append("PASS: All expected_behaviour values are valid.")

    return issues


def save_ground_truth(ground_truth: dict, path=None) -> Path:
    if path is None:
        path = GROUND_TRUTH_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    return path


def load_ground_truth(path=None) -> dict:
    if path is None:
        configs = sorted(GROUND_TRUTH_PATH.parent.glob("ground_truth_*.json"))
        if not configs:
            raise FileNotFoundError("No ground truth file found in benchmarks/retrieval/")
        path = configs[-1]
    with open(path) as f:
        return json.load(f)


def format_validation(issues: List[str]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("Ground Truth Validation")
    lines.append("=" * 60)
    for issue in issues:
        lines.append(f"  {issue}")
    lines.append("-" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating ground truth...")
    gt = generate_ground_truth()

    path = save_ground_truth(gt)
    print(f"Saved to: {path}")

    print("\nValidation:")
    issues = validate_ground_truth(gt)
    print(format_validation(issues))

    stats = gt["statistics"]
    print(f"\nStatistics:")
    print(f"  Version:            {stats['version']}")
    print(f"  Total queries:      {stats['total_queries']}")
    print(f"  Total records:      {stats['total_records']}")
    print(f"  PHI-targeting:      {stats['phi_targeting_queries']}")
    print(f"  General:            {stats['general_queries']}")
    print(f"  Referenced records: {stats['referenced_record_count']}")

    for cat, count in stats["category_distribution"].items():
        print(f"  Category '{cat}':   {count}")
    for subcat, count in stats["subcategory_distribution"].items():
        print(f"    - {subcat}:        {count}")
    for beh, count in stats["expected_behaviour_distribution"].items():
        print(f"  Behaviour '{beh}': {count}")
