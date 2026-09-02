import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

RETRIEVAL_DIR = Path(__file__).parent
BENCHMARK_DIR = RETRIEVAL_DIR.parent
PROJECT_DIR = BENCHMARK_DIR.parent

sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from secure_rag.detection import load_detector_stack
from secure_rag.masker import mask_text
from secure_rag.pdf_loader import chunk_text
from secure_rag.embedding import embed_chunks
from secure_rag.policies import load_policy
from secure_rag.vector_store import VectorStore

sys.path.insert(0, str(BENCHMARK_DIR))
from _common import load_records, load_queries, load_split, EVALUATION_CONFIGS
from retrieval.ground_truth import load_ground_truth

RUNNER_VERSION = "3"
RETRIEVAL_RESULTS_VERSION = "v1"
RETRIEVAL_RESULTS_VERSION_V2 = "v2"
MAX_K = 10
MAX_K_V1 = 10
MAX_K_V2 = 50
K_VALUES = [1, 3, 5, 10]
K_VALUES_V1 = [1, 3, 5, 10]
K_VALUES_V2 = [1, 3, 5, 10, 20, 30, 50]
RETRIEVAL_RESULTS_PATH = RETRIEVAL_DIR / f"retrieval_results_{RETRIEVAL_RESULTS_VERSION}.json"
RETRIEVAL_RESULTS_PATH_V2 = RETRIEVAL_DIR / f"retrieval_results_{RETRIEVAL_RESULTS_VERSION_V2}.json"


def _build_index_with_record_map(
    records: Dict[str, dict], use_masking: bool
) -> Tuple[VectorStore, List[str], List[str]]:
    if use_masking:
        medical_policy = load_policy("medical")
        medical_detectors = load_detector_stack("medical")
    texts_with_ids = []
    for rid in sorted(records.keys()):
        r = records[rid]
        text = r["text"]
        if use_masking:
            text = mask_text(text, policy=medical_policy, detectors=medical_detectors)
        texts_with_ids.append((rid, text))

    chunks = []
    chunk_record_map = []

    for rid, text in texts_with_ids:
        record_chunks = chunk_text(text)
        chunks.extend(record_chunks)
        chunk_record_map.extend([rid] * len(record_chunks))

    embeddings = embed_chunks(chunks)
    vector_store = VectorStore(embeddings)
    return vector_store, chunks, chunk_record_map


def _retrieve_top_k(
    query: str, vector_store: VectorStore, k: int
) -> Tuple[List[int], List[float]]:
    q_vec = embed_chunks([query])
    q_vec = np.array(q_vec).astype("float32")
    distances, indices = vector_store.search(q_vec, k=k)
    indices_list = indices if isinstance(indices, list) else indices[0].tolist() if hasattr(indices, 'tolist') else list(indices)
    distances_list = distances[0].tolist() if hasattr(distances, 'shape') and distances.ndim > 1 else distances.tolist() if hasattr(distances, 'tolist') else list(distances)
    if isinstance(distances_list, list) and len(distances_list) > 0 and isinstance(distances_list[0], list):
        distances_list = distances_list[0]
    if isinstance(indices_list, list) and len(indices_list) > 0 and isinstance(indices_list[0], list):
        indices_list = indices_list[0]
    return indices_list, distances_list


def run_retrieval(version: str = "v1") -> dict:
    print("=" * 60)
    print("Retrieval Runner — Secure RAG Retrieval Evaluation")
    print("=" * 60)

    retrieval_version = RETRIEVAL_RESULTS_VERSION_V2 if version == "v2" else RETRIEVAL_RESULTS_VERSION
    max_k = MAX_K_V2 if version == "v2" else MAX_K
    k_values = K_VALUES_V2 if version == "v2" else K_VALUES

    if version == "v2":
        from retrieval.ground_truth import _load_mrn_records_raw, GROUND_TRUTH_PATH_V2
        mrn_records = _load_mrn_records_raw()
        with open(GROUND_TRUTH_PATH_V2) as f:
            ground_truth = json.load(f)
        test_records = mrn_records
        print(f"\nLoaded {len(mrn_records)} MRN records (v2 full dataset)")
        gt_by_qid = {q["qid"]: q for q in ground_truth["queries"]}
        combined = []
        for qid, gt_entry in gt_by_qid.items():
            combined.append({
                "qid": qid,
                "question": gt_entry["question"],
                "category": gt_entry["category"],
                "subcategory": gt_entry["subcategory"],
                "ground_truth_records": gt_entry["relevant_records"],
                "expected_behaviour": gt_entry["expected_behaviour"],
            })
        combined.sort(key=lambda x: x["qid"])
    else:
        records = load_records()
        queries_data = load_queries()
        split = load_split()
        ground_truth = load_ground_truth()
        if ground_truth.get("version") == "v2":
            from pathlib import Path as _P
            gt_v1_path = RETRIEVAL_DIR / "ground_truth_v1.json"
            if _P(gt_v1_path).exists():
                with open(gt_v1_path) as f:
                    ground_truth = json.load(f)
        test_ids = split["test"]
        test_records = {rid: records[rid] for rid in test_ids if rid in records}
        print(f"\nLoaded {len(records)} records, {len(test_ids)} test records")

        gt_by_qid = {q["qid"]: q for q in ground_truth["queries"]}
        queries_by_qid = {}
        for group in queries_data:
            for q in group["queries"]:
                queries_by_qid[q["qid"]] = q

        combined = []
        for qid, gt_entry in gt_by_qid.items():
            q_entry = queries_by_qid.get(qid)
            if q_entry is None:
                continue
            combined.append({
                "qid": qid,
                "question": q_entry["question"],
                "category": gt_entry["category"],
                "subcategory": gt_entry["subcategory"],
                "ground_truth_records": gt_entry["relevant_records"],
                "expected_behaviour": gt_entry["expected_behaviour"],
            })
        combined.sort(key=lambda x: x["qid"])

    print(f"Evaluation queries: {len(combined)}")
    print(f"K values: {k_values}")
    print(f"Max K: {max_k}")
    if max_k > len(test_records):
        print(f"Note: MAX_K ({max_k}) capped at index size ({len(test_records)}) by VectorStore; K > index handled safely via min(k, ntotal)")

    print("\n" + "-" * 60)
    print("Building indices...")
    print("-" * 60)

    configs_meta = {}
    indices = {}

    for config in EVALUATION_CONFIGS:
        print(f"\n  [{config['id']}] {config['display_name']}")
        use_masking = config["id"] == "secure_rag"

        start = time.time()
        idx_type = "masked" if use_masking else "raw"
        vs, chunks, rec_map = _build_index_with_record_map(test_records, use_masking=use_masking)
        elapsed = time.time() - start

        chunks_per_record = {}
        for rid in rec_map:
            chunks_per_record[rid] = chunks_per_record.get(rid, 0) + 1

        configs_meta[config["id"]] = {
            "index_type": idx_type,
            "num_chunks": len(chunks),
            "num_records": len(test_records),
            "build_time_s": round(elapsed, 3),
            "chunks_per_record": chunks_per_record,
        }
        indices[config["id"]] = {
            "vector_store": vs,
            "chunks": chunks,
            "chunk_record_map": rec_map,
        }
        print(f"    Index: {len(chunks)} chunks from {len(test_records)} records ({elapsed:.2f}s)")
        print(f"    Chunk->Record map: {len(rec_map)} entries")
        unique_cpr = set(chunks_per_record.values())
        print(f"    Chunks per record: {unique_cpr} (record-level dedup required if >1)")

    print("\n" + "-" * 60)
    print("Running retrieval...")
    print("-" * 60)

    queries_output = []

    for i, entry in enumerate(combined):
        qid = entry["qid"]
        question = entry["question"]

        query_results = {}

        for config in EVALUATION_CONFIGS:
            cid = config["id"]
            idx_data = indices[cid]
            vs = idx_data["vector_store"]
            rec_map = idx_data["chunk_record_map"]

            indices_list, scores_list = _retrieve_top_k(question, vs, k=max_k)

            gt_records = entry["ground_truth_records"]
            gt_set = set(gt_records)
            retrieved = []
            seen_records = set()
            for rank, (chunk_idx, score) in enumerate(zip(indices_list, scores_list)):
                record_id = rec_map[chunk_idx] if 0 <= chunk_idx < len(rec_map) else "UNKNOWN"
                relevant = record_id in gt_set
                retrieved.append({
                    "chunk_index": int(chunk_idx),
                    "score": round(float(score), 4),
                    "record_id": record_id,
                    "rank": rank,
                    "relevant": relevant,
                })

            query_results[cid] = {
                "retrieved": retrieved,
            }

        queries_output.append({
            "qid": qid,
            "question": question,
            "category": entry["category"],
            "subcategory": entry["subcategory"],
            "ground_truth_records": entry["ground_truth_records"],
            "expected_behaviour": entry["expected_behaviour"],
            "results": query_results,
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(combined)} queries...")

    print(f"  Processed {len(combined)}/{len(combined)} queries — complete")

    evaluation_extra = {
        "total_records": len(test_records),
        "total_queries": len(combined),
        "dataset_split": "full" if version == "v2" else "test",
    }
    if version == "v2":
        evaluation_extra["note"] = "v2 uses full MRN dataset (120 records) with multi-record AGG queries; record-level set-based evaluation"

    retrieval_results = {
        "version": retrieval_version,
        "runner_version": RUNNER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": "Self-contained canonical retrieval artifact for Secure RAG retrieval evaluation. "
                       "Each query includes all ground truth metadata (category, subcategory, expected_behaviour). "
                       "Each retrieved item includes a relevance flag. "
                       "Record-level metrics must deduplicate record_id, not chunk count. "
                       "K > index size handled safely via VectorStore capping at ntotal. "
                       "Consumed by downstream phases: IR metrics, failure analysis, reporting.",
        "configs": configs_meta,
        "k_values": k_values,
        "max_k": max_k,
        "evaluation": evaluation_extra,
        "queries": queries_output,
    }

    return retrieval_results


def save_results(results: dict, path=None) -> Path:
    if path is None:
        version = results.get("version", RETRIEVAL_RESULTS_VERSION)
        if version == RETRIEVAL_RESULTS_VERSION_V2:
            path = RETRIEVAL_RESULTS_PATH_V2
        else:
            path = RETRIEVAL_RESULTS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_results(path=None, version: str = None) -> dict:
    if path is None:
        if version is not None:
            target = RETRIEVAL_RESULTS_PATH_V2 if version == "v2" else RETRIEVAL_RESULTS_PATH
            path = target
        else:
            configs = sorted(RETRIEVAL_RESULTS_PATH.parent.glob("retrieval_results_*.json"))
            if not configs:
                raise FileNotFoundError("No retrieval results file found.")
            path = configs[-1]
    with open(path) as f:
        return json.load(f)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.1f}m {seconds % 60:.0f}s"


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("RETRIEVAL RUNNER — SUMMARY")
    print("=" * 60)
    print(f"\n  Version:          {results['version']}")
    print(f"  Runner version:   {results['runner_version']}")
    print(f"  Queries:          {results['evaluation']['total_queries']}")
    print(f"  Test records:     {results['evaluation']['total_records']}")
    print(f"  K values:         {results['k_values']}")
    print(f"  Max K:            {results['max_k']}")
    print(f"\n  Configurations:")
    for cid, meta in results["configs"].items():
        print(f"    {cid:<15} {meta['index_type']:<10} {meta['num_chunks']:>4} chunks, "
              f"{meta['num_records']:>2} records  ({meta['build_time_s']:.2f}s)")

    out_path = RETRIEVAL_RESULTS_PATH_V2 if results.get("version") == RETRIEVAL_RESULTS_VERSION_V2 else RETRIEVAL_RESULTS_PATH
    print(f"\n  Output:           {out_path}")
    print(f"\n  Total results stored: {len(results['queries'])} query entries")
    print("\n" + "=" * 60)


def validate(results: dict) -> List[str]:
    issues = []

    version = results.get("version", RETRIEVAL_RESULTS_VERSION)
    expected_max_k = MAX_K_V2 if version == RETRIEVAL_RESULTS_VERSION_V2 else MAX_K

    if version not in (RETRIEVAL_RESULTS_VERSION, RETRIEVAL_RESULTS_VERSION_V2):
        issues.append(f"FAIL: Expected version v1 or v2, got {results['version']}")

    if not results["queries"]:
        issues.append("FAIL: No queries in retrieval results.")
        return issues

    expected_configs = {c["id"] for c in EVALUATION_CONFIGS}
    actual_configs = set(results["configs"].keys())
    if actual_configs != expected_configs:
        issues.append(f"FAIL: Config mismatch. Expected {expected_configs}, got {actual_configs}")

    seen_qids = set()
    for q_entry in results["queries"]:
        qid = q_entry["qid"]
        if qid in seen_qids:
            issues.append(f"FAIL: Duplicate qid: {qid}")
        seen_qids.add(qid)

        if "results" not in q_entry:
            issues.append(f"FAIL: {qid} has no results")
            continue

        for cid in expected_configs:
            if cid not in q_entry["results"]:
                issues.append(f"FAIL: {qid} missing config {cid}")
                continue

            retrieved = q_entry["results"][cid].get("retrieved", [])
            if len(retrieved) != expected_max_k and len(retrieved) != min(expected_max_k, results["configs"][cid]["num_chunks"]):
                issues.append(f"FAIL: {qid}/{cid}: expected {expected_max_k} retrieved, got {len(retrieved)}")

            for item in retrieved:
                if "chunk_index" not in item:
                    issues.append(f"FAIL: {qid}/{cid}: missing chunk_index")
                if "score" not in item:
                    issues.append(f"FAIL: {qid}/{cid}: missing score")
                if "record_id" not in item:
                    issues.append(f"FAIL: {qid}/{cid}: missing record_id")
                if "rank" not in item:
                    issues.append(f"FAIL: {qid}/{cid}: missing rank")
                if "relevant" not in item:
                    issues.append(f"FAIL: {qid}/{cid}: missing relevant flag")

        gt_records = q_entry.get("ground_truth_records", None)
        if gt_records is None:
            issues.append(f"FAIL: {qid} missing ground_truth_records")
        elif not isinstance(gt_records, list):
            issues.append(f"FAIL: {qid} ground_truth_records not a list")

        for meta_field in ("category", "subcategory", "expected_behaviour", "question"):
            if meta_field not in q_entry:
                issues.append(f"FAIL: {qid} missing query metadata: {meta_field}")

    if not issues:
        issues.append(f"PASS: {len(results['queries'])} queries across {len(actual_configs)} configs, "
                      f"{expected_max_k} retrieved per query — all valid.")

    return issues


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", choices=["v1", "v2"])
    args = parser.parse_args()

    print(f"Retrieval Runner — Phase 3 (version={args.version})")
    print()

    results = run_retrieval(version=args.version)
    path = save_results(results)
    print(f"\nRetrieval results saved to: {path}")

    print("\nValidating retrieval results...")
    v_issues = validate(results)
    for issue in v_issues:
        print(f"  {issue}")

    print_summary(results)
