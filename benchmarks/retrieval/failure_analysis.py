"""
Retrieval Failure Analysis — Secure RAG retrieval evaluation.

Consumes metrics_v1.json + retrieval_results_v1.json to explain WHY
retrieval succeeds or fails for each query. No retrieval occurs.
No metrics are recomputed.

Taxonomy (from Phase 1 design):
  Entity Retrieval Failure
    Diagnosis | Hospital | Medication | Treatment | Demographics | Contact/Identifier
  General Query Failure
    Summary | Factual/Attribute
  Ranking Failure
    Low Similarity Score | Competitor Record Higher
  Chunk Boundary Failure
    Answer Fragmented | Context Lost
  Masking Degradation
    Entity Name Masked | Location Masked | Identifier Masked
  Embedding Similarity Failure
    Vocabulary Mismatch | Domain Drift

AUDIT — Multi-record (v2) implications (P0 STEP 4):
  Single-record assumption: one query -> one relevant record, binary hit@k.

  1. entity_retrieval_failure / embedding_similarity_failure:
     Current: binary hit_rate@10 ==0 => failure. For multi-record, a query may
     retrieve 6/17 relevant records (recall 0.35) and still be classified as
     "succeeded" by hit_rate==1, hiding partial failure. Conversely recall<1
     is now the key signal. Future taxonomy should add:
       - partial_retrieval_failure (0 < recall < 1)
       - complete_miss (recall==0) vs incomplete_coverage
       - separate categories per expected_behaviour multi_record_retrieval

  2. masking_degradation:
     Current: baseline_a hit@10==1 and secure_rag hit@10==0 => masking_degradation.
     Under multi-record this binary comparison misses graded degradation:
     e.g., baseline recall@50=0.8 vs secure recall@50=0.4. Future should compare
     recall@k deltas and precision, not just hit_rate.

  3. ranking_failure handling assumes single ground truth rank. With N relevant,
     need to consider distribution of ranks (MRR, median rank, coverage curve).
     Current gt_ranks = [rank where record_id in gt_records] is already multi-aware
     but threshold check (min_rank >=10) is insufficient.

  No taxonomy redesign implemented in STEP 4; this audit documents required changes
  for P0 STEP 5. Metrics to run remain record-level set-based; failure_analysis
  continues to classify via hit_rate@10 for backward compat.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

RETRIEVAL_DIR = Path(__file__).parent
BENCHMARK_DIR = RETRIEVAL_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

FAILURE_ANALYSIS_VERSION = "v1"
FAILURE_ANALYSIS_FRAMEWORK_VERSION = "1"
FAILURE_ANALYSIS_PATH = RETRIEVAL_DIR / f"failure_analysis_{FAILURE_ANALYSIS_VERSION}.json"
FAILURE_ANALYSIS_VERSION_V2 = "v2"
FAILURE_ANALYSIS_FRAMEWORK_VERSION_V2 = "2"
FAILURE_ANALYSIS_PATH_V2 = RETRIEVAL_DIR / f"failure_analysis_{FAILURE_ANALYSIS_VERSION_V2}.json"

RECALL_TIER_COMPLETE = "complete_retrieval"
RECALL_TIER_PARTIAL = "partial_retrieval"
RECALL_TIER_ZERO = "zero_retrieval"
RECALL_TIERS = (RECALL_TIER_COMPLETE, RECALL_TIER_PARTIAL, RECALL_TIER_ZERO)


def get_recall_tier(recall: float) -> str:
    if recall == 1.0:
        return RECALL_TIER_COMPLETE
    if recall == 0.0 or recall == 0:
        return RECALL_TIER_ZERO
    return RECALL_TIER_PARTIAL

SUBCATEGORY_TO_ENTITY_TYPE = {
    "factual_hospital": "hospital",
    "summary": None,
    "phi_aadhaar": "contact_identifier",
    "phi_phone": "contact_identifier",
    "phi_mrn": "contact_identifier",
}

SUBCATEGORY_TO_TAXONOMY = {
    "factual_hospital": ["entity_retrieval_failure", "hospital"],
    "summary": ["general_query_failure", "summary"],
    "phi_aadhaar": ["entity_retrieval_failure", "contact_identifier"],
    "phi_phone": ["entity_retrieval_failure", "contact_identifier"],
    "phi_mrn": ["entity_retrieval_failure", "contact_identifier"],
}


def load_retrieval_results() -> dict:
    from retrieval.runner import load_results
    return load_results()


def load_metrics() -> dict:
    from retrieval.metrics import load_metrics
    return load_metrics()


SUBCATEGORY_ENTITY_MAP = {
    "factual_hospital": {"primary": "entity_retrieval_failure", "specific": "hospital"},
    "summary": {"primary": "general_query_failure", "specific": "summary"},
    "phi_aadhaar": {"primary": "entity_retrieval_failure", "specific": "contact_identifier"},
    "phi_phone": {"primary": "entity_retrieval_failure", "specific": "contact_identifier"},
    "phi_mrn": {"primary": "entity_retrieval_failure", "specific": "contact_identifier"},
}


def _entity_type_label(subcategory: str) -> str:
    return SUBCATEGORY_ENTITY_MAP.get(subcategory, {}).get("specific", "unknown")


def classify_query_failure(
    qid: str,
    pq_metrics: dict,
    retrieval_entry: dict,
) -> dict:
    category = pq_metrics.get("category", "unknown")
    subcategory = pq_metrics.get("subcategory", "unknown")
    gt_records = pq_metrics.get("ground_truth_records", [])
    question = retrieval_entry.get("question", "")
    expected_behaviour = retrieval_entry.get("expected_behaviour", "unknown")
    num_relevant = pq_metrics.get("num_relevant", len(gt_records))

    config_results = {}
    overall_failed = False
    any_succeeded = False

    for cid in ["baseline_a", "baseline_b", "secure_rag"]:
        h10 = pq_metrics[cid]["k_10"]["hit_rate"]
        succeeded = h10 == 1
        failed = not succeeded
        if succeeded:
            any_succeeded = True
        if failed:
            overall_failed = True

        k10_metrics = pq_metrics[cid].get("k_10", {})
        recall = k10_metrics.get("recall", 0.0)
        relevant_retrieved = k10_metrics.get("relevant_retrieved", 0)
        total_relevant = k10_metrics.get("total_relevant", num_relevant)
        recall_tier = get_recall_tier(recall)

        retrieved = retrieval_entry["results"][cid].get("retrieved", [])
        gt_ranks = [
            item["rank"] for item in retrieved
            if item["record_id"] in gt_records
        ]
        gt_retrieved_anywhere = len(gt_ranks) > 0
        min_gt_rank = min(gt_ranks) if gt_ranks else None

        failures = []

        if failed:
            if min_gt_rank is not None and min_gt_rank >= 10:
                failures.append({
                    "category": "ranking_failure",
                    "specific": "competitor_record_higher",
                    "evidence": {
                        "ground_truth_rank": min_gt_rank,
                        "k_analyzed": 10,
                    },
                    "rationale": (
                        f"Ground truth record {'/'.join(gt_records)} was retrieved "
                        f"at rank {min_gt_rank} (beyond top-10). "
                        f"Other records scored higher in embedding similarity."
                    ),
                })
            elif min_gt_rank is not None and min_gt_rank < 10:
                failures.append({
                    "category": "ranking_failure",
                    "specific": "competitor_record_higher",
                    "evidence": {
                        "ground_truth_rank": min_gt_rank,
                        "k_analyzed": 10,
                    },
                    "rationale": (
                        f"Ground truth was retrieved at rank {min_gt_rank} which is within top-10, "
                        f"but hit_rate@10=0 indicates the record was not captured at rank < 10. "
                        f"However metrics show hit_rate@10=0 which is inconsistent. "
                        f"This suggests a mismatch in relevance assignment."
                    ),
                })
            else:
                if category == "phi_targeting":
                    entity_type = _entity_type_label(subcategory)
                    failures.append({
                        "category": "entity_retrieval_failure",
                        "specific": entity_type,
                        "evidence": {
                            "ground_truth_retrieved": False,
                            "expected_behaviour": expected_behaviour,
                        },
                        "rationale": (
                            f"Query targets {entity_type.replace('_', ' ')} but "
                            f"the ground truth record {'/'.join(gt_records)} was not "
                            f"retrieved within the top-10 results. The embedding model "
                            f"did not match the query to the correct patient record."
                        ),
                    })
                elif category == "general":
                    failures.append({
                        "category": "embedding_similarity_failure",
                        "specific": "vocabulary_mismatch",
                        "evidence": {
                            "ground_truth_retrieved": False,
                            "expected_behaviour": expected_behaviour,
                        },
                        "rationale": (
                            f"General query '{question[:60]}...' did not retrieve the "
                            f"ground truth record. The query vocabulary likely overlaps "
                            f"with multiple records due to similar medical content."
                        ),
                    })

        if cid == "secure_rag" and pq_metrics["baseline_a"]["k_10"]["hit_rate"] == 1 and failed:
            failures.append({
                "category": "masking_degradation",
                "specific": "entity_name_masked",
                "evidence": {
                    "baseline_a_succeeded": True,
                    "secure_rag_failed": True,
                },
                "rationale": (
                    "Pre-embedding masking removed entity tokens from the indexed chunks, "
                    "causing the embedding to differ from the non-masked variant. "
                    "The raw index (Baseline A) successfully retrieved the ground truth record "
                    "while the masked index (Secure RAG) did not."
                ),
            })

        config_results[cid] = {
            "succeeded": succeeded,
            "hit_rate_10": h10,
            "failures": failures,
            "ground_truth_retrieved": gt_retrieved_anywhere,
            "ground_truth_min_rank": min_gt_rank,
            "recall": recall,
            "recall_tier": recall_tier,
            "relevant_retrieved": relevant_retrieved,
            "total_relevant": total_relevant,
        }

    per_query_recall_tiers = {cid: config_results[cid]["recall_tier"] for cid in config_results}
    is_multi = num_relevant > 1
    is_single = num_relevant == 1

    return {
        "qid": qid,
        "question": question,
        "category": category,
        "subcategory": subcategory,
        "expected_behaviour": expected_behaviour,
        "num_relevant": num_relevant,
        "is_multi_record": is_multi,
        "is_single_record": is_single,
        "overall_failed": overall_failed,
        "any_succeeded": any_succeeded,
        "recall_tiers": per_query_recall_tiers,
        "configs": config_results,
    }


def run_failure_analysis(metrics: dict = None, retrieval_results: dict = None, version: str = None) -> dict:
    if metrics is None:
        metrics = load_metrics()
    if retrieval_results is None:
        retrieval_results = load_retrieval_results()

    inferred_version = metrics.get("version", FAILURE_ANALYSIS_VERSION)
    if version is None:
        version = inferred_version if inferred_version in (FAILURE_ANALYSIS_VERSION, FAILURE_ANALYSIS_VERSION_V2) else FAILURE_ANALYSIS_VERSION

    per_query_metrics = metrics["per_query"]
    retrieval_by_qid = {q["qid"]: q for q in retrieval_results["queries"]}

    analyses = {}
    for qid, pq in per_query_metrics.items():
        r_entry = retrieval_by_qid.get(qid)
        if r_entry is None:
            continue
        analyses[qid] = classify_query_failure(qid, pq, r_entry)

    category_counts: Dict[str, int] = {}
    config_category_counts: Dict[str, Dict[str, int]] = {}
    subcategory_counts: Dict[str, int] = {}
    recall_tier_counts: Dict[str, int] = {}
    config_recall_tier_counts: Dict[str, Dict[str, int]] = {}
    multi_recall_tier_counts: Dict[str, int] = {}
    single_recall_tier_counts: Dict[str, int] = {}

    for cid in ["baseline_a", "baseline_b", "secure_rag"]:
        config_category_counts[cid] = {}
        config_recall_tier_counts[cid] = {t: 0 for t in RECALL_TIERS}

    for qid, analysis in analyses.items():
        for cid in ["baseline_a", "baseline_b", "secure_rag"]:
            for failure in analysis["configs"][cid]["failures"]:
                cat = failure["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
                config_category_counts[cid][cat] = config_category_counts[cid].get(cat, 0) + 1

        subcat = analysis.get("subcategory", "unknown")
        if analysis["overall_failed"]:
            subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

        for cid in ["baseline_a", "baseline_b", "secure_rag"]:
            tier = analysis["configs"][cid].get("recall_tier", RECALL_TIER_ZERO)
            recall_tier_counts[tier] = recall_tier_counts.get(tier, 0) + 1
            config_recall_tier_counts[cid][tier] = config_recall_tier_counts[cid].get(tier, 0) + 1
            if analysis.get("is_multi_record"):
                multi_recall_tier_counts[tier] = multi_recall_tier_counts.get(tier, 0) + 1
            elif analysis.get("is_single_record"):
                single_recall_tier_counts[tier] = single_recall_tier_counts.get(tier, 0) + 1

    per_config_summary = {}
    for cid in ["baseline_a", "baseline_b", "secure_rag"]:
        total = sum(1 for a in analyses.values() if a["configs"][cid]["hit_rate_10"] == 0)
        tier_dist = config_recall_tier_counts[cid]
        total_c = len(analyses)
        per_config_summary[cid] = {
            "total_failures": total,
            "total_queries": total_c,
            "failure_rate": round(total / total_c, 6) if analyses else 0.0,
            "categories": {},
            "recall_tiers": dict(sorted(tier_dist.items())),
            "recall_tier_rates": {k: round(v / total_c, 6) if total_c else 0.0 for k, v in tier_dist.items()},
        }
        for cat, count in sorted(config_category_counts[cid].items()):
            per_config_summary[cid]["categories"][cat] = count

    total_failures = sum(1 for a in analyses.values() if a["overall_failed"])

    is_v2 = version == FAILURE_ANALYSIS_VERSION_V2
    framework_version = FAILURE_ANALYSIS_FRAMEWORK_VERSION_V2 if is_v2 else FAILURE_ANALYSIS_FRAMEWORK_VERSION
    src_metrics = f"metrics_{version}.json"
    src_results = f"retrieval_results_{version}.json"

    taxonomy = {
        "entity_retrieval_failure": "Query specifies a named entity but the correct record was not retrieved",
        "general_query_failure": "General/summary query failed to retrieve the correct record",
        "ranking_failure": "Correct record retrieved but ranked below k threshold",
        "masking_degradation": "Pre-embedding masking caused retrieval failure in Secure RAG",
        "embedding_similarity_failure": "Semantic mismatch between query and record embeddings",
    }
    if is_v2:
        taxonomy.update({
            RECALL_TIER_COMPLETE: "All relevant records retrieved (recall == 1.0)",
            RECALL_TIER_PARTIAL: "Some but not all relevant records retrieved (0 < recall < 1.0)",
            RECALL_TIER_ZERO: "No relevant records retrieved (recall == 0.0)",
        })

    statistics = {
        "total_queries": len(analyses),
        "total_failures": total_failures,
        "overall_failure_rate": round(total_failures / len(analyses), 6) if analyses else 0.0,
        "failure_categories": dict(sorted(category_counts.items())),
        "failure_by_subcategory": dict(sorted(subcategory_counts.items())),
        "per_config": per_config_summary,
    }
    if is_v2:
        multi_total = sum(1 for a in analyses.values() if a.get("is_multi_record"))
        single_total = sum(1 for a in analyses.values() if a.get("is_single_record"))
        statistics.update({
            "recall_tiers_overall": dict(sorted(recall_tier_counts.items())),
            "recall_tiers_multi_record": dict(sorted(multi_recall_tier_counts.items())),
            "recall_tiers_single_record": dict(sorted(single_recall_tier_counts.items())),
            "multi_record_queries": multi_total,
            "single_record_queries": single_total,
            "recall_tier_definition": {
                RECALL_TIER_COMPLETE: "recall == 1.0",
                RECALL_TIER_PARTIAL: "0 < recall < 1.0",
                RECALL_TIER_ZERO: "recall == 0.0",
            },
        })

    description = (
        "Canonical failure analysis artifact for Secure RAG retrieval evaluation. "
        "Classifies retrieval failures using the approved taxonomy. "
        "Each failure includes category, specific type, supporting evidence, and rationale."
    )
    if is_v2:
        description += (
            " v2 adds recall-aware tiers (complete/partial/zero) computed from recall = retrieved_relevant / total_relevant at k=10. "
            "Single-record queries retain binary recall 0/1 semantics. Multi-record queries are classified by coverage."
        )

    result = {
        "version": version,
        "framework_version": framework_version,
        "source_artifacts": {
            "metrics": src_metrics,
            "retrieval_results": src_results,
        },
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": description,
        "taxonomy": taxonomy,
        "statistics": statistics,
        "per_query": analyses,
    }

    return result


def save_failure_analysis(analysis: dict, path=None) -> Path:
    if path is None:
        version = analysis.get("version", FAILURE_ANALYSIS_VERSION)
        if version == FAILURE_ANALYSIS_VERSION_V2:
            path = FAILURE_ANALYSIS_PATH_V2
        else:
            path = FAILURE_ANALYSIS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    return path


def load_failure_analysis(path=None) -> dict:
    if path is None:
        configs = sorted(FAILURE_ANALYSIS_PATH.parent.glob("failure_analysis_*.json"))
        if not configs:
            raise FileNotFoundError("No failure analysis file found.")
        path = configs[-1]
    with open(path) as f:
        return json.load(f)


def validate(analysis: dict) -> List[str]:
    issues = []

    if analysis["version"] not in (FAILURE_ANALYSIS_VERSION, FAILURE_ANALYSIS_VERSION_V2):
        issues.append(f"FAIL: Expected version v1 or v2, got {analysis['version']}")

    per_query = analysis.get("per_query", {})
    if not per_query:
        issues.append("FAIL: No per-query analyses.")
        return issues

    pq_count = len(per_query)
    expected_configs = {"baseline_a", "baseline_b", "secure_rag"}

    seen_qids = set()
    for qid, a in per_query.items():
        if qid in seen_qids:
            issues.append(f"FAIL: Duplicate qid: {qid}")
        seen_qids.add(qid)

        actual_configs = set(a.get("configs", {}).keys())
        if actual_configs != expected_configs:
            issues.append(f"FAIL: {qid} configs mismatch ({actual_configs})")

        for cid in expected_configs:
            conf = a["configs"].get(cid, {})
            if "succeeded" not in conf:
                issues.append(f"FAIL: {qid}/{cid} missing succeeded")
            if "failures" not in conf:
                issues.append(f"FAIL: {qid}/{cid} missing failures")
            else:
                for f in conf["failures"]:
                    if "category" not in f:
                        issues.append(f"FAIL: {qid}/{cid} failure missing category")
                    if "rationale" not in f:
                        issues.append(f"FAIL: {qid}/{cid} failure missing rationale")

    stats = analysis.get("statistics", {})
    if stats.get("total_queries") != pq_count:
        issues.append(f"FAIL: statistics.total_queries ({stats.get('total_queries')}) != actual ({pq_count})")

    if "failure_categories" not in stats:
        issues.append("FAIL: missing failure_categories in statistics")
    if "per_config" not in stats:
        issues.append("FAIL: missing per_config in statistics")

    if not issues:
        issues.append(f"PASS: {pq_count} queries analyzed, "
                      f"{stats.get('total_failures', 0)} failures classified, "
                      f"{len(stats.get('failure_categories', {}))} category types.")

    return issues


def print_summary(analysis: dict):
    stats = analysis["statistics"]
    version = analysis.get("version", "unknown")
    out_path = FAILURE_ANALYSIS_PATH_V2 if version == FAILURE_ANALYSIS_VERSION_V2 else FAILURE_ANALYSIS_PATH
    print("\n" + "=" * 60)
    print(f"FAILURE ANALYSIS — SUMMARY (v{version})")
    print("=" * 60)
    print(f"\n  Total queries:     {stats['total_queries']}")
    print(f"  Total failures:    {stats['total_failures']}")
    print(f"  Failure rate:      {stats['overall_failure_rate']:.2%}")

    print(f"\n  Failure categories:")
    for cat, count in sorted(stats["failure_categories"].items()):
        print(f"    {cat:<35} {count:>4}")

    print(f"\n  Per-config failures:")
    for cid, csum in stats["per_config"].items():
        cats = "  ".join(f"{k}={v}" for k, v in sorted(csum["categories"].items()))
        print(f"    {cid:<15} {csum['total_failures']:>4}/{csum['total_queries']}  "
              f"({csum['failure_rate']:.1%})  {cats}")

    print(f"\n  Failures by subcategory:")
    for sub, count in sorted(stats.get("failure_by_subcategory", {}).items()):
        print(f"    {sub:<25} {count:>4}")

    if "recall_tiers_overall" in stats:
        print(f"\n  Recall tiers (overall, all configs combined):")
        for tier, count in sorted(stats["recall_tiers_overall"].items()):
            print(f"    {tier:<25} {count:>4}")
        print(f"\n  Recall tiers — multi-record queries:")
        for tier, count in sorted(stats.get("recall_tiers_multi_record", {}).items()):
            print(f"    {tier:<25} {count:>4}")
        print(f"\n  Recall tiers — single-record queries:")
        for tier, count in sorted(stats.get("recall_tiers_single_record", {}).items()):
            print(f"    {tier:<25} {count:>4}")
        print(f"\n  Per-config recall tiers:")
        for cid, csum in stats["per_config"].items():
            tiers = "  ".join(f"{k}={v}" for k, v in sorted(csum.get("recall_tiers", {}).items()))
            print(f"    {cid:<15} {tiers}")

    print(f"\n  Output:           {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    args = parser.parse_args()

    print(f"Retrieval Failure Analysis — Phase 5 (version={args.version})")
    print()

    analysis = run_failure_analysis(version=args.version)
    path = save_failure_analysis(analysis)
    print(f"Failure analysis saved to: {path}")

    print("\nValidating failure analysis...")
    v_issues = validate(analysis)
    for issue in v_issues:
        print(f"  {issue}")

    print_summary(analysis)
