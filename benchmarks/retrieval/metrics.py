import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

RETRIEVAL_DIR = Path(__file__).parent
BENCHMARK_DIR = RETRIEVAL_DIR.parent
METRICS_VERSION = "v2"
METRICS_FRAMEWORK_VERSION = "2"
METRICS_PATH = RETRIEVAL_DIR / f"metrics_{METRICS_VERSION}.json"

K_VALUES = [1, 3, 5, 10, 20, 30, 50]
METRIC_NAMES = ["hit_rate", "precision", "recall", "mrr"]


def hit_rate_at_k(retrieved: list, k: int, relevant_set=None) -> int:
    if relevant_set is not None:
        if len(relevant_set) == 0:
            return 0
        for item in retrieved[:k]:
            if item.get("record_id") in relevant_set:
                return 1
        return 0
    return 1 if any(item.get("relevant") for item in retrieved[:k]) else 0


def precision_at_k(retrieved: list, k: int, relevant_set=None) -> float:
    if k <= 0:
        return 0.0
    if relevant_set is not None:
        if len(relevant_set) == 0:
            return 0.0
        eff_k = min(k, len(retrieved)) if retrieved else k
        if eff_k == 0:
            return 0.0
        denom = k if len(retrieved) >= k else eff_k
        seen = set()
        count = 0
        for item in retrieved[:k]:
            rid = item.get("record_id")
            if rid in relevant_set and rid not in seen:
                seen.add(rid)
                count += 1
        return count / denom
    relevant = sum(1 for item in retrieved[:k] if item.get("relevant"))
    return relevant / k


def recall_at_k(retrieved: list, k: int, total_relevant: int = None, relevant_set=None) -> float:
    if relevant_set is not None:
        if len(relevant_set) == 0:
            return 0.0
        seen = set()
        for item in retrieved[:k]:
            rid = item.get("record_id")
            if rid in relevant_set:
                seen.add(rid)
        return len(seen) / len(relevant_set)
    if total_relevant is None:
        total_relevant = 0
    if total_relevant == 0:
        return 0.0
    relevant = sum(1 for item in retrieved[:k] if item.get("relevant"))
    return relevant / total_relevant


def mrr_at_k(retrieved: list, k: int, relevant_set=None) -> float:
    if relevant_set is not None:
        if len(relevant_set) == 0:
            return 0.0
        seen = set()
        for i, item in enumerate(retrieved[:k]):
            rid = item.get("record_id")
            if rid in seen:
                continue
            seen.add(rid)
            if rid in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    for i, item in enumerate(retrieved[:k]):
        if item.get("relevant"):
            return 1.0 / (i + 1)
    return 0.0


def compute_per_query_metrics(
    retrieval_results: dict,
) -> Tuple[dict, dict, list]:
    queries = retrieval_results["queries"]
    config_ids = list(retrieval_results["configs"].keys())
    k_values = retrieval_results["k_values"]

    per_query = {}
    for q_entry in queries:
        qid = q_entry["qid"]
        category = q_entry.get("category", "unknown")
        subcategory = q_entry.get("subcategory", "unknown")
        gt_records = q_entry.get("ground_truth_records", [])
        gt_set = set(gt_records)

        q_metrics = {
            "category": category,
            "subcategory": subcategory,
            "ground_truth_records": gt_records,
            "num_relevant": len(gt_set),
        }

        for cid in config_ids:
            retrieved = q_entry["results"][cid].get("retrieved", [])

            c_metrics = {}
            for k in k_values:
                c_metrics[f"k_{k}"] = {
                    "hit_rate": hit_rate_at_k(retrieved, k, gt_set),
                    "precision": precision_at_k(retrieved, k, gt_set),
                    "recall": recall_at_k(retrieved, k, relevant_set=gt_set),
                    "mrr": mrr_at_k(retrieved, k, gt_set),
                    "relevant_retrieved": len(set(item["record_id"] for item in retrieved[:k] if item["record_id"] in gt_set)),
                    "total_relevant": len(gt_set),
                }
            q_metrics[cid] = c_metrics

        per_query[qid] = q_metrics

    return per_query, config_ids, k_values


def _agg(values: list) -> dict:
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {
        "count": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _extract_metric_values(
    per_query: dict, config_id: str, k_key: str, metric: str
) -> List[float]:
    return [q[config_id][k_key][metric] for q in per_query.values()]


def compute_aggregated(
    per_query: dict, config_ids: list, k_values: list
) -> dict:
    aggregated = {}
    for cid in config_ids:
        c_agg = {}
        for k in k_values:
            k_key = f"k_{k}"

            overall = {}
            for m in METRIC_NAMES:
                vals = _extract_metric_values(per_query, cid, k_key, m)
                overall[m] = _agg(vals)

            categories = {}
            for qid, q in per_query.items():
                cat = q.get("category", "unknown")
                if cat not in categories:
                    categories[cat] = {m: [] for m in METRIC_NAMES}
                for m in METRIC_NAMES:
                    categories[cat][m].append(q[cid][k_key][m])

            by_category = {}
            for cat, cat_vals in sorted(categories.items()):
                by_category[cat] = {m: _agg(cat_vals[m]) for m in METRIC_NAMES}

            subcategories = {}
            for qid, q in per_query.items():
                sub = q.get("subcategory", "unknown")
                if sub not in subcategories:
                    subcategories[sub] = {m: [] for m in METRIC_NAMES}
                for m in METRIC_NAMES:
                    subcategories[sub][m].append(q[cid][k_key][m])

            by_subcategory = {}
            for sub, sub_vals in sorted(subcategories.items()):
                by_subcategory[sub] = {m: _agg(sub_vals[m]) for m in METRIC_NAMES}

            c_agg[k_key] = {
                "overall": overall,
                "by_category": by_category,
                "by_subcategory": by_subcategory,
            }
        aggregated[cid] = c_agg
    return aggregated


def compute_comparisons(
    aggregated: dict, config_ids: list, k_values: list
) -> dict:
    comparisons = {}
    pairs = []
    if "baseline_a" in config_ids and "secure_rag" in config_ids:
        pairs.append(("baseline_a", "secure_rag", "baseline_a_vs_secure_rag"))
    if "baseline_b" in config_ids and "secure_rag" in config_ids:
        pairs.append(("baseline_b", "secure_rag", "baseline_b_vs_secure_rag"))

    for num_id, den_id, label in pairs:
        for k in k_values:
            k_key = f"k_{k}"
            comp = {}
            for m in METRIC_NAMES:
                num_mean = aggregated[num_id][k_key]["overall"][m]["mean"]
                den_mean = aggregated[den_id][k_key]["overall"][m]["mean"]
                if den_mean > 0:
                    comp[f"{m}_degradation"] = round(num_mean / den_mean, 6)
                    comp[f"{m}_absolute_diff"] = round(num_mean - den_mean, 6)
                else:
                    comp[f"{m}_degradation"] = None
                    comp[f"{m}_absolute_diff"] = round(num_mean - den_mean, 6)

            comparisons[f"{label}_{k_key}"] = comp

    return comparisons


def compute_metrics(retrieval_results: dict = None, version: str = None) -> dict:
    if retrieval_results is None:
        retrieval_results = _load_retrieval_results()

    per_query, config_ids, k_values = compute_per_query_metrics(retrieval_results)

    aggregated = compute_aggregated(per_query, config_ids, k_values)

    comparisons = compute_comparisons(aggregated, config_ids, k_values)

    inferred_version = retrieval_results.get("version", METRICS_VERSION)
    if version is None:
        version = inferred_version if inferred_version in ("v1", "v2") else METRICS_VERSION
    framework_version = "1" if version == "v1" else METRICS_FRAMEWORK_VERSION
    source_artifact = f"retrieval_results_{version}.json"

    metrics = {
        "version": version,
        "metrics_framework_version": framework_version,
        "source_artifact": source_artifact,
        "source_runner_version": retrieval_results.get("runner_version", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "description": (
            "Canonical IR metrics for Secure RAG retrieval evaluation. "
            "Metrics are computed from the self-contained retrieval artifact. "
            "Record-level set-based evaluation: relevant_records is a set of record IDs. "
            "Precision = |retrieved_records ∩ relevant| / k (unique records, K capped at index size if K > n). "
            "Recall = |retrieved_records ∩ relevant| / |relevant| (0 if |relevant|==0). "
            "HitRate = 1 if intersection non-empty else 0 (0 if |relevant|==0). "
            "MRR = 1/rank_of_first_relevant (unique record rank, 0 if none or |relevant|==0). "
            "Zero-relevant queries return 0 for all metrics. Duplicate retrieved/ground-truth IDs deduplicated via set. "
            "No retrieval, embedding, or LLM calls were performed."
        ),
        "k_values": k_values,
        "config_ids": config_ids,
        "metric_definitions": {
            "hit_rate": "1 if at least one relevant record in top-k else 0 (0 if no relevant records)",
            "precision": "Unique relevant retrieved records / k (k capped at actual retrieved length if K > index size; 0 if no relevant records)",
            "recall": "Unique relevant retrieved records / total relevant records (0 if no relevant records)",
            "mrr": "1/rank of first relevant record (unique record order, 0 if none or no relevant records)",
        },
        "per_query": per_query,
        "aggregated": aggregated,
        "comparisons": comparisons,
    }

    return metrics


def _load_retrieval_results() -> dict:
    from benchmarks.retrieval.runner import load_results as load_retrieval_results

    return load_retrieval_results()


def save_metrics(metrics: dict, path=None) -> Path:
    if path is None:
        path = METRICS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(path=None) -> dict:
    if path is None:
        if METRICS_PATH.exists():
            path = METRICS_PATH
        else:
            configs = sorted(METRICS_PATH.parent.glob("metrics_*.json"))
            if not configs:
                raise FileNotFoundError("No metrics file found.")
            path = configs[-1]
    with open(path) as f:
        return json.load(f)


def validate(metrics: dict) -> List[str]:
    issues = []

    version = metrics.get("version", METRICS_VERSION)
    expected_k = [1, 3, 5, 10] if version == "v1" else K_VALUES

    config_ids = metrics.get("config_ids", [])
    expected_configs = {"baseline_a", "baseline_b", "secure_rag"}
    if set(config_ids) != expected_configs:
        issues.append(f"FAIL: Config mismatch. Expected {expected_configs}, got {set(config_ids)}")

    k_values = metrics.get("k_values", [])
    if set(k_values) != set(expected_k):
        issues.append(f"FAIL: K values mismatch. Expected {expected_k}, got {k_values}")

    per_query = metrics.get("per_query", {})
    if not per_query:
        issues.append("FAIL: No per-query metrics.")

    aggregate_sections = 0
    for cid in config_ids:
        agg = metrics.get("aggregated", {}).get(cid, {})
        if not agg:
            issues.append(f"FAIL: No aggregated metrics for {cid}")
            continue
        for k in k_values:
            k_key = f"k_{k}"
            if k_key not in agg:
                issues.append(f"FAIL: {cid} missing {k_key}")
                continue
            section = agg[k_key]
            if "overall" not in section:
                issues.append(f"FAIL: {cid}/{k_key} missing overall")
                continue
            for m in METRIC_NAMES:
                if m not in section["overall"]:
                    issues.append(f"FAIL: {cid}/{k_key} missing metric {m}")
                    continue
                agg_info = section["overall"][m]
                if "mean" not in agg_info:
                    issues.append(f"FAIL: {cid}/{k_key}/{m} missing mean")
            if "by_category" not in section:
                issues.append(f"FAIL: {cid}/{k_key} missing by_category")
            if "by_subcategory" not in section:
                issues.append(f"FAIL: {cid}/{k_key} missing by_subcategory")
        aggregate_sections += 1

    if "comparisons" not in metrics:
        issues.append("FAIL: Missing comparisons section")
    else:
        expected_comp_count = 2 * len(k_values)
        actual_comp_count = len(metrics["comparisons"])
        if actual_comp_count < expected_comp_count:
            issues.append(f"FAIL: Expected ~{expected_comp_count} comparisons, got {actual_comp_count}")

    if not issues:
        issues.append(f"PASS: {len(per_query)} queries, {len(config_ids)} configs, "
                      f"{len(k_values)} k values, {aggregate_sections} aggregate sections — all valid.")

    return issues


def print_summary(metrics: dict):
    print("\n" + "=" * 60)
    print("IR METRICS — SUMMARY")
    print("=" * 60)
    print(f"\n  Source artifact:   {metrics['source_artifact']} (runner v{metrics['source_runner_version']})")
    print(f"  Queries:           {len(metrics['per_query'])}")

    for cid in metrics["config_ids"]:
        print(f"\n  [{cid}]")
        for k in metrics["k_values"]:
            k_key = f"k_{k}"
            o = metrics["aggregated"][cid][k_key]["overall"]
            print(f"    k={k}:  HitRate={o['hit_rate']['mean']:.4f}  "
                  f"Prec={o['precision']['mean']:.4f}  "
                  f"Recall={o['recall']['mean']:.4f}  "
                  f"MRR={o['mrr']['mean']:.4f}")

    print(f"\n  Comparisons (degradation vs Secure RAG):")
    for comp_key, comp in metrics.get("comparisons", {}).items():
        parts = comp_key.split("_k_")
        label = parts[0]
        k_val = parts[1] if len(parts) > 1 else "?"
        degs = "  ".join(
            f"{m}_degrad={comp.get(m + '_degradation', 'N/A')}" for m in METRIC_NAMES
        )
        print(f"    {label:<35} k={k_val}  {degs}")

    print(f"\n  Output:           {METRICS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print("IR Metrics — Phase 4")
    print()

    metrics = compute_metrics()
    path = save_metrics(metrics)
    print(f"Metrics saved to: {path}")

    print("\nValidating metrics...")
    v_issues = validate(metrics)
    for issue in v_issues:
        print(f"  {issue}")

    print_summary(metrics)
