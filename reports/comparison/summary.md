# Secure-RAG: Dense vs BM25 vs Hybrid — Comparison Summary

## 1. Purpose

We compared three retrieval approaches:

1. **Dense Retrieval** — Sentence Transformers + FAISS
2. **BM25** — lexical/keyword retrieval
3. **Hybrid** — Dense + BM25 using Reciprocal Rank Fusion (RRF)

The goal was to determine whether BM25 or Hybrid meaningfully improves on Dense retrieval and whether privacy masking affects retrieval quality.

## 2. Evaluation Setup

- Dataset: **120 medical patient records**
- Queries: **605 total**
  - 601 single-target queries
  - 5 aggregate queries
  - 4 genuinely multi-record aggregate queries
  - 1 aggregate query with a single relevant record
- Metrics: HitRate, Precision, Recall, MRR
- Retrieval depths: **k = 2, 5, 10, 20, 30, 50**
- Same dataset and ground-truth basis across the three experiments

**Important:** The accurate breakdown is **601 single-target + 4 genuinely multi-record aggregate + 1 single-target aggregate = 605**.

---

## 3. Overall Result

### Main finding

**Dense and Hybrid perform almost identically.**

BM25 is competitive, but it does not provide a meaningful overall improvement over Dense.

At **k=10**:

| Method | HitRate | Precision | Recall | MRR |
|---|---:|---:|---:|---:|
| Dense masked | 0.0909 | 0.0145 | 0.0888 | 0.0325 |
| BM25 masked | 0.0909 | 0.0144 | 0.0887 | 0.0308 |
| Hybrid masked | 0.0909 | 0.0145 | 0.0888 | 0.0325 |

At **k=50**, all three reach approximately the same overall recall/HitRate (~0.4215), while Dense and Hybrid retain slightly better MRR than BM25.

### Easy explanation

> **Adding BM25, or combining BM25 with Dense, did not substantially improve retrieval on this dataset. Dense retrieval is already competitive.**

---

## 4. Single-Record Queries

There are **601 single-target queries**, and all three methods perform poorly.

At **k=10**:

| Method | Recall / HitRate | Precision | MRR |
|---|---:|---:|---:|
| Dense | 0.0849 | 0.0085 | 0.0260 |
| BM25 | 0.0849 | 0.0085 | 0.0252 |
| Hybrid | 0.0849 | 0.0085 | 0.0260 |

The main problem is **query formulation**, not clearly the retrieval algorithm.

Many queries are generic and repeated across records, such as:

- "Summarize the patient record."
- "Which hospital did the patient visit?"
- "What is the patient's Aadhaar number?"

These queries do not contain enough information to identify the intended patient/MRN.

### Interpretation

> **The single-record results expose an evaluation/query-design limitation. They should not be treated as proof that Dense, BM25, or Hybrid is intrinsically poor.**

---

## 5. Multi-Record / Aggregate Queries

This is where **retrieval depth** becomes important.

At **k=2**, there are not enough retrieved records to achieve high recall for queries with many relevant records.

At **k=10**:

| Method | Recall | Precision | MRR |
|---|---:|---:|---:|
| Dense | 0.6783 | 0.9250 | 1.0000 |
| BM25 | 0.6636 | 0.9000 | 0.8750 |
| Hybrid | 0.6783 | 0.9250 | 1.0000 |

At **k=20**:

| Method | Recall | Precision | MRR |
|---|---:|---:|---:|
| Dense | 0.9728 | 0.7250 | 1.0000 |
| BM25 | 1.0000 | 0.7500 | 0.8750 |
| Hybrid | 0.9875 | 0.7375 | 1.0000 |

### Interpretation

> **The major limitation is retrieval depth, not the choice between Dense, BM25, and Hybrid.**

Increasing k greatly improves aggregate recall, but precision falls because more irrelevant records enter the candidate set.

---

## 6. Aggregate Query Results

The five aggregate queries were:

| Query | Relevant Records |
|---|---:|
| Amlodipine 5mg | 17 |
| Metformin 500mg | 7 |
| Paracetamol 650mg | 20 |
| Hypertension | 16 |
| Type 2 Diabetes + Hypertension | 1 |

### Mean recall across the five aggregate queries

| k | Dense | BM25 | Hybrid |
|---:|---:|---:|---:|
| 2 | 0.326 | 0.314 | 0.326 |
| 5 | 0.514 | 0.502 | 0.514 |
| 10 | 0.743 | 0.731 | 0.743 |
| 20 | 0.978 | **1.000** | 0.990 |
| 30 | 1.000 | 1.000 | 1.000 |
| 50 | 1.000 | 1.000 | 1.000 |

### Takeaway

- **k=2:** too shallow
- **k=10:** much better, but still misses relevant records
- **k=20:** nearly complete aggregate recall
- **k=30:** complete recall for all tested aggregate queries
- **k=50:** no additional recall benefit, while precision becomes poor

Therefore:

> **k≈20 is a strong candidate depth for aggregate retrieval, but it should be validated against precision and latency before changing production.**

---

## 7. BM25 vs Dense

BM25 has one clear advantage:

> It reaches complete aggregate coverage at slightly lower depth for some lexical queries.

Examples:

- Amlodipine: BM25 reaches full coverage earlier than Dense.
- Paracetamol: BM25 reaches full coverage at k=20.
- Hypertension: BM25 reaches full coverage slightly earlier.

However:

- Overall performance is very similar.
- BM25 has slightly lower MRR.
- BM25 does not solve the single-record query problem.
- BM25 uses simple tokenization and untuned default parameters.

### Conclusion

> **BM25 is a useful baseline/comparator, but current evidence does not justify replacing Dense retrieval with BM25.**

---

## 8. Does Hybrid Add Value?

### Short answer: **Not substantially.**

Hybrid uses Dense + BM25 with RRF.

It produces small improvements in a few aggregate cases, but it does not meaningfully improve:

- overall HitRate
- single-record recall
- single-record precision
- overall MRR
- high-k overall recall

### Conclusion

> **Hybrid adds complexity but provides only a narrow retrieval benefit on the current benchmark.**

Therefore, Hybrid should remain a research comparator rather than becoming the default production retriever based on the current evidence.

---

## 9. Masking Impact

Raw vs masked retrieval was available for Dense and BM25.

### Dense at k=10

| Result | Queries |
|---|---:|
| Degraded | 25 |
| Improved | 26 |
| Unchanged | 554 |

### BM25 at k=10

| Result | Queries |
|---|---:|
| Degraded | 5 |
| Improved | 5 |
| Unchanged | 595 |

This shows that masking **does change some individual rankings**, but aggregate retrieval performance remains approximately unchanged.

For BM25, the 5 degraded queries are exactly offset by 5 improved queries, so aggregate metrics remain identical.

### Correct research wording

Do **not** say:

> "Masking has zero effect on retrieval."

Say:

> **"Pre-embedding masking causes limited per-query retrieval changes while approximately preserving aggregate retrieval performance on the evaluated dataset."**

A raw-vs-masked Hybrid comparison cannot currently be verified because no raw Hybrid baseline exists.

---

## 10. Important Reporting Corrections

### BM25

Correct wording:

> **At k=10, 5 queries degraded and 5 improved for HitRate/Precision/Recall, resulting in unchanged aggregate metrics.**

### Hybrid

The reported **"21 masking degradations"** is not supported by the available raw-vs-masked artifacts.

A raw Hybrid baseline is required before making that claim.

### Query counts

Use:

> **605 total queries: 601 single-target + 4 genuinely multi-record aggregate + 1 single-target aggregate.**

### Hybrid report

The Hybrid report contains metric values that disagree with its machine-readable `metrics_hybrid_v2.json`. This must be corrected before publication.

---

## 11. Final Comparison

| Dimension | Dense | BM25 | Hybrid | Conclusion |
|---|---|---|---|---|
| Overall retrieval | Strong | Similar | Similar to Dense | Dense/Hybrid tie |
| Single-record | Poor | Poor | Poor | Query design is main issue |
| Multi-record | Strong with depth | Strong with depth | Strong with depth | No major winner |
| Aggregate recall | Excellent by k=30 | Excellent by k=20 | Excellent by k=30 | BM25 slight high-depth edge |
| Low-k performance | Strong | Slightly weaker | Strong | Dense/Hybrid |
| High-k performance | Similar | Similar | Similar | No meaningful winner |
| Precision | Strong at low k | Competitive | Strong at low k | No decisive winner |
| MRR | Best/tied | Slightly lower | Best/tied | Dense/Hybrid |
| Masking robustness | Small aggregate impact | Very small aggregate impact | Raw comparison unavailable | Generally preserved |
| Complexity | Moderate | Low | Highest | Dense best tradeoff |
| Research value | Strong baseline | Useful comparator | Limited incremental value | Keep all as evaluation |

---

## 12. Final Recommendation

The current evidence supports:

### **Dense Secure-RAG retrieval + adaptive candidate depth**

A possible future strategy is:

```text
Identifiable / single-record query
        → low k (e.g. 2–5)

Aggregate / multi-record query
        → higher candidate depth (around k=20)
```

**Do not change the production default solely from this comparison.** Validate adaptive depth experimentally first.

---

## 13. Main Research Conclusion

The important finding is not that one retrieval algorithm wins.

The stronger conclusion is:

> **Secure-RAG's pre-embedding privacy enforcement can approximately preserve retrieval quality while preventing raw sensitive information from entering the embedding/indexing stage.**

The retrieval experiments suggest that the privacy mechanism is largely **orthogonal to the retrieval strategy**.

The paper should therefore focus on:

1. Privacy protection before embedding
2. Retrieval utility after masking
3. Limited retrieval degradation
4. Domain-aware entity preservation
5. Adaptive retrieval depth for multi-record queries
6. Generalization across domains

Avoid claiming that Secure-RAG improves retrieval quality.

---

## 14. Next Experiments

### Priority 1 — Adaptive Retrieval Depth

Compare fixed k values against an adaptive strategy and measure:

- Recall
- Precision
- MRR
- HitRate
- latency
- retrieved-candidate count

### Priority 2 — Better Single-Record Queries

Create queries containing identifying information and determine how much of the current failure rate is caused by query formulation.

### Priority 3 — Privacy Ablation

Directly compare:

- Raw RAG
- Post-retrieval masking
- Secure-RAG / pre-embedding masking

using the same dataset, queries, retriever, and k.

### Priority 4 — Entity-Type Sensitivity

Measure the retrieval effect of masking:

- PERSON
- PHONE
- EMAIL
- MRN
- Aadhaar
- other sensitive entities

### Priority 5 — Policy Ablation

Compare different masking/preservation policies, especially whether preserving medical entities improves retrieval utility.

### Priority 6 — Cross-Domain Generalization

Test the same Secure-RAG architecture on enterprise, legal, or financial data using domain-specific policies and detectors.

### Priority 7 — Privacy Leakage

Measure whether sensitive information can still be retrieved or exposed from the indexed representation after pre-embedding masking.

---

## One-Sentence Explanation

> **We compared Dense, BM25, and Hybrid retrieval and found that none clearly outperforms Dense; the bigger issues are retrieval depth and query design, while pre-embedding masking causes limited per-query changes and largely preserves overall retrieval quality.**
