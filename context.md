# Secure RAG — Engineering Context

## Repository Philosophy

Secure RAG is a privacy-aware retrieval-augmented generation framework designed for research experimentation, not production deployment. The repository is organized around three distinct systems:

1. **Production Runtime** (`secure_rag/`) — The canonical Secure RAG pipeline shipped as a Python package
2. **Research Evaluation Framework** (`benchmarks/`) — An independent evaluation harness for comparing privacy strategies
3. **Deployment Infrastructure** (Docker, Compose, CI, GHCR) — Containerization and automation for distribution

The central research goal is to study the privacy-utility tradeoff introduced by masking sensitive entities before embedding documents. The core design principle is privacy by design: sensitive values should be removed before they enter embeddings and the vector store.

**Architectural Principle:** Runtime frameworks should expose only canonical production behaviour. Experimental baselines belong to a separate evaluation harness and must never leak into the runtime API.

---

## Repository Architecture

### Production Runtime (`secure_rag/`)

The runtime is a Python package that implements the canonical Secure RAG pipeline. It is distributed via TestPyPI and Docker. The runtime has no knowledge of benchmark configurations or research baselines.

**Public API:**
- `build_rag(file_path)` — Build a masked vector index from a document
- `rag_answer(query, vector_store, chunks)` — Generate grounded answers using the index

**Key Modules:**
- `rag_pipeline.py` — Core pipeline orchestration
- `masker.py` — PII detection and masking
- `embedding.py` — SentenceTransformer embeddings
- `vector_store.py` — FAISS indexing
- `retriever.py` — Semantic retrieval
- `generator.py` — LLM-based answer generation
- `pdf_loader.py` — Document loading and chunking
- `cli.py` — Interactive CLI entrypoint

### Research Evaluation Framework (`benchmarks/`)

The benchmark is an independent evaluation harness that compares three privacy strategies using a synthetic medical dataset. It consumes the runtime as a black box but does not define it.

**Evaluation Configurations:**
- **Baseline A** — Raw Retrieval-Augmented Generation (no masking)
- **Baseline B** — Post-Retrieval Privacy Masking (masking after retrieval)
- **Secure RAG** — Pre-Embedding Privacy Enforcement (canonical runtime)

**Key Components:**
- `privacy_eval.py` — Evaluation orchestrator with configuration registry
- `generate_dataset.py` — Synthetic Indian medical dataset generator
- `dataset.jsonl` — 120 synthetic medical records with known PII
- `dataset_queries.json` — 600+ benchmark queries
- `train_test_split.json` — Fixed train/test split for reproducibility
- `results.json` — Structured evaluation output

**Metrics:**
- Document Leakage — PII present in indexed chunks
- Retrieval Leakage (k=5) — PII in top-5 retrieved chunks
- Masking Recall — PII successfully removed by masker
- PHI in Answers — PII in LLM-generated responses

### Deployment Infrastructure

**Docker:**
- `Dockerfile.runtime` — Multi-stage build for production runtime
- CPU-only PyTorch for deterministic ARM64 builds
- spaCy model baked into image
- Non-root runtime user with writable home directory

**Docker Compose:**
- `docker-compose.yml` — Contributor onboarding workflow
- `./data` mounted to `/data` for document access
- Ollama service behind optional profile
- Preferred workflow: `docker compose run --rm secure-rag`

**CI:**
- `python-ci.yml` — Package installation, pytest, build, twine validation
- `docker-ci.yml` — Docker build health, CLI entrypoint verification
- Separate workflows for package and container validation

**GHCR:**
- `publish-ghcr.yml` — Release-driven container publishing
- Triggers only on `release: published`
- Lowercase repository naming for GHCR compliance
- Tags: `latest` and version-specific

---

## Runtime Design Principles

### Single Canonical Pipeline

The runtime exposes exactly one retrieval pipeline. There are no optional branches, no configuration modes, and no research abstractions in the public API.

**Pipeline:**
1. Load document (`.txt` or `.pdf`)
2. Clean transcript artifacts
3. Split into records
4. **Mask each record (mandatory)**
5. Chunk each record independently
6. Generate embeddings
7. Index in FAISS
8. Retrieve using raw query
9. Generate grounded answer
10. Stream response

### Mandatory Pre-Embedding Masking

Masking before embedding is the defining architectural choice. Records are always masked before chunking, embedding, and indexing. This guarantees that raw sensitive values never enter the vector store.

**Tradeoff:** Retrieval for entity-specific queries degrades because the query remains raw while indexed entities are masked. This is treated as a core research result, not an accidental regression.

### No Runtime Modes

The runtime has no concept of "privacy modes." The benchmark framework compares three evaluation configurations, but these are research baselines implemented externally, not runtime features.

### Simplified Public API

The public API is minimal:
- `build_rag(file_path)` — Single parameter, always masks
- `rag_answer(query, vector_store, chunks)` — No mode parameters

This surface area prevents benchmark concepts from leaking into the runtime.

### Query Masking Is Intentionally Disabled

Queries are never masked. The project explicitly evaluates what happens when privacy is enforced on stored content while the user query remains natural.

---

## Benchmark Design Principles

### Evaluation Framework

The benchmark is an independent evaluation harness that answers the research question: How does Secure RAG compare against alternative privacy strategies while preserving retrieval utility?

### Baseline A — Raw Retrieval-Augmented Generation

No masking is applied during indexing, retrieval, or answer generation. This measures the baseline privacy leakage of standard RAG.

**Implementation:** Composes runtime primitives directly: raw text → `chunk_text()` → `embed_chunks()` → `VectorStore` → retrieve → generate_answer

### Baseline B — Post-Retrieval Privacy Masking

Documents are indexed without masking. Retrieved context is masked immediately before answer generation. This isolates the privacy benefit of masking at inference time only.

**Implementation:** Composes runtime primitives: raw text → `chunk_text()` → `embed_chunks()` → `VectorStore` → retrieve → `mask_text()` → generate_answer

### Secure RAG — Pre-Embedding Privacy Enforcement

Sensitive entities are masked before chunking and embedding. The vector store never contains raw sensitive information. No answer-time masking is performed. This represents the canonical Secure RAG pipeline.

**Implementation:** Uses canonical runtime: `build_rag(file_path)` → `rag_answer(query, vector_store, chunks)`

### Runtime Boundary

The benchmark consumes the runtime as a black box. The dependency is one-way: `benchmarks/` imports from `secure_rag/`, but `secure_rag/` never imports from `benchmarks/`.

### Configuration Registry

Evaluation configurations are defined once in `EVALUATION_CONFIGS`. Each configuration owns:
- Stable machine identifier (`id`)
- Human-readable display name (`display_name`)
- Methodology description (`description`)
- Index/chunk source (`get_idx`)
- Answer generation logic (`answer`)

Adding a new baseline requires one entry in the registry. No scattered loops or hardcoded mode strings.

### Research Terminology

The benchmark uses approved research terminology:
- "Baseline A" instead of "raw mode"
- "Baseline B" instead of "post mode"
- "Secure RAG" instead of "pre mode"

Result JSON keys use stable identifiers: `baseline_a`, `baseline_b`, `secure_rag`.

---

## Deployment Philosophy

### Docker

Docker provides a self-contained runtime environment for contributors and users. The image is built with multi-stage optimization, CPU-only PyTorch, and the spaCy model pre-installed.

**Design decisions:**
- Multi-stage build to minimize image size
- CPU-only PyTorch to avoid unnecessary CUDA dependencies on ARM64
- spaCy model baked into image for immediate usability
- Non-root runtime user with writable home directory for cache initialization

### Docker Compose

Compose exists to make contributor onboarding easier without changing application code. It provides an interactive CLI workflow with optional local inference support.

**Preferred workflow:** `docker compose run --rm secure-rag` — attaches directly to terminal, cleans up container after exit.

### GHCR

Container publishing is release-driven, not push-driven. Images are published only when a GitHub Release is created, ensuring that published artifacts are intentional and immutable.

### CI

CI responsibilities are intentionally split:
- Python CI validates package installation, tests, build, and metadata
- Docker CI validates container build health and CLI entrypoint

This separation ensures that package and container failures are isolated and independently debuggable.

---

## Architectural Decisions

### Runtime No Longer Owns Benchmark Concepts

**Decision:** Removed `use_masking` and `mask_mode` parameters from the runtime API.

**Reasoning:** Optional masking and mode selection are benchmark concerns, not runtime concerns. The runtime should expose only the canonical Secure RAG pipeline.

**Impact:** `build_rag(file_path)` now always masks. `rag_answer(query, vector_store, chunks)` never masks at answer time.

### Benchmark Consumes Runtime

**Decision:** Benchmark implements raw and post-retrieval baselines by composing runtime primitives directly, rather than calling the runtime with mode parameters.

**Reasoning:** This preserves the runtime as a black box while allowing the benchmark to construct alternative baselines for comparison.

**Impact:** Baseline A and Baseline B use `chunk_text()`, `embed_chunks()`, `VectorStore`, and `generate_answer()` directly. Secure RAG uses the canonical `build_rag()` and `rag_answer()`.

### Identity Separated From Presentation

**Decision:** Each benchmark configuration has a stable machine identifier (`id`) separate from human-readable display names (`display_name`).

**Reasoning:** Result JSON keys should be stable for programmatic consumption, while console output should be readable for humans. Identity and presentation can evolve independently.

**Impact:** Result JSON uses `baseline_a`, `baseline_b`, `secure_rag`. Console output displays "Baseline A", "Baseline B", "Secure RAG".

### Configuration Registry

**Decision:** Centralized evaluation configuration selection into `EVALUATION_CONFIGS` registry.

**Reasoning:** Scattered hardcoded loops over mode strings made adding new baselines difficult. A registry makes configuration explicit and extensible.

**Impact:** Adding a new baseline requires one entry in `EVALUATION_CONFIGS`. All evaluation loops iterate the registry.

### Research Terminology

**Decision:** Use approved research terminology (Baseline A, Baseline B, Secure RAG) instead of implementation mode names (raw, post, pre).

**Reasoning:** Mode names are implementation details. Research terminology describes the evaluated strategies and is more meaningful to readers.

**Impact:** Documentation, console output, and result descriptions use research terminology. Internal identifiers remain stable.

### Record-Based Chunking

**Decision:** Split input into records first, then mask and chunk each record independently.

**Reasoning:** Whole-document chunking caused cross-record contamination. Patient records are the correct retrieval unit for structured medical-style data.

**Impact:** Each patient record is masked and chunked independently, preventing PII from one record leaking into chunks from another.

### CPU-Only PyTorch in Docker

**Decision:** Install CPU-only PyTorch from dedicated CPU index before installing project dependencies.

**Reasoning:** Default PyTorch ARM64 wheels include CUDA dependencies (~1GB) even for CPU-only inference. This wastes bandwidth and complicates builds.

**Impact:** Docker builds are faster, smaller, and deterministic on ARM64 platforms.

---

## Lessons Learned

### Nested Generators Can Be Unnecessary Indirection

The original `rag_answer()` used a nested `cleaned_response()` generator that buffered the full response before yielding. Flattening to a direct `yield` achieved the same behaviour with less code and clearer intent.

**Lesson:** When refactoring, question whether nested abstractions add value or just indirection. Simpler is often better.

### Removing Optional Parameters in Steps Is Correct

Removing `use_masking` and `mask_mode` in separate steps was the right approach. Each removal affected different parts of the pipeline and had different downstream impacts. Combining them would have made validation harder.

**Lesson:** For complex API changes, stepwise removal allows targeted validation at each stage. Don't batch changes that affect different subsystems.

### Tests Using Defaults Validate the Canonical Path

All tests used default argument values during the runtime refactor. This validated that the default configuration was already the intended Secure RAG behaviour. No test modifications were required.

**Lesson:** When removing optional parameters, check if tests already use the intended defaults. If so, the refactor may be simpler than expected.

### Benchmark Built Its Own Index

The benchmark composed runtime primitives directly rather than calling `build_rag()`. This meant the `use_masking` removal had zero impact on benchmark code initially. The only cross-boundary call was `rag_answer(mask_mode=...)`, which required Phase 3 attention.

**Lesson:** Understand the actual dependency graph before refactoring. Assumptions about coupling can be wrong. The benchmark was more decoupled than expected.

### Docker Daemon Availability Is Not Guaranteed

Local Docker validation was deferred to CI because the daemon was unavailable on the development machine. CI covers Docker builds on push, so this was acceptable. Consider adding `make docker-test` for local validation when the daemon is available.

**Lesson:** CI can compensate for local environment limitations, but local validation is still valuable. Document when and how to validate locally.

### Configuration Centralization Simplifies Extension

Before the configuration registry, adding a new baseline required editing multiple hardcoded loops. After centralization, adding a baseline requires one entry in `EVALUATION_CONFIGS`. This significantly reduces maintenance burden.

**Lesson:** Centralized configuration registries are a powerful pattern for extensibility. They make adding new variants declarative rather than imperative.

### Identity-Presentation Separation Enables Evolution

Separating stable identifiers from display names allows result JSON keys to remain stable while console output can be improved for readability. This is a pattern worth reusing in other contexts.

**Lesson:** Separate internal identity from external presentation. This allows UI/UX improvements without breaking programmatic consumers.

### Prompt Echo Truncation Is a Practical Cleanup Layer

Some chat-style models repeated prompt scaffolding such as `Context:`, `Question:`, or chat control tokens back into the answer. The solution was to post-process generated output and truncate at known scaffold markers.

**Lesson:** When dealing with multiple model providers, accept that output formats will vary. Build practical cleanup layers rather than trying to control provider behaviour.

### Rich Markup Must Be Disabled for Model Output

Some models emit bracketed tokens such as `[/ASSIST]`. Rich interprets bracketed strings as markup by default, causing `MarkupError` during streamed printing. The fix was to print model output with `markup=False`.

**Lesson:** When streaming arbitrary text through formatting libraries, disable markup interpretation. The text is not formatting instructions.

### spaCy Model Installation in Docker Is Worth the Build Cost

The spaCy model `en_core_web_sm` is installed during Docker image build. This means container users don't need a separate runtime step to install the model.

**Lesson:** Pre-install dependencies that are required for basic functionality. The build-time cost is worth the improved user experience.

### Container User Must Have a Real Home Directory

Creating the runtime user with `--no-create-home` caused `PermissionError: '/home/appuser'`. Hugging Face and related libraries initialize caches under `~/.cache`. Without a real home directory, cache initialization fails.

**Lesson:** Respect library expectations about filesystem structure. Don't fight conventions; provide the expected environment.

### Medical ID Regex Expansion Was Necessary

Compact medical ID formats leaked because earlier masking patterns were too narrow. The fix expanded masking to cover patterns like `MRN1002`, `MRN 1002`, `MRN:1002`, `UHID-12345`.

**Lesson:** Structured data often uses compact or punctuation-delimited identifiers. Masking must catch these forms before indexing. Test against real data formats.

### Identity-Based Retrieval Is an Open Research Tradeoff

Names and identifiers become placeholders during indexing, so raw identity terms in queries no longer map cleanly to stored content. Condition- and treatment-oriented retrieval is stronger than identity-oriented retrieval.

**Lesson:** Some limitations are research results, not bugs. Document them as such and focus on understanding the tradeoff rather than "fixing" it.

### CI Separation Isolation Is Valuable

Python CI and Docker CI are intentionally separate. Python CI validates package installation, pytest, build, and metadata. Docker CI validates container build health and CLI entrypoint.

**Lesson:** Separate CI workflows by concern. Package validation and container validation are different failure domains. Isolation makes debugging easier.

### Release-Driven Publishing Prevents Accidental Pushes

Publishing container images only on `release: published` ensures that published artifacts are intentional and immutable. Push-driven publishing would make every push a potential release.

**Lesson:** Make publishing an explicit, intentional action. Don't make releases a side effect of ordinary development activity.

### GHCR Lowercase Naming Is Required

GHCR rejects mixed-case repository names, but GitHub repository metadata preserves original casing. The fix was to lowercase `GITHUB_REPOSITORY` in a shell step and reuse that normalized value for image tags.

**Lesson:** External systems may have different naming constraints than the source platform. Normalize identifiers to match the target system's requirements.

---

## Future Guidelines

### Do Not Add Benchmark Concepts to Runtime

Future contributors must not add benchmark abstractions to the runtime. The runtime should remain a black box with a single canonical pipeline. All research configurations belong in `benchmarks/`.

### Preserve One-Way Dependency

The dependency direction must remain: `benchmarks/` → `secure_rag/`. Never import from `benchmarks/` in runtime code. This preserves the runtime as a stable, independent package.

### Keep Runtime Deterministic

The runtime should have no optional branches or configuration modes. All behaviour should be deterministic and explicit. If a new behaviour is needed, it should be implemented as a separate baseline in the benchmark, not a runtime mode.

### Add New Baselines via Registry

When adding new evaluation configurations to the benchmark, add one entry to `EVALUATION_CONFIGS`. Do not add new hardcoded loops or mode strings. The registry is the single source of truth for configuration.

### Maintain Stable Result Identifiers

When adding new baselines, choose stable machine identifiers for result JSON keys. Do not change existing identifiers. This preserves comparability with historical results.

### Separate Identity from Presentation

When adding new configurations, provide both a stable identifier and a human-readable display name. Use the identifier for programmatic access and the display name for console output.

### Validate Architecture with Evidence

Before making architectural changes, validate with direct evidence. Run tests, benchmarks, and infrastructure checks. Do not rely on assumptions about behaviour.

### Prefer Minimal Changes

Make the smallest change that achieves the goal. Avoid refactoring unrelated code or adding unnecessary features. The project follows a disciplined approach to changes.

### Document Tradeoffs Explicitly

When making design decisions, document the tradeoffs explicitly. Some limitations are research results, not bugs. Distinguish between what should be fixed and what should be understood.

### Test Against Real Data Formats

When implementing masking or parsing logic, test against real data formats. Structured data often uses compact or punctuation-delimited identifiers that naive patterns miss.

### Respect Library Conventions

When configuring container environments or filesystem structure, respect library expectations. Don't fight conventions about cache directories, home directories, or other filesystem expectations.

### Make Publishing Intentional

Publishing artifacts (packages, containers) should be an explicit, intentional action. Don't make releases a side effect of ordinary development activity.

---

## Repository Evolution

The repository evolved from a monolithic codebase with benchmark concepts leaking into the runtime to a clean separation of concerns across three distinct systems.

**Initial State:**
- Runtime exposed optional masking via `use_masking` parameter
- Runtime exposed query modes via `mask_mode` parameter
- Benchmark terminology (raw, post, pre) appeared in runtime code
- Benchmark called runtime with mode parameters
- Documentation described runtime modes as features

**Refactor Outcomes:**

**Runtime:**
- Removed `use_masking` — masking is now mandatory
- Removed `mask_mode` — no answer-time masking in runtime
- Simplified API to `build_rag(file_path)` and `rag_answer(query, vector_store, chunks)`
- No benchmark terminology remains in runtime code
- Runtime is now a black box with single canonical pipeline

**Benchmark:**
- Removed runtime coupling — no longer calls `rag_answer(mask_mode=...)`
- Implements raw and post baselines by composing primitives directly
- Centralized configuration selection in `EVALUATION_CONFIGS` registry
- Separated identity from presentation (stable IDs vs display names)
- Uses approved research terminology (Baseline A, Baseline B, Secure RAG)
- Results use stable identifiers: `baseline_a`, `baseline_b`, `secure_rag`

**Documentation:**
- README updated to describe canonical pipeline
- Removed "Privacy Modes" section
- Updated Python API examples to use canonical signatures
- Research evaluation section describes benchmark configurations
- context.md updated to reflect mandatory masking and no runtime modes

**Architecture:**
- Clear separation: Runtime, Benchmark, Deployment
- One-way dependency: Benchmark → Runtime
- Runtime owns canonical pipeline
- Benchmark owns evaluation configurations
- Deployment owns distribution infrastructure

The repository now accurately reflects the approved architecture: a production runtime with mandatory pre-embedding masking, an independent research evaluation framework, and deployment infrastructure for distribution.

---

## Phase 1 — Medical Dataset Refactor (Domain Model)

### Scope

Phase 1 refactored only the synthetic dataset layer (`data/sample_patient_data.txt`
and its generator). The runtime (`secure_rag/`), retrieval, masking, prompts,
benchmarks, and the evaluation framework were intentionally not modified.

### Problem

The previous dataset was a lookup table: Disease A → Treatment A, exactly one
treatment per disease and one disease per treatment. Retrieval over such data is
artificially easy and does not reflect real clinical practice, where treatments
are reused, patients have comorbidities, and medication overlaps across diseases
create retrieval ambiguity.

### Design Decision: Shared Domain Model

The dataset now simulates a small hospital database. Patients are generated from
shared knowledge libraries rather than independently:

```
Disease Library (33 diseases) -> Treatment Library -> Medication Library
Hospital Library -> Doctor Roster -> Patient Generator
```

- **Disease Library:** 33 recurring diseases with long-tail weights, age
  profiles, presenting complaints, multiple treatment plans, structured
  follow-ups, exam notes, and comorbidity hints.
- **Treatment Library:** 2-4 medically plausible plans per disease; different
  patients with the same disease receive different valid subsets.
- **Medication Library:** derived automatically from plans as a
  medication → diseases map. Reuse across diseases is the intended source of
  retrieval ambiguity (e.g., Paracetamol 650mg spans Viral Fever, Influenza,
  COVID-19, Dengue, Malaria, Typhoid Fever, Osteoarthritis).
- **Hospital Library / Doctor Roster:** weighted real Indian hospitals and
  reused physician names, so records share context like a real hospital DB.

### Design Decision: Exact Long-Tail Primary Distribution

Primary-diagnosis counts are allocated by scaling disease weights to exactly
120 slots and shuffling deterministically, rather than `random.choices`
multinomial sampling. Multinomial draws produced high-variance tails (e.g.,
Dengue expected ~6 primary patients, drew 2). Exact slot allocation gives a
stable long tail (Hypertension 15-16, Type 2 Diabetes ~15, ... Gout 1) that
still looks realistic.

### Design Decision: Comorbidity-Aware Secondaries

Each patient gets 1-3 diagnoses. Secondaries are sampled without replacement,
biased 3:1 toward the primary disease's documented comorbidities, so records
like Diabetes + Hypertension or COPD + Pneumonia appear naturally.

---

## Retrieval Diagnostic Audit — Medication Lookup Failure

### Query Investigated

```
Which patients were prescribed Amlodipine 5mg?
```

### Summary

The answer disappears at the masking stage. The generated dataset contains
`Amlodipine 5mg`, and chunking preserves it intact. However, `mask_text()` uses
spaCy NER after regex masking, and spaCy misclassifies medication terms such as
`Amlodipine` as named entities. The medication is replaced with placeholders
before embedding and indexing, so vector search never sees the literal
medication requested by the query.

This was an investigation only. No runtime code, prompts, retrieval settings,
embeddings, or chunking were modified.

### Pipeline Observed

Current runtime path:

```
load_data()
  -> split_into_records()
  -> mask_text()
  -> chunk_record()
  -> embed_chunks()
  -> VectorStore
  -> retrieve()
  -> generate_answer()
```

The relevant code path is in `secure_rag/rag_pipeline.py`: `build_rag()` masks
each record before chunking and embedding.

### Stage Results

| Stage | Result | Finding |
| --- | --- | --- |
| Dataset | PASS | `data/sample_patient_data.txt` contains 17 `Amlodipine` occurrences in 17 records. |
| Chunking | PASS | Each matching patient remains one chunk; `Amlodipine 5mg` is intact before masking. |
| Masking | FAIL | Every relevant chunk loses both `Amlodipine` and `5mg`. |
| Embedding | FAIL downstream | Embeddings are built from masked chunks that no longer contain the queried medication. |
| Vector Search / Ranking | FAIL downstream | Top-10 chunks contain no literal `Amlodipine` and no `5mg`. |
| Retrieved Context | FAIL | Runtime Top-2 context does not contain the answer. |
| Prompt Construction | PASS / not bottleneck | The final prompt faithfully includes retrieved context, but the context lacks the medication. |
| LLM Generation | PASS / not implicated | Returning `I don't know` is consistent with the context supplied to the model. |

### Root Cause

Selected root cause:

```
Masking
```

Specifically, spaCy NER inside `mask_text()` masks medication names and nearby
numeric dosage text. Examples observed:

```
Treatment: Amlodipine 5mg, Lifestyle modification.
```

became:

```
Treatment: [NAME_MASKED]mg, [NAME_MASKED] modification.
```

and in some records:

```
Treatment: Low-salt diet, Renal function monitoring, Amlodipine 5mg.
```

became:

```
Treatment: Low-salt diet, [ORG_MASKED] function monitoring, [NAME_MASKED]Notes:
```

The medication is therefore lost before embeddings and FAISS search.

### Dataset Evidence

The dataset contains 17 expected relevant records:

```
MRN1001  Nakul Rattan        Hypertension; Asthma; Migraine                  Amlodipine 5mg
MRN1005  Ekiya Koshy         Hyperlipidemia; Hypertension                    Amlodipine 5mg
MRN1021  Jagdish Bhardwaj    Hypertension                                    Amlodipine 5mg
MRN1025  Libni Wable         Hypertension; Anxiety Disorder                  Amlodipine 5mg
MRN1026  Jagvi Zacharia      Hypertension; Fibromyalgia                      Amlodipine 5mg
MRN1052  Robert Lanka        Hypertension; GERD                              Amlodipine 5mg
MRN1054  Faras Gupta         Chronic Kidney Disease                          Amlodipine 5mg
MRN1059  Girik Biswas        Hypertension; Hyperlipidemia                    Amlodipine 5mg
MRN1066  Nisha Bora          Hypertension                                    Amlodipine 5mg
MRN1070  Krisha Pingle       Hypertension; Viral Fever; Psoriasis            Amlodipine 5mg
MRN1074  Dalbir Chhabra      Hypertension; Hyperlipidemia                    Amlodipine 5mg
MRN1085  Ansh Morar          Hypertension; COPD; Sinusitis                   Amlodipine 5mg
MRN1092  Gauri Chokshi       Chronic Kidney Disease                          Amlodipine 5mg
MRN1107  Oni Murthy          Hypertension                                    Amlodipine 5mg
MRN1111  Shivansh Raghavan   Hypertension                                    Amlodipine 5mg
MRN1118  Chakradhar Keer     Hypertension; Urinary Tract Infection           Amlodipine 5mg
MRN1119  Jeevika Issac       Hypertension                                    Amlodipine 5mg
```

### Retrieval Evidence

FAISS uses L2 distance, so lower is better. Top-10 retrieval for the exact query:

```
1.  distance=1.175774 | MRN1060 | Epilepsy; Influenza | Amlodipine=NO | 5mg=NO
2.  distance=1.186596 | MRN1111 | Hypertension | Amlodipine=NO | 5mg=NO
3.  distance=1.209115 | MRN1022 | GERD; Epilepsy; Sciatica | Amlodipine=NO | 5mg=NO
4.  distance=1.214506 | MRN1066 | Hypertension | Amlodipine=NO | 5mg=NO
5.  distance=1.231099 | MRN1094 | Chronic Kidney Disease; Rheumatoid Arthritis | Amlodipine=NO | 5mg=NO
6.  distance=1.233959 | MRN1059 | Hypertension; Hyperlipidemia | Amlodipine=NO | 5mg=NO
7.  distance=1.250587 | MRN1053 | Migraine | Amlodipine=NO | 5mg=NO
8.  distance=1.252337 | MRN1119 | Hypertension | Amlodipine=NO | 5mg=NO
9.  distance=1.259530 | MRN1025 | Hypertension; Anxiety Disorder | Amlodipine=NO | 5mg=NO
10. distance=1.259937 | MRN1039 | Migraine; Anxiety Disorder | Amlodipine=NO | 5mg=NO
```

Some expected records are technically ranked in the Top-10, but their retrieved
chunks no longer contain the medication because masking removed it. They cannot
answer the query.

### Ground Truth Ranking

```
MRN1001  NOT FOUND  rank=21  distance=1.323069
MRN1005  NOT FOUND  rank=59  distance=1.424293
MRN1021  NOT FOUND  rank=22  distance=1.324677
MRN1025  FOUND      rank=9   distance=1.259530
MRN1026  NOT FOUND  rank=11  distance=1.274723
MRN1052  NOT FOUND  rank=19  distance=1.314864
MRN1054  NOT FOUND  rank=33  distance=1.359998
MRN1059  FOUND      rank=6   distance=1.233959
MRN1066  FOUND      rank=4   distance=1.214506
MRN1070  NOT FOUND  rank=13  distance=1.279102
MRN1074  NOT FOUND  rank=51  distance=1.400741
MRN1085  NOT FOUND  rank=25  distance=1.340046
MRN1092  NOT FOUND  rank=50  distance=1.399263
MRN1107  NOT FOUND  rank=12  distance=1.278247
MRN1111  FOUND      rank=2   distance=1.186596
MRN1118  NOT FOUND  rank=42  distance=1.385151
MRN1119  FOUND      rank=8   distance=1.252337
```

Again, `FOUND` here only means the patient record was ranked in Top-10. It does
not mean the retrieved masked chunk still contains `Amlodipine 5mg`.

### Runtime Top-2 Context

`secure_rag/retriever.py` defaults to `k=2`. For the investigated query, runtime
Top-2 context contained:

```
MRN1060 masked chunk: no Amlodipine, no 5mg
MRN1111 masked chunk: no Amlodipine, no 5mg
```

The final prompt therefore lacked the answer. The model returning `I don't know`
is expected behavior under the prompt's grounding rules.

### Key Learning

Medication-specific retrieval fails because the privacy masking layer currently
removes non-PHI clinical terms before indexing. This is distinct from a prompt
issue or an LLM generation issue. Before changing retrieval architecture, the
masking behavior must be understood as the precise bottleneck for medication
lookup failures.

### Design Decision: Schema-Compatible Additive Extension

The record format keeps the pipeline's ingestion contract (blank-line separated
records; masker-compatible PII: `MRN1001`, 10-digit phone, `1234 5678 8901`
Aadhaar, `ABCDE1000A` PAN, email). Diagnosis, Notes, Doctor, and Follow-up were
added as text lines. Verified: `load_data` → 120 records, `mask_text` leaves
zero raw PII, `build_rag` indexes 120 chunks.

### Lessons Learned

- **Faker 40.x**: `fake.seed()` on instances is deprecated; use the class
  method `Faker.seed()`. Instance-level seeding fails silently at runtime.
- **Constructor keyword order matters**: the `Disease` dataclass field order
  silently swapped `age_range` and `complaints`, producing a `randint()`
  TypeError only at runtime. Prefer keyword arguments or a deliberate field
  order when constructors have many positional args.
- **Sample without replacement for secondaries**: `random.choices(..., k=2)`
  samples with replacement and produced duplicate diagnoses ("Asthma; Asthma").
  Use explicit weighted draw-without-replacement.
- **Exact counts beat pure sampling for "database" realism**: slot-based
  primary allocation keeps target long-tail frequencies stable while shuffling
  preserves the appearance of natural assignment order.
- **Validate against the real ingestion path**: running `split_into_records`,
  `mask_text`, and `build_rag` on the regenerated file caught schema drift that
  a self-contained generator test would not (e.g., PII formats that no longer
  matched the masker regexes).

### Scope Boundary for Later Phases

Phase 1 intentionally stops at dataset generation. Query generation, ambiguity
handling, follow-up question generation, and prompt modifications are explicitly
out of scope and should be addressed in later phases.

### EVALUATION_FRAMEWORK_CHECKLIST.md Impact

Not affected. The evaluation framework consumes `benchmarks/dataset.jsonl`
(untouched this phase), not `data/sample_patient_data.txt`.
