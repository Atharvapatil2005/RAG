# Dataset Architecture — Phase 1 (Domain Model)

_Last updated 2026-08-06_

## Goal

The previous `data/sample_patient_data.txt` behaved like a lookup table: one
disease per patient, one treatment per disease, every disease exactly one
treatment and vice versa. That made retrieval artificially easy.

Phase 1 refactors the dataset into a **small hospital database**: 120 patients
generated from shared medical knowledge libraries instead of isolated,
independent records.

## Architecture

```
Disease Library     33 recurring diseases, long-tail weights, comorbidity hints
Treatment Library   reusable treatment plans (2-4 plans per disease)
Medication Library  medications derived from plans; shared across diseases
Hospital Library    57 weighted Indian hospitals
Doctor Roster       24 reused physician names
        |
Patient Generator  120 patients drawn from the shared knowledge
```

### Disease Library

- 33 diseases (spec: 25-35), each defined by:
  - **weight** — long-tail popularity used to allocate primary-diagnosis slots
  - **age_range** — realistic demographic profile
  - **complaints** — presenting-complaint phrases
  - **plans** — multiple medically plausible treatment plans
  - **follow_ups** — structured follow-up templates
  - **exam_notes** — clinical examination fragments
  - **comorbidities** — commonly co-occurring diseases
- Primary-diagnosis counts are allocated proportional to weights (exactly 120
  slots, then shuffled deterministically). This avoids high-variance multinomial
  sampling and produces a stable long tail (Hypertension ~16, Type 2 Diabetes
  ~15, Viral Fever ~12, ... Malaria 2).

### Treatment Library

- Each disease has 2-4 distinct plans; different patients with the same disease
  receive different valid subsets.
- Patients with multiple diagnoses naturally merge treatments from the shared
  plan pool (capped at 4 per patient).

### Medication Library

- Derived automatically from the treatment plans: medication → diseases map.
- Medications are reused across diseases to introduce realistic retrieval
  ambiguity. Example: Paracetamol 650mg appears in Viral Fever, Influenza,
  COVID-19, Dengue, Malaria, Typhoid Fever, and Osteoarthritis.
- 14 medications are prescribed by 3+ distinct diseases.

### Hospital Library

- 57 real Indian hospitals with weights, so a few tier-1 hospitals (Apollo,
  Fortis, Max, Medanta, Manipal) admit most patients while smaller hospitals
  still appear.

### Patient Generator

Each patient record contains:

| Field | Source |
|---|---|
| Patient ID (MRN) | sequential `MRN1001..MRN1120` |
| Name | Faker `en_IN` (gender-aware, unique) |
| Age / Gender | Faker + disease age profile |
| Hospital | weighted hospital library |
| Admission Date | deterministic spread across 2025 |
| Doctor | reused doctor roster |
| Diagnosis (1-3) | weighted primary + comorbidity-aware secondaries |
| Treatment Plan (1-4) | primary plan + merged comorbidity items |
| Follow-up Plan | structured templates (primary + first comorbidity) |
| Medical Notes | exam fragments from the diagnosis set |
| Contact / Aadhaar / PAN / Email | generated unique PII |

## Determinism

`random.seed(42)` + `Faker.seed(42)`. Doctor roster and all sampling are derived
from the seeded stream; regeneration is byte-for-byte identical (verified).

## Schema Compatibility

The ingestion pipeline treats each blank-line separated block as a patient
record and applies regex + spaCy NER masking before embedding. Phase 1 keeps:

- One record per patient, separated by blank lines.
- Masker-compatible PII formats:
  - `Medical ID: MRN1001` → `[PATIENT_ID_MASKED]`
  - `Contact: 9876543210` → `[PHONE_MASKED]`
  - `Aadhaar: 6123 4567 8901` → `[AADHAAR_MASKED]`
  - `PAN: ABCDE1000A` → `[PAN_MASKED]`
  - `email: name@example.com` → `[EMAIL_MASKED]`

Additive schema change (documented): records now carry explicit `Diagnosis:`,
`Notes:`, `Treated by Dr. X.`, and `Follow-up:` fields on additional lines.
This is text-level metadata that the pipeline masks/chunks identically.

Verified end-to-end:
- `load_data()` → 120 records via `split_into_records()`
- `mask_text()` leaves **zero** raw PII in the masked output
- `build_rag()` indexes 120 chunks without error

## Validation

`reports/dataset/validation_report.md` reports:
- total patients / unique diseases / unique treatments
- disease and treatment frequency distributions
- average diagnoses (1.77) and treatments (3.46) per patient
- duplicate PHI check (passes)
- deterministic regeneration check (passes)
- pipeline compatibility smoke test (passes)

`reports/dataset/distribution_statistics.md` reports the full distributions.
