# Dataset Validation Report

_Generated 2026-08-06 from `data/generate_dataset.py`_

## Summary

- Total patients: 120
- Unique diseases: 33
- Unique treatments: 107
- Average diagnoses per patient: 1.77
- Average treatments per patient: 3.46
- Gender split: {'male': 58, 'female': 62}

## Disease frequency distribution

| Rank | Disease | Patients |
|---|---|---|
| 1 | Hypertension | 16 |
| 2 | Type 2 Diabetes | 15 |
| 3 | Viral Fever | 12 |
| 4 | Hyperlipidemia | 10 |
| 5 | Asthma | 9 |
| 6 | Migraine | 9 |
| 7 | Sinusitis | 9 |
| 8 | Peptic Ulcer Disease | 8 |
| 9 | Dengue | 8 |
| 10 | COPD | 8 |
| 11 | GERD | 8 |
| 12 | Influenza | 8 |
| 13 | Urinary Tract Infection | 7 |
| 14 | Epilepsy | 7 |
| 15 | Fibromyalgia | 6 |
| 16 | Osteoarthritis | 6 |
| 17 | Dehydration | 6 |
| 18 | Hypothyroidism | 5 |
| 19 | Gout | 5 |
| 20 | Gastritis | 5 |
| 21 | Rheumatoid Arthritis | 5 |
| 22 | COVID-19 | 5 |
| 23 | Pneumonia | 5 |
| 24 | Sciatica | 4 |
| 25 | Iron Deficiency Anaemia | 4 |
| 26 | Diabetic Neuropathy | 3 |
| 27 | Psoriasis | 3 |
| 28 | Tuberculosis | 3 |
| 29 | Insomnia | 3 |
| 30 | Typhoid Fever | 3 |
| 31 | Anxiety Disorder | 3 |
| 32 | Chronic Kidney Disease | 3 |
| 33 | Malaria | 2 |

## Treatment frequency distribution (top 25)

| Rank | Treatment | Patients |
|---|---|---|
| 1 | Hydration | 21 |
| 2 | Paracetamol 650mg | 20 |
| 3 | Amlodipine 5mg | 17 |
| 4 | Lifestyle modification | 16 |
| 5 | IV fluids | 12 |
| 6 | Symptomatic management | 12 |
| 7 | Dietary modification | 12 |
| 8 | Inhaled corticosteroids | 11 |
| 9 | Rest | 10 |
| 10 | Omeprazole 20mg | 9 |
| 11 | ACE inhibitor therapy | 8 |
| 12 | Pantoprazole 40mg | 8 |
| 13 | Statin therapy | 7 |
| 14 | Platelet monitoring | 7 |
| 15 | Metformin 500mg | 7 |
| 16 | Salbutamol inhaler | 7 |
| 17 | Oxygen support | 7 |
| 18 | Ciprofloxacin 500mg | 6 |
| 19 | Antacids | 6 |
| 20 | Sugar control | 6 |
| 21 | Trigger avoidance | 6 |
| 22 | Low-salt diet | 6 |
| 23 | Amoxicillin-clavulanate 625mg | 6 |
| 24 | Dietary counseling | 5 |
| 25 | Analgesics | 5 |

## Diagnoses per patient

- 1 diagnoses: 51 patients
- 2 diagnoses: 45 patients
- 3 diagnoses: 24 patients

## Treatments per patient

- 2 treatments: 15 patients
- 3 treatments: 35 patients
- 4 treatments: 70 patients

## Duplicate PHI check

Verifies that no PHI value (name, phone, Aadhaar, PAN, email, MRN) is shared
between two patients.

PASSED - no duplicate PHI across any patient.

## Deterministic regeneration check

PASSED - regenerating with seed 42 produces a dataset that is byte-for-byte identical under the same seed.

## Medication reuse check (Medication Library)

14 medication(s) prescribed by three or more distinct diseases.

## Pipeline compatibility smoke test

PASSED - `split_into_records` produced 120 records and the masker recognized the following PII patterns on a sample:
- phone: detected
- aadhaar: detected
- pan: detected
- email: detected
- medical_id: detected
