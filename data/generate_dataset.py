#!/usr/bin/env python3
"""
Synthetic medical dataset generator with a shared domain model.

Phase 1 - Medical Dataset Refactor.

Generates `sample_patient_data.txt`: a small hospital database rendered as
per-patient text records that stay compatible with the Secure RAG ingestion
pipeline (blank-line separated records; PII kept in masker-compatible formats).

Architecture (shared knowledge libraries -> patient generator):

    Disease Library     (25-35 recurring diseases with long-tail weights)
    Treatment Library   (reusable treatment plans, many per disease)
    Medication Library  (medications reused across diseases)
    Hospital Library    (weighted Indian hospital roster)
    Doctor Roster       (reused physician names)
            |
    Patient Generator   (120 patients drawn from the shared knowledge)

Determinism: `random.seed(SEED)` and `Faker.seed(SEED)` reproduce the exact
dataset on every run.

CLI:
    python3 data/generate_dataset.py                 # regenerate dataset + reports
    python3 data/generate_dataset.py --seed 42 --num-patients 120
    python3 data/generate_dataset.py --output data/sample_patient_data.txt
    python3 data/generate_dataset.py --no-reports    # dataset only
    python3 data/generate_dataset.py --compat-check  # also run pipeline smoke test
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from faker import Faker

SEED = 42
NUM_PATIENTS = 120
BASE_DIR = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "sample_patient_data.txt"
REPORT_PATH = BASE_DIR / "validation_report.md"
DISTRIBUTION_PATH = BASE_DIR / "distribution_statistics.md"

fake = Faker("en_IN")


# --------------------------------------------------------------------------- #
# Shared knowledge libraries
# --------------------------------------------------------------------------- #


@dataclass
class Disease:
    name: str
    weight: int
    age_range: Tuple[int, int]
    complaints: List[str]
    plans: List[List[str]]
    follow_ups: List[str]
    exam_notes: List[str]
    comorbidities: List[str] = field(default_factory=list)


DISEASES: List[Disease] = [
    Disease(
        "Hypertension", 18, (35, 75),
        ["persistent hypertension", "high blood pressure", "elevated blood pressure readings"],
        [
            ["Amlodipine 5mg", "Lifestyle modification"],
            ["Telmisartan 40mg", "Low-salt diet", "Amlodipine 5mg"],
            ["Losartan 50mg", "Lifestyle modification"],
            ["ACE inhibitor therapy", "Lifestyle modification", "Amlodipine 5mg"],
        ],
        [
            "Review after 2 weeks; monitor blood pressure twice daily",
            "Review after 1 month; repeat renal function tests",
            "Monitor blood pressure; review antihypertensive dose in 3 weeks",
        ],
        ["blood pressure was 152/94 mmHg", "blood pressure was 146/90 mmHg", "pulse rate was 82 bpm"],
        ["Type 2 Diabetes", "Hyperlipidemia", "Chronic Kidney Disease", "GERD"],
    ),
    Disease(
        "Type 2 Diabetes", 15, (30, 70),
        ["uncontrolled diabetes", "elevated blood glucose", "excessive thirst and frequent urination"],
        [
            ["Metformin 500mg", "Dietary modification", "Exercise"],
            ["Metformin 500mg", "Glimepiride 1mg", "Dietary modification"],
            ["Insulin therapy", "Sugar control", "Dietary modification"],
            ["Dapagliflozin 10mg", "Dietary modification", "Glucose monitoring"],
        ],
        [
            "Review after 1 month; repeat HbA1c",
            "Repeat fasting blood glucose in 2 weeks",
            "Monitor blood glucose twice daily; review after 3 weeks",
        ],
        ["random blood glucose was 212 mg/dL", "HbA1c level was 8.1%", "fasting blood glucose was 158 mg/dL"],
        ["Hypertension", "Diabetic Neuropathy", "Hyperlipidemia", "Chronic Kidney Disease"],
    ),
    Disease(
        "Viral Fever", 11, (18, 60),
        ["persistent fever", "high-grade fever", "fever with body ache"],
        [
            ["Paracetamol 650mg", "Hydration", "Rest"],
            ["Paracetamol 650mg", "Symptomatic management", "Rest"],
            ["Paracetamol 650mg", "Hydration", "Antipyretic therapy"],
        ],
        [
            "Review after 1 week if fever persists",
            "Repeat CBC after 3 days",
            "Return if fever persists beyond 5 days",
        ],
        ["temperature was 101.4°F", "temperature was 100.8°F", "pulse rate was 96 bpm"],
        ["Dehydration"],
    ),
    Disease(
        "Influenza", 9, (18, 65),
        ["flu-like symptoms", "fever with severe body ache", "cough and high fever"],
        [
            ["Paracetamol 650mg", "Hydration", "Rest"],
            ["Oseltamivir 75mg", "Symptomatic management", "Hydration"],
            ["Paracetamol 650mg", "Cough suppressant", "Rest"],
        ],
        [
            "Review after 1 week; complete oseltamivir course",
            "Return if breathlessness develops",
            "Review after 5 days",
        ],
        ["temperature was 102.0°F", "pulse rate was 98 bpm", "oxygen saturation was 96%"],
        ["Pneumonia", "Dehydration"],
    ),
    Disease(
        "Dengue", 7, (18, 55),
        ["dengue fever", "fever with severe headache", "fever with retro-orbital pain"],
        [
            ["Paracetamol 650mg", "Platelet monitoring", "IV fluids"],
            ["Paracetamol 650mg", "Hydration", "Platelet monitoring"],
            ["IV fluids", "Platelet monitoring", "Symptomatic management"],
        ],
        [
            "Daily platelet count until recovery",
            "Review after 1 week; repeat CBC",
            "Monitor for warning signs; review in 48 hours",
        ],
        ["platelet count was 78000 per microliter", "temperature was 101.6°F", "haematocrit was 44%"],
        ["Dehydration"],
    ),
    Disease(
        "Asthma", 8, (12, 55),
        ["seasonal asthma", "wheezing and breathlessness", "recurrent asthma attacks"],
        [
            ["Salbutamol inhaler", "Inhaled corticosteroids"],
            ["Salbutamol inhaler", "Montelukast 10mg", "Inhaled corticosteroids"],
            ["Inhaled corticosteroids", "Ventolin inhaler", "Trigger avoidance"],
        ],
        [
            "Review after 2 weeks; reassess inhaler technique",
            "Repeat spirometry in 1 month",
            "Monitor peak flow twice daily",
        ],
        ["oxygen saturation was 95%", "wheezing heard on auscultation", "peak expiratory flow was 320 L/min"],
        ["GERD", "Sinusitis"],
    ),
    Disease(
        "Migraine", 6, (18, 50),
        ["severe migraine", "throbbing headache", "recurrent migraine attacks"],
        [
            ["Sumatriptan 50mg", "Rest in a dark room"],
            ["Propranolol 40mg", "Trigger avoidance", "Pain management"],
            ["Naproxen 500mg", "Rest", "Preventive therapy"],
        ],
        [
            "Review after 2 weeks; maintain headache diary",
            "Neurology review after 1 month",
            "Review if headache frequency increases",
        ],
        ["neurological examination was unremarkable", "pulse rate was 88 bpm"],
        ["Anxiety Disorder"],
    ),
    Disease(
        "GERD", 5, (25, 65),
        ["severe acidity", "heartburn and regurgitation", "acid reflux"],
        [
            ["Omeprazole 20mg", "Diet modification"],
            ["Pantoprazole 40mg", "Lifestyle modification", "Antacids"],
            ["Antacids", "Diet modification"],
        ],
        [
            "Review after 2 weeks; reassess symptoms",
            "Upper GI endoscopy if symptoms persist",
            "Review after 1 month",
        ],
        ["epigastric tenderness on palpation", "vitals were stable"],
        ["Gastritis", "Peptic Ulcer Disease", "Asthma"],
    ),
    Disease(
        "Osteoarthritis", 5, (45, 80),
        ["knee joint pain", "stiffness in joints", "chronic joint pain"],
        [
            ["NSAIDs", "Physiotherapy"],
            ["Paracetamol 650mg", "Joint strengthening exercises"],
            ["Topical diclofenac gel", "Physiotherapy", "Weight management"],
        ],
        [
            "Review after 2 weeks; reassess pain score",
            "Orthopedic review after 1 month",
            "Physiotherapy twice a week",
        ],
        ["knee tenderness and crepitus on movement", "reduced range of motion in right knee"],
        ["Hypertension", "Type 2 Diabetes"],
    ),
    Disease(
        "COVID-19", 5, (18, 70),
        ["COVID-19 symptoms", "fever with loss of smell", "persistent cough and breathlessness"],
        [
            ["Paracetamol 650mg", "Hydration", "Rest"],
            ["Remdesivir", "IV fluids", "Oxygen support"],
            ["Symptomatic management", "Paracetamol 650mg", "Oxygen support"],
        ],
        [
            "Review after 1 week; repeat CT chest if worsening",
            "Monitor oxygen saturation twice daily",
            "Follow-up after 10 days",
        ],
        ["oxygen saturation was 94%", "temperature was 100.2°F", "chest X-ray showed ground-glass opacities"],
        ["Pneumonia", "Type 2 Diabetes"],
    ),
    Disease(
        "Pneumonia", 4, (30, 75),
        ["pneumonia", "fever with productive cough", "breathlessness with chest pain"],
        [
            ["Azithromycin 500mg", "Oxygen support", "Hydration"],
            ["IV antibiotics", "Oxygen support", "Paracetamol 650mg"],
            ["Amoxicillin-clavulanate 625mg", "Cough suppressant", "Hydration"],
        ],
        [
            "Review after 1 week; repeat chest X-ray",
            "Review after 2 weeks; reassess oxygen needs",
            "Repeat CBC after 5 days",
        ],
        ["crepitations heard in right lower zone", "oxygen saturation was 93%", "temperature was 102.4°F"],
        ["COPD", "Influenza", "COVID-19"],
    ),
    Disease(
        "COPD", 4, (45, 80),
        ["COPD exacerbation", "chronic cough with breathlessness", "worsening breathlessness"],
        [
            ["Salbutamol nebulization", "Ipratropium inhaler", "Prednisolone 30mg"],
            ["Inhaled corticosteroids", "Salbutamol inhaler", "Oxygen support"],
            ["Long-acting bronchodilators", "Pulmonary rehabilitation"],
        ],
        [
            "Review after 2 weeks; repeat spirometry",
            "Review after 1 month; assess inhaler technique",
            "Smoking cessation counseling; review in 3 weeks",
        ],
        ["prolonged expiration with wheeze", "oxygen saturation was 91%", "barrel-shaped chest"],
        ["Pneumonia", "Hypertension"],
    ),
    Disease(
        "Hypothyroidism", 4, (20, 65),
        ["hypothyroidism", "fatigue with weight gain", "cold intolerance"],
        [
            ["Levothyroxine 25mcg", "Thyroid function monitoring"],
            ["Levothyroxine 50mcg", "Dietary counseling"],
        ],
        [
            "Repeat TSH after 6 weeks",
            "Review after 1 month; reassess symptoms",
            "Monitor thyroid function every 3 months",
        ],
        ["dry skin and pitting edema", "pulse rate was 64 bpm"],
        ["Hyperlipidemia"],
    ),
    Disease(
        "Gastritis", 4, (20, 60),
        ["gastritis", "upper abdominal pain", "burning epigastric pain"],
        [
            ["Omeprazole 20mg", "Bland diet"],
            ["Pantoprazole 40mg", "Antacids", "Diet modification"],
        ],
        [
            "Review after 2 weeks; reassess symptoms",
            "Upper GI endoscopy if not improving",
            "Review after 1 month",
        ],
        ["epigastric tenderness", "vitals were stable"],
        ["GERD", "Peptic Ulcer Disease"],
    ),
    Disease(
        "Chronic Kidney Disease", 3, (40, 75),
        ["chronic kidney disease", "reduced urine output with fatigue", "renal dysfunction"],
        [
            ["Low-salt diet", "Renal function monitoring", "Amlodipine 5mg"],
            ["ACE inhibitor therapy", "Low-protein diet", "Renal function monitoring"],
        ],
        [
            "Repeat serum creatinine after 2 weeks",
            "Nephrology review after 1 month",
            "Monitor blood pressure and renal panel",
        ],
        ["serum creatinine was 2.1 mg/dL", "blood pressure was 148/92 mmHg", "mild pedal edema"],
        ["Hypertension", "Type 2 Diabetes"],
    ),
    Disease(
        "Hyperlipidemia", 3, (30, 70),
        ["high cholesterol", "elevated lipid levels", "high cholesterol on routine check"],
        [
            ["Atorvastatin 10mg", "Diet modification"],
            ["Rosuvastatin 10mg", "Lifestyle modification", "Diet modification"],
            ["Statin therapy", "Low-fat diet"],
        ],
        [
            "Repeat lipid profile after 3 months",
            "Review after 1 month; reassess diet",
            "Monitor liver function with statins",
        ],
        ["serum cholesterol was 246 mg/dL", "LDL cholesterol was 162 mg/dL"],
        ["Hypertension", "Type 2 Diabetes", "Hypothyroidism"],
    ),
    Disease(
        "Urinary Tract Infection", 3, (20, 65),
        ["urinary tract infection", "burning sensation while urinating", "frequent painful urination"],
        [
            ["Nitrofurantoin 100mg", "Hydration"],
            ["Ciprofloxacin 500mg", "Hydration", "Analgesics"],
        ],
        [
            "Review after 1 week; repeat urine culture",
            "Review if symptoms persist after 3 days",
            "Increase fluid intake; review in 1 week",
        ],
        ["urine showed pus cells and nitrites", "suprapubic tenderness"],
        ["Dehydration"],
    ),
    Disease(
        "Iron Deficiency Anaemia", 3, (18, 60),
        ["iron deficiency anemia", "fatigue with pale skin", "weakness and dizziness"],
        [
            ["Ferrous sulfate 200mg", "Dietary counseling"],
            ["Ferrous sulfate 200mg", "Folic acid 5mg", "Vitamin B12 supplement", "Dietary counseling"],
        ],
        [
            "Repeat CBC after 4 weeks",
            "Review after 1 month; reassess hemoglobin",
            "Iron studies after 6 weeks",
        ],
        ["hemoglobin was 9.2 g/dL", "pallor on conjunctival examination", "ferritin was 12 ng/mL"],
        ["Type 2 Diabetes"],
    ),
    Disease(
        "Tuberculosis", 2, (20, 65),
        ["tuberculosis", "persistent cough with weight loss", "fever with night sweats"],
        [
            ["Anti-TB therapy (Category 1)", "Vitamin B6 supplement", "Sputum monitoring"],
            ["Anti-TB therapy", "Directly observed therapy", "Nutritional support"],
        ],
        [
            "Monthly sputum smear monitoring",
            "Review after 2 months; reassess response",
            "Complete 6-month ATT course; review quarterly",
        ],
        ["chest X-ray showed upper lobe infiltrates", "temperature was 99.8°F", "wasted appearance"],
        [],
    ),
    Disease(
        "Epilepsy", 2, (18, 50),
        ["seizure disorder", "recurrent seizures", "epilepsy"],
        [
            ["Levetiracetam 500mg", "Neurology follow-up"],
            ["Sodium valproate 500mg", "Neurology follow-up"],
            ["Carbamazepine 200mg", "Seizure diary", "Neurology follow-up"],
        ],
        [
            "Review after 1 month; adjust dose if needed",
            "Repeat EEG after 3 months",
            "Maintain seizure diary; review in 1 month",
        ],
        ["neurological examination was unremarkable", "EEG showed epileptiform discharges"],
        ["Migraine"],
    ),
    Disease(
        "Peptic Ulcer Disease", 2, (25, 65),
        ["peptic ulcer", "burning stomach pain", "recurrent epigastric pain"],
        [
            ["Omeprazole 20mg", "Dietary restriction", "Helicobacter pylori eradication"],
            ["Pantoprazole 40mg", "Antacids", "Dietary restriction"],
        ],
        [
            "Review after 2 weeks; repeat endoscopy if indicated",
            "Review after 1 month; reassess symptoms",
        ],
        ["epigastric tenderness", "fecal occult blood test was positive"],
        ["GERD", "Gastritis"],
    ),
    Disease(
        "Typhoid Fever", 2, (15, 45),
        ["typhoid fever", "prolonged fever with abdominal pain", "step-ladder fever pattern"],
        [
            ["Azithromycin 500mg", "Hydration", "Rest"],
            ["Ceftriaxone 1g IV", "IV fluids", "Symptomatic management"],
        ],
        [
            "Review after 1 week; repeat Widal test",
            "Review if fever persists after 5 days",
            "Complete antibiotic course; review in 1 week",
        ],
        ["temperature was 103.0°F", "coated tongue and relative bradycardia"],
        ["Dehydration"],
    ),
    Disease(
        "Malaria", 1, (18, 55),
        ["malaria", "fever with chills and rigors", "intermittent high fever"],
        [
            ["Artemisinin combination therapy", "Paracetamol 650mg", "Hydration"],
            ["Chloroquine", "Paracetamol 650mg", "Rest"],
        ],
        [
            "Repeat peripheral smear after 3 days",
            "Review after 1 week; reassess fever pattern",
        ],
        ["temperature was 104.0°F", "peripheral smear showed malaria parasites"],
        ["Dehydration"],
    ),
    Disease(
        "Sinusitis", 1, (18, 60),
        ["sinusitis", "facial pain and nasal congestion", "recurrent sinus headache"],
        [
            ["Amoxicillin-clavulanate 625mg", "Steam inhalation", "Nasal decongestants"],
            ["Steam inhalation", "Saline nasal rinse", "Analgesics"],
        ],
        [
            "ENT review after 2 weeks",
            "Review if symptoms persist after 10 days",
        ],
        ["maxillary sinus tenderness on palpation", "nasal mucosa was congested"],
        ["Asthma"],
    ),
    Disease(
        "Anxiety Disorder", 1, (20, 55),
        ["anxiety disorder", "panic attacks", "palpitations with anxiety"],
        [
            ["Cognitive behavioral therapy", "Breathing exercises"],
            ["Sertraline 50mg", "Counseling", "Sleep regulation"],
        ],
        [
            "Weekly therapy sessions; review after 1 month",
            "Review after 2 weeks; reassess anxiety score",
        ],
        ["pulse rate was 104 bpm", "blood pressure was 138/86 mmHg"],
        ["Migraine"],
    ),
    Disease(
        "Dehydration", 1, (18, 60),
        ["dehydration", "dizziness and weakness", "reduced urine output"],
        [
            ["IV fluids", "Electrolyte replacement"],
            ["Oral rehydration salts", "Hydration"],
        ],
        [
            "Review after 3 days; repeat electrolytes",
            "Increase fluid intake; review after 1 week",
        ],
        ["dry mucous membranes", "pulse rate was 110 bpm"],
        [],
    ),
    Disease(
        "Fibromyalgia", 1, (25, 60),
        ["widespread muscle pain", "fibromyalgia pain", "chronic pain and fatigue"],
        [
            ["Pregabalin 75mg", "Graded exercise"],
            ["Duloxetine 30mg", "Graded exercise", "Sleep hygiene"],
        ],
        [
            "Pain clinic review after 1 month",
            "Review after 2 weeks; reassess pain score",
        ],
        ["tender points over bilateral trapezius", "normal inflammatory markers"],
        ["Anxiety Disorder"],
    ),
    Disease(
        "Psoriasis", 1, (20, 60),
        ["psoriasis flare", "scaly skin patches", "itchy silvery plaques"],
        [
            ["Topical steroids", "Emollients"],
            ["Calcipotriol ointment", "Dermatology review"],
        ],
        [
            "Dermatology review after 1 month",
            "Review after 2 weeks; reassess plaque extent",
        ],
        ["well-demarcated scaly plaques over elbows", "Auschitz sign present"],
        [],
    ),
    Disease(
        "Rheumatoid Arthritis", 1, (30, 70),
        ["joint swelling and stiffness", "rheumatoid arthritis", "morning stiffness in joints"],
        [
            ["NSAIDs", "Rheumatology review"],
            ["Methotrexate 10mg", "Folic acid supplement", "Rheumatology review"],
        ],
        [
            "Rheumatology review after 1 month",
            "Repeat inflammatory markers after 3 weeks",
        ],
        ["swollen metacarpophalangeal joints", "ESR was 62 mm/hr"],
        ["Osteoarthritis"],
    ),
    Disease(
        "Diabetic Neuropathy", 1, (35, 70),
        ["numbness and tingling in feet", "diabetic neuropathy", "burning pain in lower limbs"],
        [
            ["Gabapentin 300mg", "Sugar control"],
            ["Pregabalin 75mg", "Pain management", "Sugar control"],
        ],
        [
            "Neurology review after 1 month",
            "Review after 2 weeks; reassess neuropathic pain",
        ],
        ["reduced vibration sense in both feet", "monofilament test was abnormal"],
        ["Type 2 Diabetes"],
    ),
    Disease(
        "Gout", 1, (30, 70),
        ["gout flare", "severe joint pain in big toe", "acute podagra"],
        [
            ["Colchicine 0.5mg", "Uric acid control"],
            ["Allopurinol 100mg", "NSAIDs", "Diet modification"],
        ],
        [
            "Repeat serum uric acid after 2 weeks",
            "Review after 1 month; reassess gout flares",
        ],
        ["swollen and erythematous first metatarsophalangeal joint", "serum uric acid was 9.4 mg/dL"],
        ["Hyperlipidemia"],
    ),
    Disease(
        "Sciatica", 1, (35, 70),
        ["lower back pain radiating to leg", "sciatica", "shooting leg pain"],
        [
            ["Analgesics", "Physiotherapy"],
            ["Gabapentin 300mg", "Spine physiotherapy", "Pain management"],
        ],
        [
            "Physiotherapy twice a week; review after 2 weeks",
            "Orthopedic review after 1 month",
        ],
        ["positive straight leg raise on the left", "paraspinal muscle spasm"],
        ["Osteoarthritis"],
    ),
    Disease(
        "Insomnia", 1, (20, 65),
        ["insomnia", "difficulty sleeping", "frequent night awakenings"],
        [
            ["Sleep hygiene counseling", "Melatonin 3mg"],
            ["Cognitive behavioral therapy for insomnia", "Sleep regulation"],
        ],
        [
            "Review after 2 weeks; review sleep diary",
            "Review after 1 month; reassess sleep quality",
        ],
        ["anxious appearance", "vitals were stable"],
        ["Anxiety Disorder"],
    ),
]

# Hospital library: real Indian hospitals with weights (some see more patients).
HOSPITALS: List[Tuple[str, int]] = [
    ("Apollo Hospital", 7),
    ("Fortis Hospital", 6),
    ("Max Hospital", 5),
    ("Medanta Medicity", 5),
    ("Manipal Hospital", 5),
    ("KIMS Hospital", 4),
    ("Narayana Health", 4),
    ("AIIMS Delhi", 4),
    ("Aster Hospital", 3),
    ("Kokilaben Hospital", 3),
    ("Lilavati Hospital", 3),
    ("Nanavati Hospital", 3),
    ("Hinduja Hospital", 3),
    ("Yashoda Hospital", 3),
    ("Care Hospital", 3),
    ("Artemis Hospital", 3),
    ("Sir Ganga Ram Hospital", 3),
    ("CMC Vellore", 3),
    ("Hiranandani Hospital", 3),
    ("Ruby Hall Clinic", 2),
    ("Paras Hospital", 2),
    ("Jehangir Hospital", 2),
    ("Global Hospital", 2),
    ("Wockhardt Hospital", 2),
    ("Columbia Asia Hospital", 2),
    ("Shalby Hospital", 2),
    ("Sahyadri Hospital", 2),
    ("BLK-Max Hospital", 2),
    ("Apollo Gleneagles", 2),
    ("Rainbow Hospital", 2),
    ("HCG Hospital", 2),
    ("Jaslok Hospital", 2),
    ("Sakra Hospital", 2),
    ("Sterling Hospital", 2),
    ("Moolchand Hospital", 1),
    ("Breach Candy Hospital", 1),
    ("Kauvery Hospital", 1),
    ("SIMS Hospital", 1),
    ("VPS Lakeshore Hospital", 1),
    ("Bombay Hospital", 1),
    ("Motherhood Hospital", 1),
    ("SevenHills Hospital", 1),
    ("Metro Hospital", 1),
    ("Noble Hospital", 1),
    ("Apollo Navi Mumbai", 1),
    ("Fortis Mohali", 1),
    ("KIMS Secunderabad", 1),
    ("Dr. LH Hiranandani Hospital", 1),
    ("St. John's Medical College Hospital", 1),
    ("Deenanath Mangeshkar Hospital", 1),
    ("AIG Hospitals", 1),
    ("Apollo Ahmedabad", 1),
    ("Apollo Indore", 1),
    ("Apollo Pune", 1),
    ("Apollo Bhopal", 1),
    ("Apollo Mysore", 1),
    ("Apollo Raipur", 1),
    ("Apollo Kanpur", 1),
]


def build_medication_library() -> Dict[str, List[str]]:
    """Derive the medication library from the disease plans.

    Maps each reusable treatment item -> diseases that prescribe it. This is the
    Medication Library layer of the domain model: one medication appearing across
    several diseases introduces realistic retrieval ambiguity.
    """
    library: Dict[str, List[str]] = defaultdict(list)
    for disease in DISEASES:
        items = {item for plan in disease.plans for item in plan}
        for item in items:
            library[item].append(disease.name)
    return dict(library)


MEDICATION_LIBRARY: Dict[str, List[str]] = build_medication_library()

EMAIL_DOMAINS = ("example.com", "example.net", "example.org", "example.in")


# --------------------------------------------------------------------------- #
# Patient generator
# --------------------------------------------------------------------------- #


def build_primary_slots(num_patients: int, diseases: List[Disease]) -> List[Disease]:
    """Allocate patient slots to primary diseases proportional to their weights.

    Weights are scaled so the primary-diagnosis counts sum exactly to
    `num_patients`, then the slot list is shuffled deterministically. This gives a
    stable long-tail distribution instead of high-variance multinomial sampling.
    """
    total = sum(d.weight for d in diseases)
    scaled = {d.name: max(1, round(d.weight * num_patients / total)) for d in diseases}
    diff = num_patients - sum(scaled.values())
    order = sorted(diseases, key=lambda d: scaled[d.name], reverse=True)
    for idx in range(abs(diff)):
        target = order[idx % len(order)]
        scaled[target.name] += 1 if diff > 0 else -1
        if scaled[target.name] < 1:
            scaled[target.name] = 1

    slots: List[Disease] = []
    for disease in diseases:
        slots.extend([disease] * scaled[disease.name])
    random.shuffle(slots)
    return slots


def generate_patient(i: int, primary: Disease, used: dict, doctors: List[str]) -> dict:
    """Generate one patient from the shared knowledge libraries."""
    patient = {}
    patient["mrn"] = f"MRN{1001 + i}"
    patient["index"] = i

    # Gender and name (Faker en_IN, unique within the dataset).
    gender = random.choice(["male", "female"])
    while True:
        name = fake.name_male() if gender == "male" else fake.name_female()
        if name not in used["names"]:
            used["names"].add(name)
            break
    patient["gender"] = gender
    patient["name"] = name

    # Hospital (weighted) and doctor (reused roster).
    hospitals, weights = zip(*HOSPITALS)
    patient["hospital"] = random.choices(hospitals, weights=weights, k=1)[0]
    patient["doctor"] = random.choice(doctors)

    # Age follows the primary disease's demographic profile.
    patient["age"] = random.randint(*primary.age_range)

    # Secondary diagnoses: 0/1/2 with comorbidity awareness.
    secondaries = pick_secondary_diagnoses(primary, 2)

    # Diagnosis list (primary first, then secondaries).
    diagnoses = [primary] + secondaries
    patient["diagnoses"] = [d.name for d in diagnoses]

    # Presenting complaint comes from the primary disease.
    patient["complaint"] = random.choice(primary.complaints)

    # Treatment plan: primary plan + one item per secondary, capped at 4.
    treatment = list(random.choice(primary.plans))
    for secondary in secondaries:
        if len(treatment) >= 4:
            break
        candidate = random.choice(secondary.plans)
        for item in candidate:
            if len(treatment) >= 4:
                break
            if item not in treatment:
                treatment.append(item)
    patient["treatment"] = treatment

    # Follow-up: primary template (+ one from a secondary if present).
    followups = [random.choice(primary.follow_ups)]
    if secondaries:
        followups.append(random.choice(secondaries[0].follow_ups))
    patient["followup"] = "; ".join(followups)

    # Clinical notes: exam fragments from primary (+ first secondary).
    notes = [random.choice(primary.exam_notes)]
    if secondaries and len(notes) < 2:
        notes.append(random.choice(secondaries[0].exam_notes))
    patient["notes"] = "On examination, " + "; ".join(notes) + "."

    # PHI (uniqueness enforced by construction).
    patient["admission_date"] = date(2025, 1, 1) + timedelta(days=random.randint(0, 364))

    phone = random.randint(6, 9)
    while True:
        candidate = f"{phone}{random.randint(100000000, 999999999)}"
        if candidate not in used["phones"]:
            used["phones"].add(candidate)
            break
    patient["phone"] = candidate

    aadhaar = 612345678901 + i
    s = str(aadhaar)
    patient["aadhaar"] = f"{s[:4]} {s[4:8]} {s[8:12]}"

    while True:
        pan = (
            "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
            + f"{1000 + (i % 9000):04d}"
            + chr(65 + (i % 26))
        )
        if pan not in used["pans"]:
            used["pans"].add(pan)
            break
    patient["pan"] = pan

    first = _alphanumeric_lower(name.split()[0])
    last = _alphanumeric_lower(name.split()[-1])
    while True:
        email = f"{first}.{last}@{(EMAIL_DOMAINS[i % len(EMAIL_DOMAINS)])}"
        if email not in used["emails"]:
            used["emails"].add(email)
            break
        email = f"{first}.{last}{i}@{(EMAIL_DOMAINS[i % len(EMAIL_DOMAINS)])}"
        if email not in used["emails"]:
            used["emails"].add(email)
            break
    patient["email"] = email

    return patient


def pick_secondary_diagnoses(primary: Disease, max_count: int) -> List[Disease]:
    """Pick 0..max_count distinct secondary diagnoses biased toward comorbidities."""
    count = random.choices([0, 1, 2], weights=[40, 40, 20], k=1)[0]
    count = min(count, max_count)
    if count == 0:
        return []

    candidates: List[Disease] = []
    weights: List[int] = []
    for disease in DISEASES:
        if disease.name == primary.name:
            continue
        if disease.name in primary.comorbidities:
            candidates.append(disease)
            weights.append(3)
        else:
            candidates.append(disease)
            weights.append(1)

    picked: List[Disease] = []
    for _ in range(count):
        if not candidates:
            break
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        picked.append(chosen)
        idx = candidates.index(chosen)
        candidates.pop(idx)
        weights.pop(idx)
    return picked


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _alphanumeric_lower(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def render_patient(p: dict) -> str:
    diagnoses = "; ".join(p["diagnoses"])
    treatments = ", ".join(p["treatment"])
    return (
        f"{p['name']}, a {p['age']}-year-old {p['gender']} patient, was admitted to "
        f"{p['hospital']} on {p['admission_date'].isoformat()} with {p['complaint']}.\n"
        f"Medical ID: {p['mrn']}, Diagnosis: {diagnoses}. Treatment: {treatments}.\n"
        f"Notes: {p['notes']} Treated by Dr. {p['doctor']}. Follow-up: {p['followup']}.\n"
        f"Contact: {p['phone']}, Aadhaar: {p['aadhaar']}, PAN: {p['pan']}, "
        f"email: {p['email']}."
    )


def generate_dataset(num_patients: int = NUM_PATIENTS, seed: int = SEED) -> List[dict]:
    seed_all(seed)
    doctors = build_doctor_roster(seed=seed)
    slots = build_primary_slots(num_patients, DISEASES)
    used = {"names": set(), "phones": set(), "pans": set(), "emails": set()}
    return [generate_patient(i, slots[i], used, doctors) for i in range(num_patients)]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    Faker.seed(seed)


def build_doctor_roster(n: int = 24, seed: int = SEED) -> List[str]:
    seed_all(seed)
    roster = []
    while len(roster) < n:
        name = fake.name()
        if name not in roster:
            roster.append(name)
    return roster


def validate(patients: List[dict]) -> dict:
    stats = {
        "total_patients": len(patients),
        "unique_diseases": len({d for p in patients for d in p["diagnoses"]}),
        "unique_treatments": len({t for p in patients for t in p["treatment"]}),
        "disease_frequency": Counter(d for p in patients for d in p["diagnoses"]),
        "treatment_frequency": Counter(t for p in patients for t in p["treatment"]),
        "avg_diagnoses": sum(len(p["diagnoses"]) for p in patients) / len(patients),
        "avg_treatments": sum(len(p["treatment"]) for p in patients) / len(patients),
        "diagnoses_histogram": Counter(len(p["diagnoses"]) for p in patients),
        "treatments_histogram": Counter(len(p["treatment"]) for p in patients),
        "hospital_frequency": Counter(p["hospital"] for p in patients),
        "gender_frequency": Counter(p["gender"] for p in patients),
        "duplicate_phi": {},
        "medication_reuse": {},
    }

    # Duplicate PHI check.
    phi_fields = ["name", "phone", "aadhaar", "pan", "email", "mrn"]
    seen = {f: Counter(p[f] for p in patients) for f in phi_fields}
    for f in phi_fields:
        dupes = {value: count for value, count in seen[f].items() if count > 1}
        if dupes:
            stats["duplicate_phi"][f] = dupes

    # Medication reuse across diseases (Medication Library check).
    for med, diseases in MEDICATION_LIBRARY.items():
        if len(diseases) >= 3:
            stats["medication_reuse"][med] = diseases

    return stats


def deterministic_check(patients: List[dict], seed: int = SEED) -> Tuple[bool, str]:
    regenerated = generate_dataset(len(patients), seed=seed)
    expected = "\n\n".join(render_patient(p) for p in patients) + "\n"
    actual = "\n\n".join(render_patient(p) for p in regenerated) + "\n"
    return expected == actual, "byte-for-byte identical under the same seed"


def pipeline_compat_check(path: Path) -> dict:
    """Smoke-test ingestion compatibility without modifying secure_rag."""
    result = {"records": 0, "mask_tokens_seen": {}, "status": "skipped"}
    try:
        from secure_rag.masker import mask_text
        from secure_rag.pdf_loader import split_into_records
    except Exception as exc:  # pragma: no cover - environment dependent
        result["status"] = f"skipped (secure_rag unavailable: {exc})"
        return result

    text = path.read_text(encoding="utf-8")
    records = split_into_records(text)
    result["records"] = len(records)

    sample = "\n\n".join(records[:5])
    masked = mask_text(sample)
    tokens = [
        ("[PHONE_MASKED]", "phone"),
        ("[AADHAAR_MASKED]", "aadhaar"),
        ("[PAN_MASKED]", "pan"),
        ("[EMAIL_MASKED]", "email"),
        ("[PATIENT_ID_MASKED]", "medical_id"),
    ]
    result["mask_tokens_seen"] = {
        label: (token in masked) for token, label in tokens
    }
    result["status"] = "passed"
    return result


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def write_distribution_report(patients: List[dict], stats: dict, path: Path) -> None:
    lines = [
        "# Distribution Statistics",
        "",
        f"_Generated {date.today().isoformat()} from `data/generate_dataset.py`_",
        "",
        f"- Total patients: {stats['total_patients']}",
        f"- Unique diseases: {stats['unique_diseases']}",
        f"- Unique treatments: {stats['unique_treatments']}",
        f"- Average diagnoses per patient: {stats['avg_diagnoses']:.2f}",
        f"- Average treatments per patient: {stats['avg_treatments']:.2f}",
        "",
        "## Disease frequency (patients containing the disease)",
        "",
        "| Rank | Disease | Patients |",
        "|---|---|---|",
    ]
    for rank, (disease, count) in enumerate(stats["disease_frequency"].most_common(), 1):
        lines.append(f"| {rank} | {disease} | {count} |")

    lines += [
        "",
        "## Treatment frequency (patients prescribed the treatment)",
        "",
        "| Rank | Treatment | Patients |",
        "|---|---|---|",
    ]
    for rank, (treatment, count) in enumerate(stats["treatment_frequency"].most_common(), 1):
        lines.append(f"| {rank} | {treatment} | {count} |")

    lines += [
        "",
        "## Diagnoses per patient",
        "",
    ]
    for n in sorted(stats["diagnoses_histogram"]):
        lines.append(f"- {n} diagnosis: {stats['diagnoses_histogram'][n]} patients")

    lines += [
        "",
        "## Treatments per patient",
        "",
    ]
    for n in sorted(stats["treatments_histogram"]):
        lines.append(f"- {n} treatments: {stats['treatments_histogram'][n]} patients")

    lines += [
        "",
        "## Hospital frequency (top 20)",
        "",
        "| Rank | Hospital | Patients |",
        "|---|---|---|",
    ]
    for rank, (hospital, count) in enumerate(stats["hospital_frequency"].most_common(20), 1):
        lines.append(f"| {rank} | {hospital} | {count} |")

    lines += [
        "",
        "## Medication reuse across diseases (Medication Library)",
        "",
        "Medications prescribed by three or more distinct diseases:",
        "",
        "| Medication | Diseases |",
        "|---|---|",
    ]
    for med, diseases in sorted(stats["medication_reuse"].items()):
        lines.append(f"| {med} | {', '.join(sorted(diseases))} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(patients: List[dict], stats: dict, compat: dict, path: Path, seed: int = SEED) -> None:
    deterministic, det_detail = deterministic_check(patients, seed=seed)
    dup_phi = stats["duplicate_phi"]

    lines = [
        "# Dataset Validation Report",
        "",
        f"_Generated {date.today().isoformat()} from `data/generate_dataset.py`_",
        "",
        "## Summary",
        "",
        f"- Total patients: {stats['total_patients']}",
        f"- Unique diseases: {stats['unique_diseases']}",
        f"- Unique treatments: {stats['unique_treatments']}",
        f"- Average diagnoses per patient: {stats['avg_diagnoses']:.2f}",
        f"- Average treatments per patient: {stats['avg_treatments']:.2f}",
        f"- Gender split: {dict(stats['gender_frequency'])}",
        "",
        "## Disease frequency distribution",
        "",
        "| Rank | Disease | Patients |",
        "|---|---|---|",
    ]
    for rank, (disease, count) in enumerate(stats["disease_frequency"].most_common(), 1):
        lines.append(f"| {rank} | {disease} | {count} |")

    lines += [
        "",
        "## Treatment frequency distribution (top 25)",
        "",
        "| Rank | Treatment | Patients |",
        "|---|---|---|",
    ]
    for rank, (treatment, count) in enumerate(stats["treatment_frequency"].most_common(25), 1):
        lines.append(f"| {rank} | {treatment} | {count} |")

    lines += [
        "",
        "## Diagnoses per patient",
        "",
    ]
    for n in sorted(stats["diagnoses_histogram"]):
        lines.append(f"- {n} diagnoses: {stats['diagnoses_histogram'][n]} patients")

    lines += [
        "",
        "## Treatments per patient",
        "",
    ]
    for n in sorted(stats["treatments_histogram"]):
        lines.append(f"- {n} treatments: {stats['treatments_histogram'][n]} patients")

    lines += [
        "",
        "## Duplicate PHI check",
        "",
        "Verifies that no PHI value (name, phone, Aadhaar, PAN, email, MRN) is shared",
        "between two patients.",
        "",
    ]
    if dup_phi:
        lines.append("FAILED:")
        for field, dupes in dup_phi.items():
            lines.append(f"- `{field}`: {len(dupes)} duplicated value(s)")
    else:
        lines.append("PASSED - no duplicate PHI across any patient.")

    lines += [
        "",
        "## Deterministic regeneration check",
        "",
        f"{'PASSED' if deterministic else 'FAILED'} - regenerating with seed {seed} "
        f"produces a dataset that is {det_detail}.",
        "",
        "## Medication reuse check (Medication Library)",
        "",
        f"{len(stats['medication_reuse'])} medication(s) prescribed by three or more distinct diseases.",
        "",
        "## Pipeline compatibility smoke test",
        "",
    ]
    if compat["status"] == "passed":
        lines.append(
            f"PASSED - `split_into_records` produced {compat['records']} records and "
            "the masker recognized the following PII patterns on a sample:"
        )
        for label, seen in compat["mask_tokens_seen"].items():
            lines.append(f"- {label}: {'detected' if seen else 'NOT detected'}")
    else:
        lines.append(compat["status"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-patients", type=int, default=NUM_PATIENTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--no-reports", action="store_true")
    parser.add_argument("--compat-check", action="store_true")
    args = parser.parse_args(argv)

    patients = generate_dataset(num_patients=args.num_patients, seed=args.seed)
    rendered = "\n\n".join(render_patient(p) for p in patients) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} ({len(patients)} patients)")

    if args.no_reports:
        return 0

    stats = validate(patients)
    write_distribution_report(patients, stats, DISTRIBUTION_PATH)
    print(f"Wrote {DISTRIBUTION_PATH}")

    compat = {"status": "skipped"}
    if args.compat_check:
        compat = pipeline_compat_check(args.output)
    write_validation_report(patients, stats, compat, REPORT_PATH, seed=args.seed)
    print(f"Wrote {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
