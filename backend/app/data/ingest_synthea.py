"""Ingest Synthea-generated patient data into PostgreSQL — v3.1.
Now generates: pharmacy_refills, trust_scores, reminder_logs.

Usage:
    # With Synthea data:
    python -m app.data.ingest_synthea --synthea-dir ./synthea_output/csv

    # Pure synthetic fallback (no Synthea needed):
    python -m app.data.ingest_synthea --fallback --max-patients 100000
"""
import argparse
import csv
import logging
import random
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.data.models_pg import (
    Base, Patient, Diagnosis, Medication, AdherenceEvent,
    LabResult, Vital, Encounter, PharmacyRefill, TrustScore,
)
from app.data.database_pg import get_sync_engine, get_sync_session_factory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INSURANCE_TYPES = ["private", "medicare", "medicaid", "uninsured"]
ARCHETYPES = ["excellent", "good", "moderate", "poor", "erratic"]
ARCHETYPE_WEIGHTS = [0.15, 0.30, 0.25, 0.20, 0.10]
SDOH_FACTORS = [
    "food_insecurity", "housing_instability", "transportation_barrier",
    "social_isolation", "financial_strain", "low_health_literacy",
]
ARCHETYPE_RATES = {
    "excellent": (0.92, 0.05), "good": (0.80, 0.08),
    "moderate": (0.65, 0.12), "poor": (0.40, 0.15),
    "erratic": (0.55, 0.25),
}
MED_NAMES = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
             "albuterol", "sertraline", "losartan", "furosemide", "gabapentin"]
PHARMACIES = ["CVS", "Walgreens", "Rite Aid", "Walmart", "Costco", "Mail Order", "Amazon Pharmacy"]
CONDITION_SEVERITY_MAP = {
    "diabetes": "moderate", "hypertension": "moderate", "asthma": "mild",
    "copd": "severe", "heart failure": "severe", "depression": "moderate", "anxiety": "mild",
}


def parse_synthea_date(s: str):
    if not s: return None
    try: return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError: return None


def ingest_synthea(synthea_dir: str, max_patients: int = 100_000):
    synthea_path = Path(synthea_dir)
    patients_file = synthea_path / "patients.csv"
    if not patients_file.exists():
        logger.info(f"patients.csv not found in {synthea_dir}, falling back to synthetic generation...")
        generate_fallback(max_patients)
        return

    engine = get_sync_engine()
    Base.metadata.create_all(engine)
    SessionLocal = get_sync_session_factory()

    logger.info("Phase 1/8: Loading Synthea patients...")
    synthea_to_uuid = {}
    batch = []
    with open(patients_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_patients: break
            sid = row.get("Id", "")
            pid = uuid.uuid4()
            synthea_to_uuid[sid] = pid
            dob = parse_synthea_date(row.get("BIRTHDATE", ""))
            age = (date.today() - dob).days // 365 if dob else random.randint(18, 90)
            gender = "M" if row.get("GENDER", "").upper() == "M" else "F"
            archetype = random.choices(ARCHETYPES, weights=ARCHETYPE_WEIGHTS, k=1)[0]
            n_sdoh = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
            batch.append(Patient(
                patient_id=pid, first_name=row.get("FIRST", f"Patient_{i}"),
                last_name=row.get("LAST", f"Last_{i}"), date_of_birth=dob, age=age,
                gender=gender, race=row.get("RACE", "white").lower().replace(" ", "_"),
                ethnicity=row.get("ETHNICITY", ""), address_city=row.get("CITY", ""),
                address_state=row.get("STATE", ""), address_zip=row.get("ZIP", ""),
                insurance_type=random.choice(INSURANCE_TYPES),
                bmi=round(max(16, min(50, random.gauss(27, 5))), 1),
                smoker=random.random() < 0.15, n_sdoh_risks=n_sdoh,
                sdoh_risk_factors=random.sample(SDOH_FACTORS, n_sdoh) if n_sdoh else [],
                adherence_archetype=archetype,
            ))
            if len(batch) >= 5000:
                with SessionLocal() as session: session.add_all(batch); session.commit()
                logger.info(f"  {i + 1} patients..."); batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info(f"  Total patients: {len(synthea_to_uuid)}")

    # Phases 2-8: conditions, meds, adherence, pharmacy, labs, vitals, encounters, trust
    patient_map = {str(pid): pid for pid in synthea_to_uuid.values()}
    _generate_all_related_data(SessionLocal, patient_map)
    logger.info("Ingestion complete!")


def generate_fallback(n_patients: int = 100_000):
    logger.info(f"Generating {n_patients:,} synthetic patients...")
    random.seed(42)
    np.random.seed(42)

    engine = get_sync_engine()
    Base.metadata.create_all(engine)
    SessionLocal = get_sync_session_factory()

    FIRST_M = ["James","John","Robert","Michael","David","William","Richard","Joseph","Thomas","Christopher",
               "Daniel","Matthew","Anthony","Mark","Steven","Paul","Andrew","Joshua","Kenneth","Kevin"]
    FIRST_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth","Susan","Jessica","Sarah","Karen",
               "Lisa","Nancy","Betty","Margaret","Sandra","Ashley","Emily","Donna","Michelle","Carol"]
    LAST = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
            "Hernandez","Lopez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin","Lee",
            "Perez","Thompson","White","Harris","Clark","Lewis","Robinson","Walker","Young","Allen",
            "King","Wright","Scott","Torres","Nguyen","Hill","Flores","Green","Adams","Nelson",
            "Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts","Sanchez","Ramirez","Phillips"]
    RACES = ["white","black","hispanic","asian","native_american","other"]
    RACE_W = [0.58, 0.13, 0.19, 0.06, 0.01, 0.03]

    patient_map = {}
    batch = []
    for i in range(n_patients):
        pid = uuid.uuid4()
        gender = random.choice(["M", "F"])
        first = random.choice(FIRST_M if gender == "M" else FIRST_F)
        last = random.choice(LAST)
        age = random.randint(18, 90)
        archetype = random.choices(ARCHETYPES, weights=ARCHETYPE_WEIGHTS, k=1)[0]
        n_sdoh = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
        patient_map[str(pid)] = pid
        batch.append(Patient(
            patient_id=pid, first_name=first, last_name=last, age=age, gender=gender,
            race=random.choices(RACES, weights=RACE_W, k=1)[0],
            ethnicity=random.choice(["non-hispanic", "hispanic"]),
            address_city=random.choice(["Boston","New York","Chicago","Houston","Phoenix","Philadelphia"]),
            address_state=random.choice(["MA","NY","IL","TX","AZ","PA","CA","FL","OH","GA"]),
            address_zip=str(random.randint(10000, 99999)),
            insurance_type=random.choice(INSURANCE_TYPES),
            bmi=round(max(16, min(50, random.gauss(27, 5))), 1),
            smoker=random.random() < 0.15, n_sdoh_risks=n_sdoh,
            sdoh_risk_factors=random.sample(SDOH_FACTORS, n_sdoh) if n_sdoh else [],
            adherence_archetype=archetype,
        ))
        if len(batch) >= 5000:
            with SessionLocal() as session: session.add_all(batch); session.commit()
            if (i + 1) % 25000 == 0: logger.info(f"  {i + 1:,} patients...")
            batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info(f"  {n_patients:,} patients created")

    _generate_all_related_data(SessionLocal, patient_map)
    logger.info(f"Fallback generation complete: {n_patients:,} patients")


def _generate_all_related_data(SessionLocal, patient_map: dict):
    """Generate adherence, pharmacy refills, labs, vitals, encounters, trust scores."""
    today = date.today()
    start_date = today - timedelta(days=5 * 365)

    # Load patient archetypes
    with SessionLocal() as session:
        rows = list(session.execute(text("SELECT patient_id, adherence_archetype FROM patients")))
    archetypes = {str(r[0]): r[1] or "moderate" for r in rows}
    pids = list(archetypes.keys())

    # Phase: Adherence events
    logger.info("Phase 2/8: Generating adherence events (5 years)...")
    batch = []
    count = 0
    for pid in pids:
        archetype = archetypes[pid]
        base_rate, variance = ARCHETYPE_RATES.get(archetype, (0.65, 0.12))
        n_meds = random.randint(1, 3)
        patient_meds = random.sample(MED_NAMES, min(n_meds, len(MED_NAMES)))
        for day_offset in range(0, 5 * 365, random.choice([1, 2, 3])):
            d = start_date + timedelta(days=day_offset)
            daily_rate = max(0, min(1, random.gauss(base_rate, variance)))
            for med in patient_meds:
                taken = random.random() < daily_rate
                latency = round(abs(random.gauss(15, 10)), 1) if taken else None
                response = "YES" if taken else random.choice(["NO", "NO_RESPONSE", "SNOOZE"])
                confidence = max(0, min(1, base_rate + random.gauss(0, 0.1)))
                batch.append(AdherenceEvent(
                    patient_id=uuid.UUID(pid), medication_name=med, event_date=d,
                    taken=taken,
                    taken_time=f"{random.randint(6, 22):02d}:{random.randint(0, 59):02d}" if taken else None,
                    reminder_15min_sent=True, reminder_5min_sent=True,
                    missed_alert_sent=not taken and random.random() < 0.9,
                    patient_response=response,
                    response_latency_min=latency,
                    source=random.choice(["app", "sms", "smart_pillbox", "manual"]),
                    system_confidence=round(confidence, 3),
                    clinician_verified=random.random() < 0.05,
                ))
                count += 1
            if len(batch) >= 50000:
                with SessionLocal() as session: session.add_all(batch); session.commit()
                batch = []
                if count % 500000 == 0: logger.info(f"  {count:,} adherence events...")
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info(f"  Total adherence events: {count:,}")

    # Phase: Pharmacy refills
    logger.info("Phase 3/8: Generating pharmacy refills...")
    batch = []
    refill_count = 0
    for pid in pids:
        archetype = archetypes[pid]
        n_meds = random.randint(1, 3)
        patient_meds = random.sample(MED_NAMES, min(n_meds, len(MED_NAMES)))
        gap_factor = {"excellent": 0, "good": 2, "moderate": 7, "poor": 15, "erratic": random.randint(0, 20)}
        for med in patient_meds:
            supply_days = random.choice([30, 60, 90])
            last_fill = start_date
            for _ in range((5 * 365) // supply_days):
                actual_gap = supply_days + gap_factor.get(archetype, 5) + random.randint(-3, 10)
                fill_date = last_fill + timedelta(days=actual_gap)
                if fill_date > today: break
                batch.append(PharmacyRefill(
                    patient_id=uuid.UUID(pid), medication_name=med,
                    refill_date=fill_date, supply_days=supply_days,
                    expected_gap_days=supply_days, actual_gap_days=actual_gap,
                    pharmacy_name=random.choice(PHARMACIES),
                ))
                last_fill = fill_date
                refill_count += 1
            if len(batch) >= 20000:
                with SessionLocal() as session: session.add_all(batch); session.commit()
                batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info(f"  Total pharmacy refills: {refill_count:,}")

    # Phase: Labs
    logger.info("Phase 4/8: Generating labs...")
    lab_tests = ["hba1c", "ldl_cholesterol", "creatinine", "glucose", "tsh"]
    batch = []
    for pid in pids[:50000]:
        for months_ago in range(0, 60, 6):
            d = today - timedelta(days=months_ago * 30)
            for test in random.sample(lab_tests, 2):
                batch.append(LabResult(
                    patient_id=uuid.UUID(pid), test_name=test,
                    value=round(random.gauss(100, 20), 1),
                    unit="mg/dL", lab_date=d, flag="normal",
                ))
            if len(batch) >= 20000:
                with SessionLocal() as session: session.add_all(batch); session.commit()
                batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info("  Labs generated.")

    # Phase: Vitals
    logger.info("Phase 5/8: Generating vitals...")
    batch = []
    for pid in pids[:50000]:
        for months_ago in range(0, 60):
            d = today - timedelta(days=months_ago * 30 + random.randint(-5, 5))
            if d > today: continue
            batch.append(Vital(
                patient_id=uuid.UUID(pid), vital_date=d,
                systolic_bp=random.randint(100, 160), diastolic_bp=random.randint(60, 100),
                heart_rate=random.randint(55, 100),
                temperature=round(random.gauss(98.6, 0.5), 1),
                spo2=round(random.gauss(97, 2), 1),
                weight_lbs=round(random.gauss(170, 30), 1),
            ))
            if len(batch) >= 20000:
                with SessionLocal() as session: session.add_all(batch); session.commit()
                batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info("  Vitals generated.")

    # Phase: Encounters
    logger.info("Phase 6/8: Generating encounters...")
    types = ["office_visit", "office_visit", "office_visit", "er", "urgent_care", "telehealth"]
    complaints = ["routine_checkup", "medication_review", "chest_pain", "shortness_of_breath",
                   "follow_up", "vaccination", "lab_review", "acute_illness"]
    batch = []
    for pid in pids[:50000]:
        n = random.randint(5, 30)
        for _ in range(n):
            d = today - timedelta(days=random.randint(0, 5 * 365))
            batch.append(Encounter(
                patient_id=uuid.UUID(pid), encounter_date=d,
                encounter_type=random.choice(types), chief_complaint=random.choice(complaints),
            ))
            if len(batch) >= 20000:
                with SessionLocal() as session: session.add_all(batch); session.commit()
                batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info("  Encounters generated.")

    # Phase: Diagnoses (for patients without Synthea conditions)
    logger.info("Phase 7/8: Generating diagnoses...")
    CONDITIONS = {
        "diabetes_type2": 0.11, "hypertension": 0.33, "heart_failure": 0.02,
        "copd": 0.06, "asthma": 0.08, "depression": 0.07, "anxiety": 0.06,
        "ckd": 0.03, "atrial_fibrillation": 0.02, "osteoarthritis": 0.10,
    }
    batch = []
    for pid in pids:
        for cond, prev in CONDITIONS.items():
            if random.random() < prev:
                batch.append(Diagnosis(
                    patient_id=uuid.UUID(pid), condition=cond,
                    icd10_code=f"E{random.randint(10,14)}.{random.randint(0,9)}",
                    severity=random.choices(["mild","moderate","severe"], weights=[0.3,0.5,0.2])[0],
                    status=random.choices(["active","managed","resolved"], weights=[0.6,0.3,0.1])[0],
                    onset_date=today - timedelta(days=random.randint(30, 5*365)),
                ))
        if len(batch) >= 10000:
            with SessionLocal() as session: session.add_all(batch); session.commit()
            batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info("  Diagnoses generated.")

    # Phase: Medications
    logger.info("Phase 7b/8: Generating medications...")
    batch = []
    for pid in pids:
        n_meds = random.randint(1, 4)
        for med in random.sample(MED_NAMES, min(n_meds, len(MED_NAMES))):
            hour = random.randint(6, 21)
            batch.append(Medication(
                patient_id=uuid.UUID(pid), medication_name=med,
                dosage=f"{random.choice([5,10,20,50,100,250,500])}mg",
                frequency=random.choice(["once_daily","twice_daily","weekly"]),
                route="oral", start_date=today - timedelta(days=random.randint(30, 1800)),
                active=True,
                reminder_time_1=f"{hour:02d}:00",
                reminder_time_2=f"{(hour + 12) % 24:02d}:00" if random.random() < 0.3 else None,
            ))
        if len(batch) >= 10000:
            with SessionLocal() as session: session.add_all(batch); session.commit()
            batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info("  Medications generated.")

    # Phase: Trust scores
    logger.info("Phase 8/8: Computing trust scores (FR-3.2.3)...")
    TRUST_MAP = {
        "excellent": (0.92, "consistent_adherer"),
        "good": (0.78, "consistent_adherer"),
        "moderate": (0.62, "occasional_skipper"),
        "poor": (0.35, "chronic_non_adherer"),
        "erratic": (0.50, "unreliable_reporter"),
    }
    batch = []
    for pid in pids:
        archetype = archetypes[pid]
        base_trust, classification = TRUST_MAP.get(archetype, (0.6, "occasional_skipper"))
        n_meds = random.randint(1, 3)
        for med in random.sample(MED_NAMES, min(n_meds, len(MED_NAMES))):
            score = max(0, min(1, base_trust + random.gauss(0, 0.08)))
            if score >= 0.85: cls = "consistent_adherer"
            elif score >= 0.6: cls = "occasional_skipper"
            elif classification == "unreliable_reporter": cls = "unreliable_reporter"
            else: cls = "chronic_non_adherer"
            batch.append(TrustScore(
                patient_id=uuid.UUID(pid), medication_name=med,
                score=round(score, 3), classification=cls,
                components={
                    "historical_rate_30d": round(score + random.gauss(0, 0.05), 3),
                    "response_pattern": round(max(0, 1 - abs(random.gauss(0, 0.3))), 3),
                    "refill_alignment": round(max(0, min(1, score + random.gauss(0, 0.1))), 3),
                    "outcome_correlation": round(max(0, min(1, 0.7 + random.gauss(0, 0.15))), 3),
                },
            ))
        if len(batch) >= 10000:
            with SessionLocal() as session: session.add_all(batch); session.commit()
            batch = []
    if batch:
        with SessionLocal() as session: session.add_all(batch); session.commit()
    logger.info("  Trust scores computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Synthea data into PostgreSQL")
    parser.add_argument("--synthea-dir", default="./synthea_output/csv")
    parser.add_argument("--max-patients", type=int, default=100_000)
    parser.add_argument("--fallback", action="store_true", help="Skip Synthea, generate synthetic directly")
    args = parser.parse_args()
    if args.fallback:
        generate_fallback(args.max_patients)
    else:
        ingest_synthea(args.synthea_dir, args.max_patients)
