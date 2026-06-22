"""
Synthetic patient data generator v2.
50K patients, 5 years longitudinal, medication reminder simulation.
Outputs to SQLite for fast indexed queries at scale.
"""
import json, random, uuid, sqlite3, os, time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

NUM_PATIENTS = 50_000
YEARS_OF_DATA = 5
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "predictive_care.db"

# --- Name pools ---
FIRST_NAMES_M = ["James","John","Robert","Michael","David","William","Richard","Joseph","Thomas","Christopher",
    "Daniel","Matthew","Anthony","Mark","Steven","Paul","Andrew","Joshua","Kenneth","Kevin",
    "Brian","George","Timothy","Ronald","Edward","Jason","Jeffrey","Ryan","Jacob","Gary",
    "Nicholas","Eric","Jonathan","Stephen","Larry","Justin","Scott","Brandon","Benjamin","Samuel"]
FIRST_NAMES_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth","Susan","Jessica","Sarah","Karen",
    "Lisa","Nancy","Betty","Margaret","Sandra","Ashley","Dorothy","Kimberly","Emily","Donna",
    "Michelle","Carol","Amanda","Melissa","Deborah","Stephanie","Rebecca","Sharon","Laura","Cynthia",
    "Kathleen","Amy","Angela","Shirley","Anna","Brenda","Pamela","Emma","Nicole","Helen"]
LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
    "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
    "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts"]

RACES = ["white","black","hispanic","asian","native_american","pacific_islander","other"]
RACE_WEIGHTS = [0.58, 0.13, 0.19, 0.06, 0.01, 0.005, 0.015]
GENDERS = ["M","F"]
BLOOD_TYPES = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
INSURANCE_TYPES = ["medicare","medicaid","private","employer","self_pay","tricare"]
INSURANCE_WEIGHTS = [0.18, 0.21, 0.22, 0.30, 0.07, 0.02]

CONDITIONS = {
    "diabetes_type2": {"prevalence": 0.11, "meds": ["metformin","glipizide","insulin_glargine","sitagliptin"]},
    "hypertension": {"prevalence": 0.33, "meds": ["lisinopril","amlodipine","losartan","hydrochlorothiazide"]},
    "heart_failure": {"prevalence": 0.02, "meds": ["carvedilol","furosemide","spironolactone","sacubitril_valsartan"]},
    "copd": {"prevalence": 0.06, "meds": ["albuterol","tiotropium","fluticasone_salmeterol","prednisone"]},
    "asthma": {"prevalence": 0.08, "meds": ["albuterol","montelukast","fluticasone","budesonide"]},
    "depression": {"prevalence": 0.07, "meds": ["sertraline","escitalopram","bupropion","duloxetine"]},
    "anxiety": {"prevalence": 0.06, "meds": ["escitalopram","buspirone","sertraline","hydroxyzine"]},
    "ckd": {"prevalence": 0.03, "meds": ["lisinopril","amlodipine","sodium_bicarbonate","calcitriol"]},
    "atrial_fibrillation": {"prevalence": 0.02, "meds": ["apixaban","metoprolol","amiodarone","diltiazem"]},
    "osteoarthritis": {"prevalence": 0.10, "meds": ["acetaminophen","meloxicam","diclofenac","duloxetine"]},
}

ADHERENCE_ARCHETYPES = {
    "excellent": {"base_rate": 0.95, "drift": -0.01, "weight": 0.25},
    "good": {"base_rate": 0.82, "drift": -0.02, "weight": 0.30},
    "moderate": {"base_rate": 0.65, "drift": -0.04, "weight": 0.25},
    "poor": {"base_rate": 0.40, "drift": -0.06, "weight": 0.15},
    "erratic": {"base_rate": 0.55, "drift": -0.03, "weight": 0.05},
}

SDOH_FACTORS = ["food_insecurity","housing_instability","transportation_barrier",
                "social_isolation","financial_strain","low_health_literacy"]


def _pick_archetype():
    names = list(ADHERENCE_ARCHETYPES.keys())
    weights = [ADHERENCE_ARCHETYPES[n]["weight"] for n in names]
    return random.choices(names, weights=weights, k=1)[0]


def _gen_patients(n):
    patients = []
    today = date.today()
    for i in range(n):
        gender = random.choice(GENDERS)
        first = random.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        age = random.randint(18, 92)
        dob = today - timedelta(days=age * 365 + random.randint(0, 364))
        race = random.choices(RACES, weights=RACE_WEIGHTS, k=1)[0]
        bmi = round(np.clip(np.random.normal(28.5, 6.0), 16, 55), 1)
        sdoh_risks = random.sample(SDOH_FACTORS, k=random.choices([0,1,2,3], weights=[0.4,0.3,0.2,0.1])[0])
        archetype = _pick_archetype()

        patients.append({
            "patient_id": str(uuid.uuid4()),
            "first_name": first,
            "last_name": last,
            "gender": gender,
            "date_of_birth": dob.isoformat(),
            "age": age,
            "race": race,
            "blood_type": random.choice(BLOOD_TYPES),
            "bmi": bmi,
            "insurance_type": random.choices(INSURANCE_TYPES, weights=INSURANCE_WEIGHTS, k=1)[0],
            "zipcode": f"{random.randint(10000,99999)}",
            "sdoh_risk_factors": json.dumps(sdoh_risks),
            "n_sdoh_risks": len(sdoh_risks),
            "adherence_archetype": archetype,
            "created_at": (today - timedelta(days=random.randint(0, YEARS_OF_DATA * 365))).isoformat(),
        })
        if (i + 1) % 10000 == 0:
            print(f"  patients: {i+1}/{n}")
    return patients


def _gen_diagnoses(patients):
    diagnoses = []
    today = date.today()
    for p in patients:
        age = p["age"]
        for cond, info in CONDITIONS.items():
            age_factor = 1.0 + max(0, (age - 50)) * 0.01
            if random.random() < info["prevalence"] * age_factor:
                dx_date = today - timedelta(days=random.randint(30, YEARS_OF_DATA * 365))
                diagnoses.append({
                    "diagnosis_id": str(uuid.uuid4()),
                    "patient_id": p["patient_id"],
                    "condition": cond,
                    "icd10_code": f"E{random.randint(10,14)}.{random.randint(0,9)}" if "diabetes" in cond else f"I{random.randint(10,50)}.{random.randint(0,9)}",
                    "diagnosed_date": dx_date.isoformat(),
                    "status": random.choices(["active","managed","resolved"], weights=[0.6,0.3,0.1])[0],
                    "severity": random.choices(["mild","moderate","severe"], weights=[0.3,0.5,0.2])[0],
                })
    return diagnoses


def _gen_medications(patients, diagnoses):
    meds = []
    dx_by_patient = {}
    for d in diagnoses:
        dx_by_patient.setdefault(d["patient_id"], []).append(d)

    for p in patients:
        patient_dx = dx_by_patient.get(p["patient_id"], [])
        for dx in patient_dx:
            if dx["status"] == "resolved":
                continue
            cond_meds = CONDITIONS.get(dx["condition"], {}).get("meds", [])
            n_meds = random.choices([1, 2], weights=[0.7, 0.3])[0]
            for med_name in random.sample(cond_meds, min(n_meds, len(cond_meds))):
                meds.append({
                    "medication_id": str(uuid.uuid4()),
                    "patient_id": p["patient_id"],
                    "medication_name": med_name,
                    "condition": dx["condition"],
                    "dosage": f"{random.choice([5,10,20,25,50,100,250,500])}mg",
                    "frequency": random.choice(["daily","twice_daily","weekly"]),
                    "prescribed_date": dx["diagnosed_date"],
                    "status": "active",
                    # Scheduled reminder times (HH:MM)
                    "reminder_time_1": f"{random.randint(6,10):02d}:{random.choice(['00','15','30','45'])}",
                    "reminder_time_2": f"{random.randint(18,22):02d}:{random.choice(['00','15','30','45'])}" if random.random() < 0.3 else None,
                })
    return meds


def _gen_adherence(patients, medications):
    """Generate daily adherence events with 5-min/15-min reminder simulation."""
    events = []
    today = date.today()
    meds_by_patient = {}
    for m in medications:
        meds_by_patient.setdefault(m["patient_id"], []).append(m)

    patient_map = {p["patient_id"]: p for p in patients}
    archetypes_map = {}
    batch = []

    for idx, p in enumerate(patients):
        pid = p["patient_id"]
        archetype = p["adherence_archetype"]
        arch_cfg = ADHERENCE_ARCHETYPES[archetype]
        archetypes_map[pid] = archetype
        p_meds = meds_by_patient.get(pid, [])
        if not p_meds:
            continue

        days = YEARS_OF_DATA * 365
        base_rate = arch_cfg["base_rate"]
        drift = arch_cfg["drift"]

        for med in p_meds:
            reminder_time = med["reminder_time_1"]
            rh, rm = int(reminder_time.split(":")[0]), int(reminder_time.split(":")[1])

            for d in range(0, days, random.choice([1, 1, 1, 2])):  # skip some days for realism
                event_date = today - timedelta(days=days - d)
                # Adherence probability drifts over years
                year_fraction = d / 365.0
                prob = np.clip(base_rate + drift * year_fraction + random.gauss(0, 0.05), 0.05, 0.99)

                # Erratic archetype has high variance
                if archetype == "erratic":
                    prob = np.clip(prob + random.gauss(0, 0.20), 0.05, 0.99)

                taken = random.random() < prob

                # Simulate reminder sequence: 15-min warning, 5-min warning, due, taken/missed
                reminder_15_sent = True
                reminder_5_sent = True
                if taken:
                    # Response latency: how many minutes after due time they took it
                    response_latency = max(0, int(np.random.exponential(15)))
                    taken_hour = rh + (rm + response_latency) // 60
                    taken_min = (rm + response_latency) % 60
                    taken_time = f"{min(taken_hour,23):02d}:{taken_min:02d}"
                else:
                    response_latency = None
                    taken_time = None

                batch.append({
                    "event_id": str(uuid.uuid4()),
                    "patient_id": pid,
                    "medication_id": med["medication_id"],
                    "medication_name": med["medication_name"],
                    "event_date": event_date.isoformat(),
                    "scheduled_time": reminder_time,
                    "reminder_15min_sent": reminder_15_sent,
                    "reminder_5min_sent": reminder_5_sent,
                    "taken": taken,
                    "taken_time": taken_time,
                    "response_latency_min": response_latency,
                    "self_reported": random.random() < 0.1,
                    "source": random.choices(["smart_pillbox", "app_confirm", "self_report", "pharmacy_data"],
                                              weights=[0.3, 0.4, 0.15, 0.15])[0],
                })

                if len(batch) >= 50000:
                    events.extend(batch)
                    batch = []

        if (idx + 1) % 10000 == 0:
            print(f"  adherence: {idx+1}/{len(patients)} patients")

    events.extend(batch)
    return events, archetypes_map


def _gen_pharmacy(patients, medications, archetypes_map):
    refills = []
    today = date.today()
    meds_by_patient = {}
    for m in medications:
        meds_by_patient.setdefault(m["patient_id"], []).append(m)

    for p in patients:
        pid = p["patient_id"]
        arch = archetypes_map.get(pid, "moderate")
        for med in meds_by_patient.get(pid, []):
            supply_days = random.choice([30, 60, 90])
            n_refills = (YEARS_OF_DATA * 365) // supply_days
            last_fill = today - timedelta(days=YEARS_OF_DATA * 365)
            for _ in range(n_refills):
                gap_factor = {"excellent": 0, "good": 2, "moderate": 7, "poor": 15, "erratic": random.randint(0, 20)}
                actual_gap = supply_days + gap_factor.get(arch, 5) + random.randint(-3, 10)
                fill_date = last_fill + timedelta(days=actual_gap)
                if fill_date > today:
                    break
                refills.append({
                    "refill_id": str(uuid.uuid4()),
                    "patient_id": pid,
                    "medication_id": med["medication_id"],
                    "medication_name": med["medication_name"],
                    "fill_date": fill_date.isoformat(),
                    "supply_days": supply_days,
                    "expected_gap_days": supply_days,
                    "actual_gap_days": actual_gap,
                    "pharmacy_name": random.choice(["CVS","Walgreens","RiteAid","Walmart","Costco","Mail Order"]),
                })
                last_fill = fill_date
    return refills


def _gen_labs(patients, diagnoses, archetypes_map):
    labs = []
    today = date.today()
    dx_by_patient = {}
    for d in diagnoses:
        dx_by_patient.setdefault(d["patient_id"], []).append(d["condition"])

    for p in patients:
        pid = p["patient_id"]
        conds = set(dx_by_patient.get(pid, []))
        arch = archetypes_map.get(pid, "moderate")
        adherence_factor = {"excellent": 0, "good": 0.5, "moderate": 1.0, "poor": 1.5, "erratic": 1.2}.get(arch, 1.0)

        # Generate labs every 3-6 months over the full period
        for months_ago in range(0, YEARS_OF_DATA * 12, random.randint(3, 6)):
            lab_date = today - timedelta(days=months_ago * 30 + random.randint(-10, 10))
            if lab_date > today:
                continue

            # Year-based drift for longitudinal realism
            year = months_ago / 12.0

            if "diabetes_type2" in conds:
                hba1c = round(np.clip(6.5 + adherence_factor * 0.8 + year * 0.1 * adherence_factor + random.gauss(0, 0.4), 4.5, 14.0), 1)
                labs.append({"lab_id": str(uuid.uuid4()), "patient_id": pid, "lab_date": lab_date.isoformat(),
                             "test_name": "hba1c", "value": hba1c, "unit": "%",
                             "reference_low": 4.0, "reference_high": 5.6, "flag": "high" if hba1c > 7.0 else "normal"})

                glucose = round(np.clip(100 + adherence_factor * 40 + random.gauss(0, 20), 60, 400), 0)
                labs.append({"lab_id": str(uuid.uuid4()), "patient_id": pid, "lab_date": lab_date.isoformat(),
                             "test_name": "fasting_glucose", "value": glucose, "unit": "mg/dL",
                             "reference_low": 70, "reference_high": 100, "flag": "high" if glucose > 126 else "normal"})

            if "ckd" in conds:
                creatinine = round(np.clip(1.2 + adherence_factor * 0.5 + year * 0.05 + random.gauss(0, 0.2), 0.5, 8.0), 2)
                labs.append({"lab_id": str(uuid.uuid4()), "patient_id": pid, "lab_date": lab_date.isoformat(),
                             "test_name": "creatinine", "value": creatinine, "unit": "mg/dL",
                             "reference_low": 0.7, "reference_high": 1.3, "flag": "high" if creatinine > 1.3 else "normal"})

            if "hypertension" in conds or random.random() < 0.3:
                ldl = round(np.clip(110 + adherence_factor * 25 + random.gauss(0, 15), 40, 250), 0)
                labs.append({"lab_id": str(uuid.uuid4()), "patient_id": pid, "lab_date": lab_date.isoformat(),
                             "test_name": "ldl_cholesterol", "value": ldl, "unit": "mg/dL",
                             "reference_low": 0, "reference_high": 100, "flag": "high" if ldl > 130 else "normal"})

            # CBC for everyone periodically
            if random.random() < 0.4:
                wbc = round(np.clip(random.gauss(7.5, 2.0), 2.0, 20.0), 1)
                labs.append({"lab_id": str(uuid.uuid4()), "patient_id": pid, "lab_date": lab_date.isoformat(),
                             "test_name": "wbc", "value": wbc, "unit": "K/uL",
                             "reference_low": 4.5, "reference_high": 11.0,
                             "flag": "high" if wbc > 11.0 else ("low" if wbc < 4.5 else "normal")})
    return labs


def _gen_vitals(patients, diagnoses, archetypes_map):
    vitals = []
    today = date.today()
    dx_by_patient = {}
    for d in diagnoses:
        dx_by_patient.setdefault(d["patient_id"], []).append(d["condition"])

    for p in patients:
        pid = p["patient_id"]
        conds = set(dx_by_patient.get(pid, []))
        arch = archetypes_map.get(pid, "moderate")
        adh_f = {"excellent": 0, "good": 0.3, "moderate": 0.7, "poor": 1.2, "erratic": 0.9}.get(arch, 0.7)
        has_htn = "hypertension" in conds

        # Monthly vitals over 5 years
        for months_ago in range(0, YEARS_OF_DATA * 12):
            v_date = today - timedelta(days=months_ago * 30 + random.randint(-5, 5))
            if v_date > today:
                continue
            sbp = round(np.clip((135 if has_htn else 118) + adh_f * 10 + random.gauss(0, 8), 85, 210))
            dbp = round(np.clip((85 if has_htn else 75) + adh_f * 5 + random.gauss(0, 5), 50, 130))
            hr = round(np.clip(72 + random.gauss(0, 8), 45, 130))
            temp = round(np.clip(98.6 + random.gauss(0, 0.3), 96.0, 103.0), 1)
            spo2 = round(np.clip(97 - ("copd" in conds) * 3 + random.gauss(0, 1.5), 85, 100))

            vitals.append({
                "vital_id": str(uuid.uuid4()),
                "patient_id": pid,
                "vital_date": v_date.isoformat(),
                "systolic_bp": sbp, "diastolic_bp": dbp,
                "heart_rate": hr, "temperature": temp, "spo2": spo2,
                "weight_lbs": round(np.clip(p["bmi"] * 2.8 + random.gauss(0, 5), 90, 400), 1),
            })
    return vitals


def _gen_encounters(patients, diagnoses, archetypes_map):
    encounters = []
    today = date.today()
    dx_by_patient = {}
    for d in diagnoses:
        dx_by_patient.setdefault(d["patient_id"], []).append(d)

    for p in patients:
        pid = p["patient_id"]
        arch = archetypes_map.get(pid, "moderate")
        er_base = {"excellent": 0.02, "good": 0.04, "moderate": 0.08, "poor": 0.15, "erratic": 0.10}.get(arch, 0.08)
        n_dx = len(dx_by_patient.get(pid, []))

        for months_ago in range(0, YEARS_OF_DATA * 12):
            # Routine visit
            if random.random() < 0.3:
                enc_date = today - timedelta(days=months_ago * 30 + random.randint(0, 29))
                if enc_date > today:
                    continue
                encounters.append({
                    "encounter_id": str(uuid.uuid4()),
                    "patient_id": pid,
                    "encounter_date": enc_date.isoformat(),
                    "encounter_type": random.choices(["outpatient","telehealth"], weights=[0.7,0.3])[0],
                    "los_days": 0,
                    "readmission_flag": False,
                    "chief_complaint": random.choice(["routine_followup","medication_review","lab_review","symptom_check"]),
                    "disposition": "home",
                })

            # ER visit probability increases with poor adherence and more conditions
            er_prob = er_base * (1 + n_dx * 0.1)
            if random.random() < er_prob:
                enc_date = today - timedelta(days=months_ago * 30 + random.randint(0, 29))
                if enc_date > today:
                    continue
                los = random.choices([0, 1, 2, 3, 5, 7, 14], weights=[0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02])[0]
                encounters.append({
                    "encounter_id": str(uuid.uuid4()),
                    "patient_id": pid,
                    "encounter_date": enc_date.isoformat(),
                    "encounter_type": "er" if los <= 1 else "inpatient",
                    "los_days": los,
                    "readmission_flag": random.random() < 0.12,
                    "chief_complaint": random.choice(["chest_pain","shortness_of_breath","hyperglycemia","fall","infection","acute_exacerbation"]),
                    "disposition": "admitted" if los > 0 else "discharged",
                })
    return encounters


def _write_to_sqlite(db_path, patients, diagnoses, medications, adherence_events,
                     pharmacy_refills, labs, vitals, encounters, archetypes_map):
    """Write all data to SQLite with proper indexes for fast queries."""
    if db_path.exists():
        os.remove(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print("  Writing to SQLite...")

    # --- Patients ---
    conn.execute("""CREATE TABLE patients (
        patient_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, gender TEXT,
        date_of_birth TEXT, age INTEGER, race TEXT, blood_type TEXT, bmi REAL,
        insurance_type TEXT, zipcode TEXT, sdoh_risk_factors TEXT, n_sdoh_risks INTEGER,
        adherence_archetype TEXT, created_at TEXT
    )""")
    conn.executemany("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(p["patient_id"],p["first_name"],p["last_name"],p["gender"],p["date_of_birth"],
          p["age"],p["race"],p["blood_type"],p["bmi"],p["insurance_type"],p["zipcode"],
          p["sdoh_risk_factors"],p["n_sdoh_risks"],p["adherence_archetype"],p["created_at"])
         for p in patients])
    conn.execute("CREATE INDEX idx_patients_name ON patients(last_name, first_name)")
    conn.execute("CREATE INDEX idx_patients_age ON patients(age)")
    conn.execute("CREATE INDEX idx_patients_race ON patients(race)")
    conn.execute("CREATE INDEX idx_patients_archetype ON patients(adherence_archetype)")

    # --- Diagnoses ---
    conn.execute("""CREATE TABLE diagnoses (
        diagnosis_id TEXT PRIMARY KEY, patient_id TEXT, condition TEXT, icd10_code TEXT,
        diagnosed_date TEXT, status TEXT, severity TEXT
    )""")
    conn.executemany("INSERT INTO diagnoses VALUES (?,?,?,?,?,?,?)",
        [(d["diagnosis_id"],d["patient_id"],d["condition"],d["icd10_code"],
          d["diagnosed_date"],d["status"],d["severity"]) for d in diagnoses])
    conn.execute("CREATE INDEX idx_dx_patient ON diagnoses(patient_id)")
    conn.execute("CREATE INDEX idx_dx_condition ON diagnoses(condition)")

    # --- Medications ---
    conn.execute("""CREATE TABLE medications (
        medication_id TEXT PRIMARY KEY, patient_id TEXT, medication_name TEXT, condition TEXT,
        dosage TEXT, frequency TEXT, prescribed_date TEXT, status TEXT,
        reminder_time_1 TEXT, reminder_time_2 TEXT
    )""")
    conn.executemany("INSERT INTO medications VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(m["medication_id"],m["patient_id"],m["medication_name"],m["condition"],
          m["dosage"],m["frequency"],m["prescribed_date"],m["status"],
          m["reminder_time_1"],m.get("reminder_time_2")) for m in medications])
    conn.execute("CREATE INDEX idx_meds_patient ON medications(patient_id)")

    # --- Adherence Events (bulk insert in chunks) ---
    conn.execute("""CREATE TABLE adherence_events (
        event_id TEXT PRIMARY KEY, patient_id TEXT, medication_id TEXT, medication_name TEXT,
        event_date TEXT, scheduled_time TEXT, reminder_15min_sent INTEGER,
        reminder_5min_sent INTEGER, taken INTEGER, taken_time TEXT,
        response_latency_min INTEGER, self_reported INTEGER, source TEXT
    )""")
    CHUNK = 10000
    for i in range(0, len(adherence_events), CHUNK):
        chunk = adherence_events[i:i+CHUNK]
        conn.executemany("INSERT INTO adherence_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(e["event_id"],e["patient_id"],e["medication_id"],e["medication_name"],
              e["event_date"],e["scheduled_time"],int(e["reminder_15min_sent"]),
              int(e["reminder_5min_sent"]),int(e["taken"]),e["taken_time"],
              e["response_latency_min"],int(e["self_reported"]),e["source"]) for e in chunk])
        if (i + CHUNK) % 100000 == 0:
            print(f"    adherence_events: {min(i+CHUNK, len(adherence_events))}/{len(adherence_events)}")
    conn.execute("CREATE INDEX idx_adh_patient ON adherence_events(patient_id)")
    conn.execute("CREATE INDEX idx_adh_date ON adherence_events(event_date)")
    conn.execute("CREATE INDEX idx_adh_patient_date ON adherence_events(patient_id, event_date)")

    # --- Pharmacy ---
    conn.execute("""CREATE TABLE pharmacy_refills (
        refill_id TEXT PRIMARY KEY, patient_id TEXT, medication_id TEXT, medication_name TEXT,
        fill_date TEXT, supply_days INTEGER, expected_gap_days INTEGER,
        actual_gap_days INTEGER, pharmacy_name TEXT
    )""")
    conn.executemany("INSERT INTO pharmacy_refills VALUES (?,?,?,?,?,?,?,?,?)",
        [(r["refill_id"],r["patient_id"],r["medication_id"],r["medication_name"],
          r["fill_date"],r["supply_days"],r["expected_gap_days"],r["actual_gap_days"],
          r["pharmacy_name"]) for r in pharmacy_refills])
    conn.execute("CREATE INDEX idx_pharm_patient ON pharmacy_refills(patient_id)")

    # --- Labs ---
    conn.execute("""CREATE TABLE lab_results (
        lab_id TEXT PRIMARY KEY, patient_id TEXT, lab_date TEXT, test_name TEXT,
        value REAL, unit TEXT, reference_low REAL, reference_high REAL, flag TEXT
    )""")
    for i in range(0, len(labs), CHUNK):
        chunk = labs[i:i+CHUNK]
        conn.executemany("INSERT INTO lab_results VALUES (?,?,?,?,?,?,?,?,?)",
            [(l["lab_id"],l["patient_id"],l["lab_date"],l["test_name"],
              l["value"],l["unit"],l["reference_low"],l["reference_high"],l["flag"]) for l in chunk])
    conn.execute("CREATE INDEX idx_labs_patient ON lab_results(patient_id)")
    conn.execute("CREATE INDEX idx_labs_date ON lab_results(patient_id, lab_date)")

    # --- Vitals ---
    conn.execute("""CREATE TABLE vitals (
        vital_id TEXT PRIMARY KEY, patient_id TEXT, vital_date TEXT,
        systolic_bp INTEGER, diastolic_bp INTEGER, heart_rate INTEGER,
        temperature REAL, spo2 INTEGER, weight_lbs REAL
    )""")
    for i in range(0, len(vitals), CHUNK):
        chunk = vitals[i:i+CHUNK]
        conn.executemany("INSERT INTO vitals VALUES (?,?,?,?,?,?,?,?,?)",
            [(v["vital_id"],v["patient_id"],v["vital_date"],v["systolic_bp"],v["diastolic_bp"],
              v["heart_rate"],v["temperature"],v["spo2"],v["weight_lbs"]) for v in chunk])
    conn.execute("CREATE INDEX idx_vitals_patient ON vitals(patient_id)")
    conn.execute("CREATE INDEX idx_vitals_date ON vitals(patient_id, vital_date)")

    # --- Encounters ---
    conn.execute("""CREATE TABLE encounters (
        encounter_id TEXT PRIMARY KEY, patient_id TEXT, encounter_date TEXT,
        encounter_type TEXT, los_days INTEGER, readmission_flag INTEGER,
        chief_complaint TEXT, disposition TEXT
    )""")
    conn.executemany("INSERT INTO encounters VALUES (?,?,?,?,?,?,?,?)",
        [(e["encounter_id"],e["patient_id"],e["encounter_date"],e["encounter_type"],
          e["los_days"],int(e["readmission_flag"]),e["chief_complaint"],e["disposition"]) for e in encounters])
    conn.execute("CREATE INDEX idx_enc_patient ON encounters(patient_id)")
    conn.execute("CREATE INDEX idx_enc_type ON encounters(encounter_type)")

    # --- Metadata ---
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("generated_at", datetime.now().isoformat()))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("num_patients", str(len(patients))))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("years_of_data", str(YEARS_OF_DATA)))

    conn.commit()

    # Also save archetypes as JSON for ML training
    with open(DATA_DIR / "archetypes.json", "w") as f:
        json.dump(archetypes_map, f)

    # Stats
    cursor = conn.cursor()
    for table in ["patients","diagnoses","medications","adherence_events","pharmacy_refills","lab_results","vitals","encounters"]:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"    {table}: {count:,} rows")

    db_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Database size: {db_size:.1f} MB")
    conn.close()


def main():
    DATA_DIR.mkdir(exist_ok=True)
    start = time.time()
    print(f"Generating {NUM_PATIENTS:,} patients with {YEARS_OF_DATA} years of data...")

    print("  Generating patients...")
    patients = _gen_patients(NUM_PATIENTS)

    print("  Generating diagnoses...")
    diagnoses = _gen_diagnoses(patients)

    print("  Generating medications...")
    medications = _gen_medications(patients, diagnoses)

    print("  Generating adherence events (this takes a while)...")
    adherence, archetypes = _gen_adherence(patients, medications)

    print("  Generating pharmacy refills...")
    pharmacy = _gen_pharmacy(patients, medications, archetypes)

    print("  Generating lab results...")
    labs = _gen_labs(patients, diagnoses, archetypes)

    print("  Generating vitals...")
    vitals = _gen_vitals(patients, diagnoses, archetypes)

    print("  Generating encounters...")
    encounters = _gen_encounters(patients, diagnoses, archetypes)

    _write_to_sqlite(DB_PATH, patients, diagnoses, medications, adherence,
                     pharmacy, labs, vitals, encounters, archetypes)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Database: {DB_PATH}")


if __name__ == "__main__":
    main()
