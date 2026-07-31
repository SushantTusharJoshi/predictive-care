# PredictiveCare

An AI-powered clinical risk prediction platform that predicts patient ER visit risk, care needs, and medication adherence failure — before they happen. Built to the architectural standard of enterprise EHR platforms, on an open stack.

**Stack:** FastAPI · PostgreSQL 16 · XGBoost/LightGBM · SHAP · Groq AI · Next.js 15

---

## What No Other Platform Does

Commercial EHR platforms (Epic, Cerner, Athena) surface risk scores. They do not explain them, act on them proactively, or learn from clinician disagreement. PredictiveCare closes that loop end-to-end:

| Capability | Epic / Cerner | PredictiveCare |
|---|---|---|
| Risk prediction | ✓ (black box) | ✓ Ensemble ML with SHAP explainability per patient |
| Narrative explanation | ✗ | ✓ Groq LLM generates plain-English risk summaries |
| Medication trust scoring | ✗ | ✓ Per-patient-per-medication behavioral trust score |
| Proactive tiered reminders | ✗ | ✓ 15-min / 5-min / dose-due / 10-min-missed cycles |
| Human-in-the-loop gate | ✗ | ✓ No autonomous scheduling; clinician approval required |
| Clinician feedback loop | ✗ | ✓ Overrides and ratings feed model retraining pipeline |
| HIPAA de-identification before LLM | ✗ | ✓ PHI stripped before any Groq API call |
| Bias audit | ✗ | ✓ Archetype distribution fairness monitoring |
| Open deployable stack | ✗ (vendor lock-in) | ✓ FastAPI + PostgreSQL + Railway/Vercel |

The core insight: **risk prediction is useless without trust-weighted action**. If a patient has a low adherence trust score, a reminder alone won't work — it needs escalation. PredictiveCare is the only open platform that chains prediction → trust scoring → tiered intervention → clinician gate into a single pipeline.

---

## ML Architecture

### Ensemble Models
- **XGBoost + LightGBM** ensemble trained on 50K Synthea-based synthetic patient cohorts
- **True temporal train/test split**: oldest 80% of patients train the model, newest 20% are held out for evaluation — no future data leaks into training (an earlier version used random splits and hit AUC=1.0 due to data leakage; this was caught and corrected)

### Prediction Horizons (Temporal Split, 50K Patients)
| Model | AUC | Precision | Recall | F1 | Algorithm | Notes |
|---|---|---|---|---|---|---|
| ER Visit (30d) | 0.74 | 0.49 | 0.64 | 0.56 | LightGBM | Primary triage signal |
| Care Need (90d) | 0.74 | 0.69 | 0.66 | 0.68 | LightGBM | Care coordination planning |
| Care Need (30d) | 0.67 | 0.21 | 0.50 | 0.30 | LightGBM | Short-horizon intervention |
| Adherence Trust Score | — | — | — | — | Behavioral | Per-patient, not predictive |

> **Note**: Metrics above are from `train_standalone.py` on synthetic data with a true temporal split. Real clinical data would likely yield higher AUC (0.75–0.85) due to richer feature signals.

### Top Predictive Features (by importance)
1. BMI
2. Heart rate (avg)
3. Response latency (avg time to take medication)
4. Systolic blood pressure (avg)
5. Adherence rate (90-day)

### SHAP Explainability
Every prediction surfaces per-patient feature attribution via SHAP values. The Groq LLM (`llama-3.1-70b-versatile`) converts SHAP output into plain-English clinical narratives — after PHI de-identification.

---

## Data Scale

- **215M+ adherence events** generated across synthetic patient cohorts
- PostgreSQL 16 with async SQLAlchemy ORM and indexed lookups
- Risk scores pre-computed at startup; no per-request ML inference on patient lists (eliminates timeout risk)

---

## HIPAA Compliance Architecture

- Every PHI access audit-logged with IP, user, patient_id, and timestamp
- De-identification pipeline strips names, DOB, ZIP, and identifiers before any LLM call
- RBAC with 5 roles: Admin, Physician, Nurse, Care Coordinator, Pharmacist
- JWT authentication on all endpoints

---

## Trust Score + Reminder Engine

The trust score is a per-patient-per-medication behavioral signal computed from:
- Adherence event history (taken / missed / late)
- Pharmacy refill alignment (refill before or after depletion)
- Streak patterns and dose timing variance

Trust score drives reminder dispatch tier:
- **High trust** → standard 15-min and 5-min pre-dose reminders
- **Low trust** → escalated to clinician flag + 10-min post-miss alert

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI 0.115, Python 3.12, SQLAlchemy async |
| Database | PostgreSQL 16 (Homebrew / Docker) |
| ML | XGBoost, LightGBM, scikit-learn, SHAP |
| LLM | Groq API (llama-3.1-70b-versatile) |
| Frontend | Next.js 15, React 19, Recharts, Tailwind CSS |
| Auth | JWT RBAC |
| CI/CD | GitHub Actions, weekly model retraining |
| Deploy | Railway (backend) + Vercel (frontend) |

---

## Known Gaps (Planned Fixes)

These are honest architectural limitations, not minor bugs:

| Gap | Impact | Planned Fix |
|---|---|---|
| Frontend hydration error | Role read from `localStorage` during SSR causes client/server mismatch | Client-side guard with `useEffect` before reading localStorage |
| HITL workflow frontend incomplete | Backend HITL endpoints exist; UI not wired up | Build clinician approval queue page |
| Bias audit visualization missing | Admin page has no fairness chart | Add Recharts breakdown by race/ethnicity/archetype |
| Search by patient UUID not implemented | Can only search by name | Add UUID search path to `/search` endpoint and frontend |
| ORM-based bulk insert is slow | 150–200M rows via SQLAlchemy ORM takes 10+ hours | Replace with raw SQL `COPY` or `execute_many` bulk inserts |
| No real patient data pipeline | Currently synthetic only | Integrate real Synthea FHIR export; add cloud BAA for PHI |
| No encryption at rest | Database not encrypted | Add PostgreSQL TDE or cloud KMS |
| Auth is JWT only | No SSO / enterprise login | Add OAuth2/OIDC for hospital system integration |

---

## Local Setup

```bash
# 1. Start PostgreSQL
/opt/homebrew/opt/postgresql@16/bin/pg_ctl -D /opt/homebrew/var/postgresql@16 start

# 2. Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in DB credentials and GROQ_API_KEY
uvicorn app.main_pg:app --reload --port 8000

# 3. Frontend
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Default Logins
| Username | Password | Role |
|---|---|---|
| admin | admin | Admin |
| physician | physician | Physician |
| nurse | nurse | Nurse |
| coordinator | coordinator | Care Coordinator |
| dr.patel | dr.patel | Physician |
| rn.williams | rn.williams | Nurse |

---

## Market Context

The clinical AI market is dominated by Epic's proprietary models and point solutions (Sepsis prediction tools, readmission risk models) that are siloed, unexplainable, and inaccessible to smaller health systems. PredictiveCare is positioned as the open alternative: a full-stack prediction + intervention + feedback platform that any health system can self-host, audit, and extend — without vendor lock-in or black-box risk scores.

---

## GitHub

[github.com/SushantTusharJoshi/predictive-care](https://github.com/SushantTusharJoshi/predictive-care)
