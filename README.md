# PredictiveCare

A clinical risk prediction platform that predicts ER visit risk, care needs and medication adherence problems, explains each prediction, and keeps a clinician in the loop.

> Synthetic data only. Not clinically validated and not for clinical use.

**Stack:** FastAPI · PostgreSQL 16 · XGBoost/LightGBM · SHAP · Groq AI · Next.js 15

---

## System Architecture

![PredictiveCare architecture: Next.js frontend calls the FastAPI backend (auth, HIPAA middleware and audit log, trust score, reminders, Groq narratives), which reads and writes PostgreSQL and the ML training and similar-patients modules](docs/architecture.webp)

---

## What It Adds

- **Explained predictions:** ensemble ML with SHAP attributions for every patient.
- **Plain-English narratives:** an LLM turns de-identified SHAP output into a short summary, with a template fallback when the LLM is unavailable.
- **Medication trust scoring:** a per-patient, per-medication behavioural signal that sets the reminder tier.
- **Tiered reminders:** 15-minute, 5-minute, dose-due and 10-minutes-missed cycles.
- **Clinician gate:** nothing is scheduled automatically; recommendations need clinician approval.
- **De-identification before any LLM call:** names, birth date, address, ZIP and identifiers are stripped first.
- **Open, deployable stack:** FastAPI, PostgreSQL, Railway and Vercel.

---

## ML Architecture

### Ensemble Models
- **XGBoost + LightGBM** ensemble. The metrics below come from `train_standalone.py`, which generates its own 50,000 synthetic patients in memory; they are not from the 100K-patient Postgres cohort. A retrain from Postgres, with a calibration slice and a logistic regression baseline, is in progress
- **True temporal train/test split**: oldest 80% of patients train the model, newest 20% are held out for evaluation — no future data leaks into training (an earlier version used random splits and hit AUC=1.0 due to data leakage; this was caught and corrected)

### Prediction Horizons (Temporal Split, 50K In-Memory Synthetic Patients)
| Model | AUC | Precision | Recall | F1 | Algorithm | Notes |
|---|---|---|---|---|---|---|
| ER Visit (30d) | 0.74 | 0.49 | 0.64 | 0.56 | LightGBM | Primary triage signal |
| Care Need (90d) | 0.74 | 0.69 | 0.66 | 0.68 | LightGBM | Care coordination planning |
| Care Need (30d) | 0.67 | 0.21 | 0.50 | 0.30 | LightGBM | Short-horizon intervention |
| Adherence Trust Score | — | — | — | — | Behavioral | Per-patient, not predictive |

> **Note**: Metrics above are from `train_standalone.py` on in-memory synthetic data with a true temporal split. They say nothing about performance on real patients.

### Top Predictive Features (by importance)
1. BMI
2. Heart rate (avg)
3. Response latency (avg time to take medication)
4. Systolic blood pressure (avg)
5. Adherence rate (90-day)

### SHAP Explainability
Every prediction surfaces per-patient feature attribution via SHAP values. The Groq-hosted LLM (default `openai/gpt-oss-120b`, set with `GROQ_MODEL`) converts SHAP output into plain-English clinical narratives — after PHI de-identification.

---

## Data Scale

- **100K synthetic patients** with 5 years of longitudinal data, in the local development database
- **215M+ adherence events** generated across that cohort
- The hosted demo runs on a small synthetic subset
- PostgreSQL 16 with async SQLAlchemy ORM and indexed lookups
- Risk scores pre-computed at startup; no per-request ML inference on patient lists (eliminates timeout risk)

---

## Privacy Controls (HIPAA-aligned)

These are design controls, not a compliance claim: a real deployment would also need a business associate agreement, a risk assessment and written policies.

- Every read of patient data writes an audit-log row (user, role, patient, time); [a test](backend/tests/test_audit_coverage.py) fails if a route forgets
- De-identification strips names, date of birth, address, ZIP and identifiers before any LLM call
- RBAC with 4 roles: Admin, Physician, Nurse, Care Coordinator
- JWT authentication on every patient endpoint

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
| LLM | Groq API (`openai/gpt-oss-120b` by default, `GROQ_MODEL` to change) |
| Frontend | Next.js 15, React 19, Recharts, Tailwind CSS |
| Auth | JWT RBAC |
| CI/CD | GitHub Actions (lint, tests) |
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

### Demo Logins
Demo accounts use the username as the password and exist only when demo logins are on. They are on by default outside production. In production (`ENVIRONMENT=production`) they are off, and the backend refuses to start while any password-equals-username account or the default `JWT_SECRET` is active.

- Real accounts: set `AUTH_USERS` to a JSON object of bcrypt hashes, for example `{"dr.kim": {"password_hash": "$2b$12$...", "role": "physician", "name": "Dr. Kim"}}`. Generate a hash with `python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('your-password'))"`.
- Public synthetic-data demo: set `DEMO_LOGINS=true` explicitly. Hide the quick-login buttons with `NEXT_PUBLIC_DEMO_LOGINS=false` on the frontend.

| Username | Password | Role |
|---|---|---|
| admin | admin | Admin |
| physician | physician | Physician |
| nurse | nurse | Nurse |
| coordinator | coordinator | Care Coordinator |
| dr.patel | dr.patel | Physician |
| rn.williams | rn.williams | Nurse |

---

## GitHub

[github.com/SushantTusharJoshi/predictive-care](https://github.com/SushantTusharJoshi/predictive-care)
