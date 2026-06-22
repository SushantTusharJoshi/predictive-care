# PredictiveCare v3.1

AI-powered healthcare risk prediction platform with HIPAA compliance, medication adherence monitoring, trust scoring, and HITL clinical workflows.

## What's New in v3.1

- **PostgreSQL 16** — Full migration from SQLite to async PostgreSQL with SQLAlchemy
- **Synthea Integration** — Ingest real Synthea patient data or generate 100K synthetic patients
- **Adherence Trust Score** (FR-3.2.3) — Per-patient-medication behavioral trust scoring
- **Tiered Medication Reminders** (FR-3.2.1) — 15-min, 5-min, and 10-min post-dose alerts
- **Pharmacy Refill Tracking** — Refill alignment feeds trust score computation
- **HIPAA Middleware** (NFR-4.3.1) — Every PHI access audit-logged with IP, user, patient_id
- **HITL Scheduling Gate** (FR-3.3.3) — No autonomous scheduling; clinician approval required
- **Clinician Feedback Loop** (FR-3.4.3) — Rate predictions for model retraining
- **Prediction Override** (FR-3.4.2) — Override model output with documented reasoning
- **HIPAA Data Flow Endpoint** — `/hipaa/data-flow` for compliance audits
- **Frontend Hydration Fix** — No more SSR/localStorage mismatch errors

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI 0.115, Python 3.12+, SQLAlchemy async |
| Database | PostgreSQL 16 (Homebrew or Docker) |
| ML | XGBoost + LightGBM ensemble, SHAP, scikit-learn |
| LLM | Groq API (llama-3.1-70b-versatile) |
| Frontend | Next.js 15, React 19, Recharts, Tailwind CSS |
| Deploy | Railway (backend) + Vercel (frontend) |
| Security | JWT RBAC, HIPAA audit logging, de-identification |

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Start PostgreSQL + Backend
docker compose up -d db
sleep 5

# Install dependencies
cd backend && pip install -r requirements-pg.txt
cd ../frontend && npm install && cd ..

# Generate 100K synthetic patients with Synthea-realistic data
cd backend && python -m app.data.ingest_synthea --fallback --max-patients 100000

# Train ML models
python -m app.ml.train_pg

# Start backend
uvicorn app.main_pg:app --reload --port 8000 &

# Start frontend
cd ../frontend && npm run dev
```

### Option 2: Local PostgreSQL (macOS)

```bash
# If using Homebrew PostgreSQL
/opt/homebrew/opt/postgresql@16/bin/psql -U postgres -c "CREATE USER pc_user WITH PASSWORD 'pc_local_dev_2024';"
/opt/homebrew/opt/postgresql@16/bin/psql -U postgres -c "CREATE DATABASE predictive_care OWNER pc_user;"

# Then follow the same steps as above (skip docker compose)
```

### Option 3: One-Command Setup

```bash
make setup    # DB + deps + tables
make ingest-fallback  # 100K patients
make train    # ML models
make dev      # Start everything
```

Open http://localhost:3000

## Login Credentials

| Username | Password | Role |
|----------|----------|------|
| admin | admin | Admin (full access) |
| physician | physician | Physician (patients + predictions + HITL) |
| nurse | nurse | Nurse (patients + adherence + reminders) |
| coordinator | coordinator | Coordinator (patients + scheduling) |
| dr.patel | dr.patel | Physician |
| rn.williams | rn.williams | Nurse |

## API Endpoints

### Core
- `GET /health` — Health check
- `POST /auth/login` — Authenticate
- `GET /patients` — Paginated patient list with risk scores
- `GET /patients/{id}` — Full patient detail + predictions
- `GET /search?q=` — Quick search by name or UUID

### Predictions & Explainability
- `POST /predict/{id}` — On-demand prediction for one patient
- `GET /patients/{id}/shap-narrative/{type}` — AI-generated SHAP explanation

### Adherence & Reminders
- `GET /patients/{id}/reminders` — Medication reminder schedule with trust scores
- `GET /patients/{id}/adherence-trend` — 30-day adherence data
- `GET /patients/{id}/longitudinal` — 5-year behavior analysis

### HITL (Human-in-the-Loop)
- `POST /patients/{id}/schedule-recommendation` — Generate scheduling recommendation
- `POST /schedule/{rec_id}/action` — Approve/dismiss recommendation
- `POST /patients/{id}/feedback` — Rate prediction accuracy
- `POST /patients/{id}/override` — Override prediction with reason

### Admin & Compliance
- `GET /stats` — Dashboard statistics
- `GET /audit` — HIPAA audit log (admin only)
- `GET /models/metrics` — ML model performance
- `GET /hipaa/data-flow` — HIPAA compliance documentation

## HIPAA Data Flow

```
Request → JWT Auth → RBAC Check → HIPAA Audit Middleware
    → De-identify PHI (if LLM call) → Process → Audit Log → Response
```

Every PHI access is logged to the `audit_log` table with: timestamp, username, role,
action, patient_id_accessed, IP address, and user agent.

## Deployment

### Backend → Railway

```bash
cd backend && railway up
railway variables set DATABASE_URL=...
railway variables set JWT_SECRET=...
railway variables set GROQ_API_KEY=...
```

### Frontend → Vercel

```bash
cd frontend && vercel --prod
vercel env add NEXT_PUBLIC_API_BASE  # → https://your-backend.railway.app
```

## Model Architecture

XGBoost + LightGBM ensemble trained on:
- Demographics (age, gender, BMI, SDOH risks)
- Adherence patterns (90-day rate, trend, response latency)
- Encounter history (ER visits, inpatient stays)
- Pharmacy data (refill gaps, alignment)
- Archetype classification (excellent/good/moderate/poor/erratic)

Targets: ER visit (30d), Care need (90d)
