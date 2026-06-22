#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║   PredictiveCare v3.1 — Quick Setup          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

command -v docker >/dev/null 2>&1 || { echo "❌ Docker required: https://docs.docker.com/get-docker/"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.12+ required."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js 20+ required."; exit 1; }
echo "✅ Prerequisites found"

echo ""
echo "📦 Step 1/5: Starting PostgreSQL..."
docker compose up -d db
sleep 5
until docker compose exec db pg_isready -U pc_user -d predictive_care > /dev/null 2>&1; do sleep 2; done
echo "   ✅ PostgreSQL ready"

echo ""
echo "🐍 Step 2/5: Installing backend deps..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements-pg.txt
echo "   ✅ Backend deps installed"

echo ""
echo "📊 Step 3/5: Creating tables & generating 100K patients..."
python -c "
from app.data.database_pg import get_sync_engine
from app.data.models_pg import Base
Base.metadata.create_all(get_sync_engine())
print('   Tables created.')
"
python -m app.data.ingest_synthea --fallback --max-patients 100000
echo "   ✅ 100K patients loaded with adherence, pharmacy, and trust scores"

echo ""
echo "🤖 Step 4/5: Training ML models..."
python -m app.ml.train_pg
echo "   ✅ Models trained"
cd ..

echo ""
echo "🎨 Step 5/5: Installing frontend deps..."
cd frontend && npm install --silent && cd ..
echo "   ✅ Frontend deps installed"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅ Setup complete!                          ║"
echo "║                                              ║"
echo "║   Start: make dev                            ║"
echo "║   Login: admin/admin                         ║"
echo "║   Backend:  http://localhost:8000             ║"
echo "║   Frontend: http://localhost:3000             ║"
echo "║                                              ║"
echo "║   HIPAA Data Flow: GET /hipaa/data-flow      ║"
echo "╚══════════════════════════════════════════════╝"
