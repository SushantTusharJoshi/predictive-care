.PHONY: help setup dev down db-shell ingest train test deploy clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup: start DB, install deps, create tables
	docker compose up -d db
	@echo "Waiting for PostgreSQL..."
	@sleep 5
	cd backend && pip install -r requirements-pg.txt
	cd frontend && npm install
	@echo "Creating database tables..."
	cd backend && python -c "from app.data.database_pg import get_sync_engine; from app.data.models_pg import Base; Base.metadata.create_all(get_sync_engine()); print('Done.')"
	@echo "\n✅ Setup complete. Run 'make ingest-fallback' then 'make train' then 'make dev'."

dev: ## Start backend + frontend
	docker compose up -d db
	@sleep 3
	cd backend && uvicorn app.main_pg:app --reload --port 8000 &
	cd frontend && npm run dev &
	@echo "\n🚀 Backend: http://localhost:8000  Frontend: http://localhost:3000"

down: ## Stop all services
	docker compose down
	@pkill -f "uvicorn app.main_pg" 2>/dev/null || true
	@pkill -f "next dev" 2>/dev/null || true

ingest-fallback: ## Generate 100K synthetic patients (no Synthea/Java needed)
	cd backend && python -m app.data.ingest_synthea --fallback --max-patients 100000

ingest-synthea: ## Download Synthea, generate 100K patients, then ingest
	cd backend && curl -sL -o synthea.jar \
		https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
	cd backend && java -jar synthea.jar -p 100000 --exporter.csv.export=true -d ./synthea_output
	cd backend && python -m app.data.ingest_synthea --synthea-dir ./synthea_output/csv

train: ## Train ML models
	cd backend && python -m app.ml.train_pg

db-shell: ## Open psql shell
	docker compose exec db psql -U pc_user -d predictive_care

db-reset: ## Drop and recreate all tables (DESTRUCTIVE)
	docker compose exec db psql -U pc_user -d predictive_care -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	cd backend && python -c "from app.data.database_pg import get_sync_engine; from app.data.models_pg import Base; Base.metadata.create_all(get_sync_engine())"

test: ## Run backend tests
	cd backend && python -m pytest tests/ -v

test-api: ## Quick API smoke test
	@curl -sf http://localhost:8000/health | python -m json.tool

deploy-backend: ## Deploy backend to Railway
	cd backend && railway up

deploy-frontend: ## Deploy frontend to Vercel
	cd frontend && vercel --prod

clean: ## Remove generated files
	docker compose down -v 2>/dev/null || true
	rm -rf backend/data/models/*.pkl backend/data/models/*.json
	rm -rf backend/synthea_output backend/synthea.jar
	rm -rf frontend/.next frontend/node_modules
