"""PredictiveCare v3.1 — main entry point.
Re-exports from main_pg for backwards compatibility.
Use: uvicorn app.main_pg:app --reload --port 8000
"""
from app.main_pg import app  # noqa: F401
