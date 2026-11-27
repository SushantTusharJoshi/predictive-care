"""Compatibility alias: the app lives in app.main.

Kept only so a host still started with `uvicorn app.main_pg:app` keeps working.
Remove once every start command uses app.main:app.
"""
from app.main import app  # noqa: F401
