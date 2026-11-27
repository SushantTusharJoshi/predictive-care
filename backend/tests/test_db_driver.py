"""The sync engine (training, ingest) must build with an installed driver."""
from sqlalchemy import create_engine

from app.data.database_pg import sync_url


def test_bare_postgresql_url_pins_psycopg():
    assert sync_url("postgresql://u@h:5432/db") == "postgresql+psycopg://u@h:5432/db"


def test_heroku_style_postgres_url_pins_psycopg():
    assert sync_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"


def test_explicit_driver_is_kept():
    assert sync_url("postgresql+psycopg2://u@h/db") == "postgresql+psycopg2://u@h/db"


def test_sync_engine_builds_without_connecting():
    engine = create_engine(sync_url("postgresql://u@localhost:1/db"))
    assert engine.dialect.driver == "psycopg"
