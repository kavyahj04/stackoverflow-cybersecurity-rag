"""Shared test setup. Unit tests use fakes: no OpenAI calls and no database.

Tests marked `db` talk to the real database and only run with RUN_DB_TESTS=1.
"""
import os

import pytest

# Settings require a database address; unit tests never connect, so any value will do
os.environ.setdefault("DATABASE_URL", "postgresql://unit-tests-do-not-connect")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_DB_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="needs the real database: run with RUN_DB_TESTS=1")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)
