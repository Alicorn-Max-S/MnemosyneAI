"""Shared fixtures for Mnemosyne tests."""

import pytest

from mnemosyne.db.sqlite_store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    """Provide an initialized SQLiteStore using a temp directory."""
    db_path = str(tmp_path / "test.db")
    s = SQLiteStore(db_path)
    await s.initialize()
    yield s
    await s.close()
