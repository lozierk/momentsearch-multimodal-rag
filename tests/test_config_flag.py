"""/api/config must tell the UI whether ingest is open.

On a locked deploy (ADMIN_TOKEN set) every ingest route 401s without a bearer
the browser UI doesn't carry. The UI reads `ingest_open` to render read-only
instead of surfacing raw 401s — these tests pin the flag to the token state.

Unit-level like the rest of the suite: resolve_llm (DB) and presign_capable
(storage) are stubbed out.
"""
from __future__ import annotations

import pytest

from src import config
from src.api import search as search_api


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(search_api.rag_search, "resolve_llm",
                        lambda uid: (None, "none"))
    monkeypatch.setattr(search_api.storage, "presign_capable", lambda: False)


def test_ingest_open_when_no_admin_token(stubbed, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    assert search_api.get_config(x_user_id=None)["ingest_open"] is True


def test_ingest_locked_when_admin_token_set(stubbed, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "sekrit")
    assert search_api.get_config(x_user_id=None)["ingest_open"] is False
