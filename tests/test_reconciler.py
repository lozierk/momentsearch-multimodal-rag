"""Tests for the durability reconciler — the recovery path for a SIGKILL'd
worker, which raises no exception and so leaves rows orphaned mid-flight.

Unit-level only: no Postgres, no Prefect, no network. The decision itself lives
in `reconciler.verdict` (a pure function), so the interesting boundaries —
just-under vs just-over the stuck window, one-below vs at the attempts cap — are
testable directly. `db.reconcile_stuck` mirrors that predicate in SQL and is
tested against a stubbed connection pool, which pins the parts that could
silently drift: which statuses it touches, and which way the attempts
comparisons point.
"""
from __future__ import annotations

import pytest

from src import config, db, reconciler

STUCK_S = 600.0
MAX_ATTEMPTS = 3


def _verdict(status="sampling", age_s=STUCK_S + 1, attempts=0):
    return reconciler.verdict(status, age_s, attempts,
                              stuck_s=STUCK_S, max_attempts=MAX_ATTEMPTS)


# ── The stuck window: liveness, not elapsed time ─────────────────────────────

def test_a_row_inside_the_window_is_left_alone():
    """updated_at is bumped by every stage change and every 25 thumbnails, so
    a recent write means someone is alive and working."""
    assert _verdict(age_s=STUCK_S - 1) is None


def test_exactly_at_the_window_is_still_alive():
    assert _verdict(age_s=STUCK_S) is None


def test_one_second_past_the_window_is_orphaned():
    assert _verdict(age_s=STUCK_S + 1) == "requeue"


@pytest.mark.parametrize("status", config.INFLIGHT_STATUSES)
def test_every_inflight_status_is_recoverable(status):
    assert _verdict(status=status) == "requeue"


# ── The guards: whose rows are these? ────────────────────────────────────────

def test_pending_rows_are_never_touched():
    """`pending` is the WFQ dispatcher's queue. An ancient pending row means the
    dispatcher is at capacity, not that anything died."""
    assert _verdict(status="pending", age_s=99_999) is None


@pytest.mark.parametrize("status", ["indexed", "skipped", "failed"])
def test_terminal_rows_are_never_touched(status):
    assert _verdict(status=status, age_s=99_999) is None


# ── The attempts cap: the poor-man's DLQ boundary ────────────────────────────

def test_one_attempt_below_the_cap_is_requeued():
    assert _verdict(attempts=MAX_ATTEMPTS - 1) == "requeue"


def test_at_the_cap_is_dead_lettered():
    assert _verdict(attempts=MAX_ATTEMPTS) == "dead"


def test_over_the_cap_is_dead_lettered():
    """Belt and braces: a row that somehow overshot must still terminate, not
    loop forever."""
    assert _verdict(attempts=MAX_ATTEMPTS + 1) == "dead"


def test_the_cap_only_applies_once_the_row_is_actually_stuck():
    assert _verdict(age_s=STUCK_S - 1, attempts=MAX_ATTEMPTS) is None


# ── db.reconcile_stuck: thin SQL, pinned so it can't drift from the spec ─────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Records every statement; hands back canned rows in call order."""

    def __init__(self, results):
        self.calls: list[tuple[str, dict]] = []
        self._results = list(results)

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeCursor(self._results.pop(0) if self._results else [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_pool(monkeypatch):
    """Stub the pool at the module boundary (same seam the real code uses)."""
    def install(*results):
        conn = _FakeConn(results)
        monkeypatch.setattr(db, "pool", lambda: type("P", (), {
            "connection": staticmethod(lambda: conn)})())
        return conn
    return install


def test_reconcile_stuck_returns_the_two_groups_separately(fake_pool):
    fake_pool([{"id": "doc_a", "user_id": "u", "attempts": 1}],
              [{"id": "doc_b", "user_id": "u", "attempts": 3}])
    requeued, dead = db.reconcile_stuck(STUCK_S, MAX_ATTEMPTS)
    assert [r["id"] for r in requeued] == ["doc_a"]
    assert [r["id"] for r in dead] == ["doc_b"]


def test_reconcile_stuck_only_ever_targets_inflight_statuses(fake_pool):
    conn = fake_pool([], [])
    db.reconcile_stuck(STUCK_S, MAX_ATTEMPTS)
    assert len(conn.calls) == 2
    for sql, params in conn.calls:
        assert params["statuses"] == list(config.INFLIGHT_STATUSES)
        assert "pending" not in params["statuses"]     # the dispatcher's rows
        assert "status = ANY(%(statuses)s)" in sql
        assert "updated_at < now() - make_interval(secs => %(stuck_s)s)" in sql


def test_reconcile_stuck_splits_on_the_attempts_cap(fake_pool):
    conn = fake_pool([], [])
    db.reconcile_stuck(STUCK_S, MAX_ATTEMPTS)
    requeue_sql, dead_sql = conn.calls[0][0], conn.calls[1][0]
    assert "status = 'pending'" in requeue_sql
    assert "attempts < %(max_attempts)s" in requeue_sql
    assert "status = 'failed'" in dead_sql
    assert "attempts >= %(max_attempts)s" in dead_sql
    assert "attempts exhausted" in dead_sql            # the DLQ error string


def test_reconcile_stuck_passes_its_knobs_through(fake_pool):
    conn = fake_pool([], [])
    db.reconcile_stuck(42, 7)
    for _, params in conn.calls:
        assert params["stuck_s"] == 42.0 and params["max_attempts"] == 7


# ── The loop ─────────────────────────────────────────────────────────────────

def test_reconcile_once_reports_both_counts(monkeypatch, capsys):
    monkeypatch.setattr(db, "reconcile_stuck",
                        lambda *a: ([{"id": "doc_a", "user_id": "u", "attempts": 1}],
                                    [{"id": "doc_b", "user_id": "u", "attempts": 3}]))
    assert reconciler.reconcile_once() == (1, 1)
    out = capsys.readouterr().out
    # attempts=1 spent means the run we just re-armed is attempt 2.
    assert f"requeued doc_a (attempt 2/{config.RECONCILE_MAX_ATTEMPTS})" in out
    assert "dead-lettered doc_b" in out


def test_reconcile_once_uses_the_configured_knobs(monkeypatch):
    seen: list = []
    monkeypatch.setattr(db, "reconcile_stuck",
                        lambda *a: (seen.append(a), ([], []))[1])
    monkeypatch.setattr(config, "RECONCILE_STUCK_S", 60.0)
    monkeypatch.setattr(config, "RECONCILE_MAX_ATTEMPTS", 5)
    reconciler.reconcile_once()
    assert seen == [(60.0, 5)]


# ── Wiring: the reconciler is useless unless the worker starts it ────────────

def test_worker_main_starts_the_reconciler(monkeypatch):
    from src import dispatcher, worker
    from src.rag import vector_store

    order: list[str] = []
    monkeypatch.setattr(worker, "init_schema", lambda: order.append("schema"))
    monkeypatch.setattr(vector_store, "ensure_collection",
                        lambda: order.append("collections"))
    monkeypatch.setattr(dispatcher, "start_in_background",
                        lambda: order.append("dispatcher"))
    monkeypatch.setattr(reconciler, "start_in_background",
                        lambda: order.append("reconciler"))

    stub_flow = type("F", (), {"to_deployment": lambda self, name: name})()
    monkeypatch.setattr(worker, "ingest_video", stub_flow)
    monkeypatch.setattr(worker, "ingest_doc", stub_flow)

    def fake_serve(*deployments, limit=None):
        order.append("serve")
        raise KeyboardInterrupt          # break out of the retry-forever loop

    monkeypatch.setattr(worker, "serve", fake_serve)
    worker.main()

    assert order == ["schema", "collections", "dispatcher", "reconciler", "serve"]
