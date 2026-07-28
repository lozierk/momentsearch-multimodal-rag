"""Durability reconciler — recovery for the failure mode that raises nothing.

Why this exists: `kill -9` on a worker is not an exception. No `except` runs, no
`finally` runs, Prefect's flow-run row rots in "Running", and OUR row sits in an
inflight status (`queued`/`fetching`/`sampling`/`embedding`) forever — holding a
dispatch slot and never reaching a terminal state. Prefect Cloud has no
dead-letter topic to catch it. Retries don't help either: a task retry needs a
live process to do the retrying.

So recovery has to be **state-driven**, not exception-driven:

  every RECONCILE_INTERVAL_S:
    find rows still inflight whose updated_at is older than RECONCILE_STUCK_S
      attempts <  RECONCILE_MAX_ATTEMPTS -> back to `pending`  (WFQ re-admits it)
      attempts >= RECONCILE_MAX_ATTEMPTS -> `failed`           (poor-man's DLQ)

Two properties make the requeue safe rather than reckless:

1. **Postgres is the source of truth.** Every writer bumps `updated_at`
   (`set_status` per stage, `set_progress` every 25 thumbnails), so a stale
   `updated_at` means no process is working on that row — it is a liveness
   heartbeat, not a start time. A slow-but-alive run keeps bumping and is never
   touched.
2. **Ingestion is idempotent.** Qdrant point ids are `uuid5(f"{id}:{n}")`, so
   re-ingesting the same document upserts over the same points: re-running from
   `pending` costs compute, never correctness, and never duplicates.

Guards: we never touch `pending` rows (the dispatcher owns those) and never
terminal rows — the `status = ANY(INFLIGHT_STATUSES)` predicate covers both. A
row the dispatcher just flipped `pending -> queued` has a fresh `updated_at`,
so the stuck window keeps us off it too.

Runs as a daemon thread in worker.py, alongside the dispatcher. With several
workers each runs one; the atomic `UPDATE ... RETURNING` claim (db.py) means a
row is recovered exactly once.
"""
from __future__ import annotations

import threading
import time

from . import config, db


def verdict(status: str, age_s: float, attempts: int, *,
            stuck_s: float, max_attempts: int) -> str | None:
    """What should happen to a row in `status`, `age_s` seconds since its last
    write, with `attempts` runs already spent? -> "requeue" | "dead" | None.

    This is the decision spec, in Python, so the boundaries are unit-testable
    with no Postgres in the loop; `db.reconcile_stuck` implements exactly this
    predicate in SQL (statuses ∈ INFLIGHT_STATUSES, age > stuck_s, attempts
    below vs at/over the cap).
    """
    if status not in config.INFLIGHT_STATUSES:
        return None            # pending is the dispatcher's; terminal is done
    if age_s <= stuck_s:
        return None            # still within the liveness window — leave it alone
    return "requeue" if attempts < max_attempts else "dead"


def reconcile_once() -> tuple[int, int]:
    """One sweep. Returns (requeued, dead-lettered) counts."""
    requeued, dead = db.reconcile_stuck(config.RECONCILE_STUCK_S,
                                        config.RECONCILE_MAX_ATTEMPTS)
    for row in requeued:
        # attempts counts runs already spent, so the run we just re-armed is
        # the next one.
        print(f"[reconcile] requeued {row['id']} "
              f"(attempt {row['attempts'] + 1}/{config.RECONCILE_MAX_ATTEMPTS})")
    for row in dead:
        print(f"[reconcile] dead-lettered {row['id']} "
              f"({row['attempts']} attempts exhausted)")
    return len(requeued), len(dead)


def run_forever() -> None:
    print(f"[reconcile] durability reconciler on — stuck after "
          f"{config.RECONCILE_STUCK_S:.0f}s, max {config.RECONCILE_MAX_ATTEMPTS} "
          f"attempts, tick {config.RECONCILE_INTERVAL_S:.0f}s")
    while True:
        try:
            reconcile_once()
        except Exception as exc:  # never let the recovery thread die
            print(f"[reconcile] error: {type(exc).__name__}: {exc}")
        time.sleep(config.RECONCILE_INTERVAL_S)


def start_in_background() -> None:
    """Start the reconciler as a daemon thread."""
    threading.Thread(target=run_forever, daemon=True, name="reconciler").start()
