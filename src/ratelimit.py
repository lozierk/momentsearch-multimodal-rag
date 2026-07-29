"""Per-IP demo budget for the one public route that spends money (/api/ask).

Retrieval is cheap and stays unlimited; answer synthesis is a multimodal
Anthropic call with up to TOP_K page/frame images attached, so an open demo URL
is an open tap on the key. The budget is deliberately crude — a fixed window
per IP, held in this process's memory — because the failure it guards against
is a script hammering the demo, not a determined attacker.

Shape of one IP's budget (see config.DEMO_BUDGET_*):

    first ask            -> window opens, count = 1
    within WINDOW_S      -> allowed while count < MAX
    cap hit              -> 429 until RESET_S after the window OPENED
    window closed        -> 429 until RESET_S after the window OPENED
    past RESET_S         -> forgiven; the next ask opens a fresh window

The reset is the important part. A permanent lockout would punish exactly the
people this is meant to serve — graders sharing one NAT'd egress IP, or anyone
coming back tomorrow — so every refusal carries an expiry and says so.

Caveats, stated rather than hidden: state is in-process, so a deploy or crash
forgives every budget, and running two api machines doubles the real allowance.
That is an accepted trade for a demo; Redis (or Fly's own edge rate limiting)
is the answer if the number ever has to be exact.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from . import config


@dataclass
class Decision:
    """One verdict. `retry_after_s` is 0 when allowed."""
    allowed: bool
    remaining: int = 0
    retry_after_s: int = 0
    reason: str = ""


def client_ip(headers, peer: str | None) -> str:
    """Who is calling, as conservatively as possible.

    Only when TRUST_PROXY_IP says we genuinely sit behind Fly's edge do we read
    Fly-Client-IP — the edge overwrites that header on every inbound request,
    so it can't be forged from outside. Otherwise the socket peer is the only
    thing we believe. X-Forwarded-For is never read at all: any client can
    append to it, so honouring it would let one caller mint unlimited budgets.
    """
    if config.TRUST_PROXY_IP:
        fly = (headers.get("fly-client-ip") or "").strip()
        if fly:
            return fly
    return (peer or "unknown").strip()


def _human_reset(seconds: int) -> str:
    """'2h' / '45m' / '30s' — the message says when, not just that it failed."""
    if seconds >= 3600:
        h = round(seconds / 3600)
        return f"{h}h" if h != 1 else "1h"
    if seconds >= 60:
        return f"{round(seconds / 60)}m"
    return f"{max(seconds, 1)}s"


class DemoBudget:
    """In-memory per-IP budget. Thread-safe (uvicorn runs handlers in a pool)."""

    def __init__(self, *, max_calls: int | None = None,
                 window_s: int | None = None, reset_s: int | None = None):
        # Read config at construction so tests can build their own instances
        # without mutating module state.
        self.max_calls = config.DEMO_BUDGET_MAX if max_calls is None else max_calls
        self.window_s = config.DEMO_BUDGET_WINDOW_S if window_s is None else window_s
        self.reset_s = config.DEMO_BUDGET_RESET_S if reset_s is None else reset_s
        self._state: dict[str, tuple[float, int]] = {}  # ip -> (window_start, count)
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        """Drop budgets old enough to be forgiven — otherwise the dict is an
        unbounded memory leak keyed by anything that ever sent a request."""
        dead = [ip for ip, (start, _) in self._state.items()
                if now - start >= self.reset_s]
        for ip in dead:
            del self._state[ip]

    def check(self, ip: str, *, now: float | None = None) -> Decision:
        """Consume one unit for `ip`. Only allowed calls are counted."""
        now = time.time() if now is None else now
        with self._lock:
            if len(self._state) > 10_000:
                self._sweep(now)
            entry = self._state.get(ip)

            if entry is None:
                self._state[ip] = (now, 1)
                return Decision(True, remaining=self.max_calls - 1)

            start, count = entry
            elapsed = now - start

            if elapsed >= self.reset_s:          # forgiven — start over
                self._state[ip] = (now, 1)
                return Decision(True, remaining=self.max_calls - 1)

            retry_after = int(self.reset_s - elapsed)

            if elapsed >= self.window_s:         # window closed
                return Decision(False, retry_after_s=retry_after, reason="window_closed")

            if count >= self.max_calls:          # cap hit inside the window
                return Decision(False, retry_after_s=retry_after, reason="cap_reached")

            self._state[ip] = (start, count + 1)
            return Decision(True, remaining=self.max_calls - count - 1)

    def message(self, d: Decision) -> str:
        return (f"The demo budget for this IP is used up — it resets in "
                f"{_human_reset(d.retry_after_s)}. "
                f"Each visitor gets {self.max_calls} answers per "
                f"{_human_reset(self.window_s)}; retrieval-only endpoints are "
                f"not limited.")

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


# The process-wide budget the API route uses.
budget = DemoBudget()


def is_admin(authorization: str | None) -> bool:
    """A valid bearer bypasses the limiter entirely, so the owner (and the
    upload/benchmark scripts, which already carry the token) is never throttled.
    An unset ADMIN_TOKEN grants nothing — otherwise a dev default would make
    every anonymous caller an admin."""
    if not config.ADMIN_TOKEN:
        return False
    return authorization == f"Bearer {config.ADMIN_TOKEN}"
