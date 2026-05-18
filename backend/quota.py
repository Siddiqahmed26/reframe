"""Per-IP daily quota for the shared key pool.

In-memory only. State is wiped on container restart, which is intentional:
HF Spaces ephemeral hosting + simple ops, and the shared pool is meant as a
free taste, not a guaranteed allowance.

Daily reset: rolls over at UTC midnight by date string comparison.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict


_lock = threading.Lock()
_counts: Dict[str, Dict[str, object]] = {}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _state_for(ip: str) -> Dict[str, object]:
    today = _today()
    rec = _counts.get(ip)
    if rec is None or rec.get("date") != today:
        rec = {"date": today, "count": 0}
        _counts[ip] = rec
    return rec


def record_usage(ip: str) -> int:
    """Increment the IP's count for today and return the new count."""
    with _lock:
        rec = _state_for(ip)
        rec["count"] = int(rec["count"]) + 1
        return int(rec["count"])


def remaining(ip: str, limit: int) -> int:
    """How many shared-key requests this IP has left today."""
    with _lock:
        rec = _state_for(ip)
        return max(0, limit - int(rec["count"]))


def is_exhausted(ip: str, limit: int) -> bool:
    return remaining(ip, limit) <= 0
