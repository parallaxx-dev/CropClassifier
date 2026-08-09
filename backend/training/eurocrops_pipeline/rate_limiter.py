"""
rate_limiter.py — shared token-bucket, one instance for the entire run (not
re-created per country), so the budget is respected across country transitions.

Deliberately conservative given Phase 0's empirical finding: the sentinelhub SDK
has its own client-side rate limiter, and pushing concurrency past a fairly low
ceiling causes it to back off hard (throughput collapses, not plateaus). This
limiter's job is to keep this pipeline's own request rate low enough that the
SDK's backoff rarely triggers at all, not to try to ride right up against CDSE's
stated 300/min cap.
"""

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self._rate = rate_per_sec
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self._rate
            time.sleep(wait)
