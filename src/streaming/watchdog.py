"""Progress watchdog for detecting a live Spark query that stopped issuing batches."""

from dataclasses import dataclass


class StreamStalledError(RuntimeError):
    """Raised when a query produces no new progress heartbeat within the timeout."""


@dataclass
class ProgressWatchdog:
    timeout_seconds: float
    last_token: str | None = None
    last_progress_at: float | None = None

    def observe(self, token: str | None, now: float) -> bool:
        """Record a progress token; return ``True`` when the stream is stale."""
        if self.last_progress_at is None:
            self.last_progress_at = now
            self.last_token = token
            return False
        if token is not None and token != self.last_token:
            self.last_token = token
            self.last_progress_at = now
            return False
        return now - self.last_progress_at > self.timeout_seconds
