"""Token-bucket rate limiter for TorchaVerse.

Provides :class:`RateLimiter`, a thread-safe token-bucket implementation
that controls the frequency of operations.  Tokens are replenished at a
configurable ``rate`` (tokens per second) up to a ``burst`` capacity.
Both blocking (:meth:`acquire`) and non-blocking (:meth:`try_acquire`)
acquisition are supported, as well as an asynchronous variant
(:meth:`acquire_async`).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

__all__ = ["RateLimiter"]


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    The bucket starts full (``burst`` tokens).  Tokens are refilled
    continuously at ``rate`` tokens per second.  When a caller requests
    more tokens than are available, :meth:`acquire` blocks until enough
    tokens have accumulated, while :meth:`try_acquire` returns ``False``
    immediately.

    Args:
        rate: Refill rate in tokens per second.  Must be ``> 0``.
        burst: Maximum bucket capacity.  Must be ``> 0``.
        initial: Initial number of tokens.  Defaults to ``burst``.

    Example:
        >>> limiter = RateLimiter(rate=10, burst=20)
        >>> limiter.acquire(5)        # blocks if needed
        >>> limiter.try_acquire(100)  # False, exceeds capacity
    """

    def __init__(
        self,
        rate: float,
        burst: float,
        initial: Optional[float] = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}.")
        if burst <= 0:
            raise ValueError(f"burst must be > 0, got {burst}.")

        self._rate: float = float(rate)
        self._burst: float = float(burst)
        self._tokens: float = float(burst if initial is None else initial)
        self._last_update: float = time.monotonic()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def rate(self) -> float:
        """Refill rate (tokens per second)."""
        return self._rate

    @property
    def burst(self) -> float:
        """Maximum bucket capacity."""
        return self._burst

    @property
    def tokens(self) -> float:
        """Current number of available tokens (after refilling)."""
        with self._lock:
            return self._refill_locked()

    # ------------------------------------------------------------------
    # Core refill logic
    # ------------------------------------------------------------------
    def _refill_locked(self) -> float:
        """Refill the bucket based on elapsed time.

        Must be called while holding ``self._lock``.  Returns the current
        token count after refilling.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        if elapsed > 0:
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_update = now
        return self._tokens

    # ------------------------------------------------------------------
    # Synchronous acquisition
    # ------------------------------------------------------------------
    def try_acquire(self, tokens: float = 1) -> bool:
        """Attempt to acquire ``tokens`` without blocking.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            ``True`` if the tokens were acquired, ``False`` otherwise.
        """
        if tokens <= 0:
            raise ValueError(f"tokens must be > 0, got {tokens}.")
        with self._lock:
            available = self._refill_locked()
            if available >= tokens:
                self._tokens = available - tokens
                return True
            return False

    def acquire(self, tokens: float = 1, timeout: Optional[float] = None) -> bool:
        """Acquire ``tokens``, blocking until they are available.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum seconds to wait.  ``None`` waits forever.

        Returns:
            ``True`` if the tokens were acquired, ``False`` on timeout.
        """
        if tokens <= 0:
            raise ValueError(f"tokens must be > 0, got {tokens}.")
        if tokens > self._burst:
            raise ValueError(
                f"Requested {tokens} tokens exceeds burst capacity "
                f"{self._burst}."
            )

        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            with self._lock:
                available = self._refill_locked()
                if available >= tokens:
                    self._tokens = available - tokens
                    return True
                # Time until enough tokens are available.
                deficit = tokens - available
                wait = deficit / self._rate

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)

            # Sleep in small slices to remain responsive to lock changes.
            time.sleep(min(wait, 0.05))

    # ------------------------------------------------------------------
    # Asynchronous acquisition
    # ------------------------------------------------------------------
    async def acquire_async(
        self, tokens: float = 1, timeout: Optional[float] = None
    ) -> bool:
        """Asynchronously acquire ``tokens``.

        Uses :func:`asyncio.sleep` to yield control while waiting, making
        it suitable for use within ``async`` coroutines.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum seconds to wait.  ``None`` waits forever.

        Returns:
            ``True`` if the tokens were acquired, ``False`` on timeout.
        """
        if tokens <= 0:
            raise ValueError(f"tokens must be > 0, got {tokens}.")
        if tokens > self._burst:
            raise ValueError(
                f"Requested {tokens} tokens exceeds burst capacity "
                f"{self._burst}."
            )

        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            with self._lock:
                available = self._refill_locked()
                if available >= tokens:
                    self._tokens = available - tokens
                    return True
                deficit = tokens - available
                wait = deficit / self._rate

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)

            await asyncio.sleep(min(wait, 0.05))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Refill the bucket to full capacity."""
        with self._lock:
            self._tokens = self._burst
            self._last_update = time.monotonic()

    def __repr__(self) -> str:
        return (
            f"RateLimiter(rate={self._rate}, burst={self._burst}, "
            f"tokens={self._tokens:.2f})"
        )
