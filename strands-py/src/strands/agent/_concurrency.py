"""Concurrency and idempotency control for Agent invocations.

Encapsulates the per-Agent state that guards against concurrent invocations and
deduplicates retried requests via caller-supplied idempotency tokens. Designed to
be used by ``Agent.stream_async`` as a single delegate, keeping the orchestration
in ``agent.py`` and the synchronization primitives + bookkeeping here.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..types.agent import ConcurrentInvocationMode
from ..types.exceptions import IdempotencyAbortedError

if TYPE_CHECKING:
    from .agent_result import AgentResult


def _resolve_future(future: asyncio.Future) -> None:
    """Resolve a waiter future on its own loop. No-op if already done or cancelled."""
    if not future.done():
        future.set_result(None)


@dataclass
class _InflightInvocation:
    """Tracks an inflight invocation for idempotency deduplication.

    Duplicate callers register an ``asyncio.Future`` via ``register_waiter`` and await
    it, then read ``result`` or ``error``. The primary calls ``settle`` on completion,
    which resolves every waiter's future on its own event loop via
    ``call_soon_threadsafe``. No waiter ever blocks a thread-pool worker, so a storm of
    duplicates cannot starve the executor the primary needs to make progress.
    """

    result: AgentResult | None = None
    error: BaseException | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _settled: bool = False
    _waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = field(default_factory=list, repr=False)

    @property
    def settled(self) -> bool:
        """Whether the primary has produced a result or error yet."""
        with self._lock:
            return self._settled

    def register_waiter(self) -> asyncio.Future | None:
        """Register the calling coroutine's loop to be notified on completion.

        Returns:
            A future to await, or None if the primary has already settled (the caller
            then reads ``result``/``error`` directly). Returning None closes the race
            where the primary settles between duplicate detection and registration.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            if self._settled:
                return None
            self._waiters.append((loop, future))
        return future

    def settle(self, result: AgentResult | None, error: BaseException | None) -> None:
        """Record the outcome and wake every registered waiter. Idempotent."""
        with self._lock:
            if self._settled:
                return
            self.result = result
            self.error = error
            self._settled = True
            waiters = self._waiters
            self._waiters = []
        for loop, future in waiters:
            try:
                loop.call_soon_threadsafe(_resolve_future, future)
            except RuntimeError:
                # Waiter's loop was already closed/torn down; nothing to wake.
                pass


@dataclass
class _BeginResult:
    """Outcome of ``_ConcurrencyController.begin``.

    Exactly one of the following is the actionable signal for the caller:

    - ``waiting_on`` is set: this call is a duplicate of an inflight token. Await
      ``waiting_on.register_waiter()`` and then yield the cached result or raise the cached error.
    - ``lock_acquired`` is False: a different invocation owns the lock. Raise
      ``ConcurrencyException``.
    - Otherwise: proceed with the invocation. Pass ``registered_token`` back to
      ``complete()`` in the success and error paths so waiters get unblocked.
    """

    waiting_on: _InflightInvocation | None
    registered_token: Any
    lock_acquired: bool


class _ConcurrencyController:
    """Owns the invocation lock and the inflight idempotency-token registry.

    In THROW mode only one invocation can be inflight at a time, so a single
    inflight slot suffices. The lock and registry use ``threading`` primitives
    because ``Agent.run_async()`` may spawn separate event loops on separate threads;
    waiter notification bridges back to each waiter's loop via ``call_soon_threadsafe``.
    """

    def __init__(self, mode: ConcurrentInvocationMode) -> None:
        self._mode = mode
        self._invocation_lock = threading.Lock()
        self._inflight_token: Any = None
        self._inflight: _InflightInvocation | None = None
        self._inflight_lock = threading.Lock()

    @property
    def mode(self) -> ConcurrentInvocationMode:
        """Return the configured concurrency mode."""
        return self._mode

    def begin(self, idempotency_token: Any) -> _BeginResult:
        """Attempt to start a new invocation.

        Combines idempotency-check + lock-acquire into a single call. The returned
        ``_BeginResult`` tells the caller which of three paths to take.

        Args:
            idempotency_token: Caller-provided dedup token, or None.

        Returns:
            See ``_BeginResult``. If ``waiting_on`` is set, the lock is *not* held
            and ``registered_token`` is None.
        """
        waiting_on, registered_token = self._check_idempotency(idempotency_token)
        if waiting_on is not None:
            return _BeginResult(waiting_on=waiting_on, registered_token=None, lock_acquired=False)

        lock_acquired = True
        if self._mode == ConcurrentInvocationMode.THROW:
            lock_acquired = self._invocation_lock.acquire(blocking=False)

        return _BeginResult(waiting_on=None, registered_token=registered_token, lock_acquired=lock_acquired)

    def complete(
        self,
        registered_token: Any,
        *,
        result: AgentResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Signal waiting duplicates and clear the inflight slot.

        Safe to call multiple times for the same ``registered_token`` (subsequent
        calls no-op once the slot has been cleared). Safe to call with
        ``registered_token=None`` (no-op).

        If both ``result`` and ``error`` are None, waiters receive
        ``IdempotencyAbortedError``.
        """
        if registered_token is None:
            return

        with self._inflight_lock:
            if self._inflight_token != registered_token:
                # Another invocation owns the slot (or it was already cleared).
                return
            inflight = self._inflight
            self._inflight_token = None
            self._inflight = None

        if inflight is None:
            return

        if error is not None:
            inflight.settle(None, error)
        elif result is not None:
            inflight.settle(result, None)
        else:
            inflight.settle(None, IdempotencyAbortedError("Primary invocation was aborted before producing a result."))

    def try_acquire_lock(self) -> bool:
        """Non-blockingly acquire the invocation lock.

        Exposed for direct tool callers that bypass the full idempotency flow but
        still need to serialize against an inflight invocation.

        Returns:
            True if the lock was acquired, False otherwise.
        """
        return self._invocation_lock.acquire(blocking=False)

    def release_lock(self) -> None:
        """Release the invocation lock if it is held. Safe to call unconditionally."""
        if self._invocation_lock.locked():
            self._invocation_lock.release()

    def _check_idempotency(self, idempotency_token: Any) -> tuple[_InflightInvocation | None, Any]:
        """Register a new inflight token, identify a duplicate, or no-op.

        Returns:
            ``(waiting_on, registered_token)``:
                - duplicate: ``(inflight_invocation, None)``
                - new request: ``(None, idempotency_token)``
                - different token already inflight, no token provided, or
                  UNSAFE_REENTRANT mode: ``(None, None)``
        """
        if idempotency_token is None or self._mode != ConcurrentInvocationMode.THROW:
            return None, None

        with self._inflight_lock:
            if self._inflight_token == idempotency_token:
                return self._inflight, None
            if self._inflight_token is not None:
                # A different token is inflight; don't overwrite. Caller will hit the
                # lock-acquire path and surface ConcurrencyException.
                return None, None
            self._inflight = _InflightInvocation()
            self._inflight_token = idempotency_token
            return None, idempotency_token
