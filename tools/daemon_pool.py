"""Shared daemon-thread ThreadPoolExecutor.

Stdlib workers are non-daemon AND registered in ``_threads_queues``, whose atexit
hook joins every worker even after ``shutdown(wait=False)`` — one wedged worker
(tool blocked on network I/O, hung provider, stuck subagent) blocks interpreter
exit forever. This variant spawns daemon workers and skips that registration.
Use it for best-effort/interruptible work that must never hold the process open;
NOT for work that must complete before exit (durable writes belong on foreground
threads with explicit bounded joins).
"""

from __future__ import annotations

import threading
import weakref
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker
from contextvars import copy_context

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def submit(self, fn, /, *args, **kwargs):
        """Submit a callable, propagating the caller's contextvars. Stdlib only does
        this from 3.14; on 3.11-3.13 a bare worker starts with an EMPTY Context and
        drops profile secret scope / HERMES_HOME override — under the multiplexed
        gateway a credential read then fails closed with ``UnscopedSecretError``.
        Unconditional: on 3.14+ ``ctx.run`` re-applies the same context (no-op)."""
        ctx = copy_context()
        from hermes_cli.plugins_dispatch import (
            current_execution_lifecycle, reserve_execution_resource,
        )
        parent = current_execution_lifecycle()
        lifecycle = (reserve_execution_resource("work_item", uuid4().hex, parent=parent)
                     if parent is not None and parent.tracked else None)
        admission = threading.Event() if lifecycle is not None else None
        submitted = [False]

        def _run_with_context(*call_args, **call_kwargs):
            if admission is not None:
                admission.wait()
                if not submitted[0]:
                    from hermes_cli.plugins_dispatch import ExecutionLifecycleError
                    raise ExecutionLifecycleError("Original work item was not submitted")
            def run_original():
                try:
                    return fn(*call_args, **call_kwargs)
                finally:
                    if lifecycle is not None:
                        lifecycle.observe("invocation_exited", thread_ident=threading.get_ident())
            return ctx.run(run_original)
        try:
            future = super().submit(_run_with_context, *args, **kwargs)
            if lifecycle is not None:
                lifecycle.observe("submitted")
            submitted[0] = True
            return future
        finally:
            # A failed _adjust_thread_count leaves a stdlib queued work item.
            # Its original wrapper must never dispatch after failed admission.
            if admission is not None:
                admission.set()

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation (3.8–3.13) with two changes:
        # daemon=True and no _threads_queues registration.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)
        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            from hermes_cli.plugins_dispatch import (
                current_execution_lifecycle, reserve_execution_resource,
            )
            parent = current_execution_lifecycle()
            lifecycle = (reserve_execution_resource("pool_worker", uuid4().hex, parent=parent)
                         if parent is not None and parent.tracked else None)
            observation_context = copy_context()

            def worker(*worker_args):
                try:
                    return _worker(*worker_args)
                finally:
                    if lifecycle is not None:
                        # finally is not proof that the actual thread has joined.
                        observation_context.run(lifecycle.observe, "worker_finally",
                                                thread_ident=threading.get_ident())
            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile (#54937).
            t = threading.Thread(
                name=thread_name, target=worker, daemon=True,
                args=(weakref.ref(self, weakref_cb), self._work_queue, self._initializer, self._initargs),
            )
            t._execution_lifecycle = lifecycle
            t.start()
            self._threads.add(t)

    def reconcile_execution_lifecycle(self, timeout=0.0):
        """Bounded joins of these original worker objects, never Future.done."""
        import time
        deadline = time.monotonic() + max(0.0, float(timeout))
        remaining = 0
        for thread in tuple(self._threads):
            lifecycle = getattr(thread, "_execution_lifecycle", None)
            if lifecycle is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                remaining += 1
            else:
                lifecycle.observe("joined", thread_ident=thread.ident)
        return remaining

    def shutdown(self, wait=True, *, cancel_futures=False):
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait:
            self.reconcile_execution_lifecycle()
