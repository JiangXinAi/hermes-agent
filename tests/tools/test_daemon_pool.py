"""Tests for tools.daemon_pool.DaemonThreadPoolExecutor.

The daemon pool exists so abandoned workers (interrupted/timed-out tool
batches, wedged memory-provider syncs) can never block interpreter exit:
stdlib ThreadPoolExecutor workers are non-daemon AND registered in
concurrent.futures.thread._threads_queues, whose atexit hook joins every
worker unconditionally — even after shutdown(wait=False).
"""

import subprocess
import sys
import threading
import time

from concurrent.futures.thread import _threads_queues

from tools.daemon_pool import DaemonThreadPoolExecutor


def test_workers_are_daemon_threads():
    pool = DaemonThreadPoolExecutor(max_workers=2)
    try:
        info = pool.submit(
            lambda: (threading.current_thread().daemon, threading.current_thread())
        ).result(timeout=10)
        is_daemon, worker = info
        assert is_daemon is True
        # Not registered with concurrent.futures' atexit join hook.
        assert worker not in _threads_queues
    finally:
        pool.shutdown(wait=True)


def test_idle_worker_reuse():
    pool = DaemonThreadPoolExecutor(max_workers=4)
    try:
        tid1 = pool.submit(threading.get_ident).result(timeout=10)
        time.sleep(0.05)  # let the worker park on the idle semaphore
        tid2 = pool.submit(threading.get_ident).result(timeout=10)
        assert tid1 == tid2
    finally:
        pool.shutdown(wait=True)


def test_wedged_worker_does_not_block_interpreter_exit():
    """A worker stuck in a long sleep must not hold the process open.

    With stdlib ThreadPoolExecutor this subprocess hangs until the sleep
    finishes (the atexit hook joins the worker); with the daemon pool it
    exits as soon as the main thread returns.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tools.daemon_pool import DaemonThreadPoolExecutor\n"
        "import time\n"
        "pool = DaemonThreadPoolExecutor(max_workers=1)\n"
        "pool.submit(time.sleep, 120)\n"
        "time.sleep(0.3)\n"
        "pool.shutdown(wait=False)\n"
        "print('main-done', flush=True)\n"
    ) % (str(_repo_root()),)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "main-done" in proc.stdout


def test_submit_propagates_caller_contextvars():
    """Pool workers inherit contextvars set in the submitting context.

    Stdlib ThreadPoolExecutor snapshots the caller's context with
    ``copy_context()``; some bundled CPython runtime builds strip that, so
    the daemon pool restores it explicitly.  Without the fix this returns
    the default because the worker runs in a bare context.
    """
    from contextvars import ContextVar

    var = ContextVar("daemon_pool_test_var", default="unset")

    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        token = var.set("hello")
        try:
            seen = pool.submit(var.get).result(timeout=10)
        finally:
            var.reset(token)
        assert seen == "hello"
    finally:
        pool.shutdown(wait=True)


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]


def test_execution_outcome_does_not_claim_actual_pool_worker_join(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
    from hermes_cli.plugins_dispatch import reserve_execution_resource, execution_lifecycle_scope
    manager = PluginManager()
    events = []
    def callback(*, event):
        events.append((event["resource_kind"], event["phase"]))
        return ({"admitted": True, "ticket": object()} if event["phase"] == "reserve" else {"recorded": True})
    PluginContext(PluginManifest(name="bounded-owner"), manager).register_hook("execution_lifecycle", callback)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    original = reserve_execution_resource("tool", "original-call", context={"task_id": "original-task"})
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        with execution_lifecycle_scope(original):
            future = pool.submit(lambda: "inert-component-result")
        assert future.result(5) == "inert-component-result"
        assert pool.reconcile_execution_lifecycle(0) == 1
        assert ("pool_worker", "joined") not in events
        pool.shutdown(wait=True)
        assert events.index(("pool_worker", "worker_finally")) < events.index(("pool_worker", "joined"))
        assert ("work_item", "invocation_exited") in events
    finally:
        pool.shutdown(wait=True)


def test_denied_worker_start_cannot_run_queued_original_work_item_later(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
    from hermes_cli.plugins_dispatch import ExecutionLifecycleError, reserve_execution_resource, execution_lifecycle_scope
    import pytest
    manager = PluginManager()
    calls = []
    def callback(*, event):
        if event["phase"] == "reserve" and event["resource_kind"] == "pool_worker":
            raise ValueError("worker admission denied")
        return ({"admitted": True, "ticket": object()} if event["phase"] == "reserve" else {"recorded": True})
    PluginContext(PluginManifest(name="bounded-owner"), manager).register_hook("execution_lifecycle", callback)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    original = reserve_execution_resource("tool", "original-call", context={})
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        with execution_lifecycle_scope(original), pytest.raises(ExecutionLifecycleError):
            pool.submit(lambda: calls.append("denied"))
        assert pool.submit(lambda: "ordinary-unscoped").result(5) == "ordinary-unscoped"
        assert calls == []
    finally:
        pool.shutdown(wait=True)
