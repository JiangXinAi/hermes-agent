"""A failed agent build must not wedge the session.

The first send with the local server off fails agent init and stores
``agent_error`` with ``agent_ready`` set. Before the fix, prompt.submit's
build kick was a no-op on that state (``agent_build_started`` stayed True),
so every later send — including the error card's Retry — replayed the stored
failure even after the server came back; only NEW sessions worked. The fix
routes prompt.submit through ``_restart_completed_failed_agent_build`` first,
which clears one completed failed generation and rebuilds with fresh
provider resolution.

These tests pin the restart helper's contract directly: it is the seam the
prompt path now calls, and its answer decides retry-vs-replay.
"""

from __future__ import annotations

import threading
import pytest

from tui_gateway import server


def test_initial_admission_refusal_precedes_original_agent_build_start(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_cli.plugins_dispatch import ExecutionLifecycleError
    manager, phases = PluginManager(), []
    def callback(*, event):
        phases.append((event["phase"], event["resource_kind"]))
        raise ValueError("component initial admission refused")
    PluginContext(PluginManifest(name="component-owner"), manager).register_hook("execution_lifecycle", callback)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(server, "_make_agent", lambda *args, **kwargs: pytest.fail("refused build must not execute"))
    session = {"session_key": "inert-stored", "agent_ready": threading.Event(), "history_lock": threading.Lock()}
    with pytest.raises(ExecutionLifecycleError):
        server._start_agent_build("inert-runtime", session)
    assert phases == [("reserve", "initial")]
    assert not session.get("agent_build_started")
    assert "_agent_build_thread" not in session
    assert not session["agent_ready"].is_set()


def failed_session(tmp_path, sid: str) -> dict:
    """A session record whose deferred agent build COMPLETED in failure."""
    ready = threading.Event()
    ready.set()
    session = {
        "agent": None,
        "agent_ready": ready,
        "agent_build_started": True,
        "agent_error": "The local model server is turned off.",
        "cwd": str(tmp_path),
        "history": [],
        "history_lock": threading.RLock(),
        "profile_home": str(tmp_path),
        "running": False,
        "session_key": sid,
    }
    server._sessions[sid] = session
    return session


def test_restart_clears_failed_generation_and_rebuilds(tmp_path, monkeypatch):
    sid = "wedged-session"
    session = failed_session(tmp_path, sid)
    failed_ready = session["agent_ready"]

    rebuilt = []
    monkeypatch.setattr(server, "_start_agent_build",
                        lambda s, sess: rebuilt.append(s))

    try:
        assert server._restart_completed_failed_agent_build(
            sid, session, failed_ready) is True

        # The failed generation is gone: error cleared, fresh unset ready
        # event, build flag dropped — the next build starts from zero.
        assert session["agent_error"] is None
        assert session["agent_ready"] is not failed_ready
        assert not session["agent_ready"].is_set()
        assert "agent_build_started" not in session
        assert rebuilt == [sid]
    finally:
        server._sessions.pop(sid, None)


def test_restart_declines_every_non_failure_state(tmp_path, monkeypatch):
    """False (caller falls through to the normal build) when there is no
    completed failure: healthy agent, build still in flight, or no error."""
    sid = "healthy-session"
    session = failed_session(tmp_path, sid)
    monkeypatch.setattr(server, "_start_agent_build",
                        lambda s, sess: None)

    try:
        # Build still in flight: ready not set.
        in_flight = threading.Event()
        session["agent_ready"] = in_flight
        assert server._restart_completed_failed_agent_build(
            sid, session, in_flight) is False

        # No error recorded.
        done = threading.Event()
        done.set()
        session["agent_ready"] = done
        session["agent_error"] = None
        assert server._restart_completed_failed_agent_build(
            sid, session, done) is False

        # Agent actually built.
        session["agent_error"] = "stale text"
        session["agent"] = object()
        assert server._restart_completed_failed_agent_build(
            sid, session, done) is False

        # No ready event at all.
        assert server._restart_completed_failed_agent_build(
            sid, session, None) is False
    finally:
        server._sessions.pop(sid, None)
