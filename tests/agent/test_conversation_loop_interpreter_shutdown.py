"""Regression guard: interpreter-shutdown errors must abort the conversation
loop immediately instead of retrying.

When the Python interpreter begins its teardown sequence, every executor-backed
operation (API calls, tool dispatch, memory sync) raises::

    RuntimeError: cannot schedule new futures after interpreter shutdown

Before the fix, the outer ``except Exception`` handler in
``run_conversation`` caught this error but did not recognise it as fatal.
Since ``api_call_count`` was nowhere near ``agent.max_iterations - 1``, the
loop continued — each iteration hit the same dead executor and failed
identically, producing a cascade of ``❌ Error during OpenAI-compatible API
call #N`` messages (#93217).

The fix adds an early check in the outer except handler: if
``sys.is_finalizing()`` is True or the error message matches the
``"cannot schedule new futures"`` pattern, the loop breaks immediately with
a clean ``interpreter_shutdown`` exit reason.
"""
from __future__ import annotations

import types
import pytest

from agent.conversation_loop import _is_interpreter_shutdown_error


@pytest.mark.parametrize("raises", [False, True])
def test_original_turn_wrapper_records_exit_and_restores_outer_scope_on_early_return(monkeypatch, raises):
    import agent.conversation_loop as module
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_cli.plugins_dispatch import (bind_execution_turn, current_execution_lifecycle,
        execution_lifecycle_scope, initial_execution_scope, reserve_execution_resource)
    events, manager = [], PluginManager()
    def callback(*, event):
        events.append((event["phase"], event["resource_kind"]))
        return {"admitted": True, "ticket": object()} if event["phase"] in {"reserve", "bind_turn"} else {"recorded": True}
    PluginContext(PluginManifest(name="component-owner"), manager).register_hook("execution_lifecycle", callback)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    initial = initial_execution_scope("inert-runtime", {"session_key": "inert-stored"})
    worker = reserve_execution_resource("conversation_worker", "inert-worker", parent=initial)
    class ComponentExit(BaseException):
        pass
    def early_turn(agent, user_message, **kwargs):
        bind_execution_turn({"session_id": "inert-stored", "task_id": "inert-task", "turn_id": "inert-turn"})
        if raises:
            raise ComponentExit("original component control flow")
        return {"completed": False}
    monkeypatch.setattr(module, "_run_conversation_turn", early_turn)
    with execution_lifecycle_scope(worker):
        if raises:
            with pytest.raises(ComponentExit):
                module.run_conversation(types.SimpleNamespace(), "inert component input")
        else:
            assert module.run_conversation(types.SimpleNamespace(), "") == {"completed": False}
        assert current_execution_lifecycle() is worker
    assert events.count(("invocation_exited", "turn")) == 1
    assert not any(phase == "joined" for phase, _ in events)


class TestInterpreterShutdownDetection:
    """Verify the interpreter-shutdown error matcher used by the
    conversation loop's outer except handler."""

    def test_matches_full_interpreter_shutdown_message(self):
        """The canonical CPython asyncio shutdown message."""
        exc = RuntimeError(
            "cannot schedule new futures after interpreter shutdown"
        )
        assert _is_interpreter_shutdown_error(exc) is True

    def test_matches_short_shutdown_variant(self):
        """Plain ThreadPoolExecutor shutdown variant (no 'interpreter')."""
        exc = RuntimeError("cannot schedule new futures after shutdown")
        assert _is_interpreter_shutdown_error(exc) is True

    def test_case_insensitive_match(self):
        """Error text may arrive in different case from some executor types."""
        exc = RuntimeError("Cannot Schedule New Futures After Interpreter Shutdown")
        assert _is_interpreter_shutdown_error(exc) is True

    def test_does_not_match_unrelated_runtime_error(self):
        """Unrelated RuntimeErrors must not trigger the shutdown path."""
        exc = RuntimeError("connection reset by peer")
        assert _is_interpreter_shutdown_error(exc) is False

    def test_does_not_match_non_runtime_error(self):
        """Non-RuntimeError exceptions must not match."""
        exc = ValueError("cannot schedule new futures")
        assert _is_interpreter_shutdown_error(exc) is False

    def test_does_not_match_none(self):
        """None must not match (defensive — caller may pass None)."""
        try:
            result = _is_interpreter_shutdown_error(None)  # type: ignore[arg-type]
        except TypeError:
            result = False
        assert result is False

    def test_does_not_match_empty_string_exception(self):
        """Empty-message exceptions must not match."""
        exc = RuntimeError("")
        assert _is_interpreter_shutdown_error(exc) is False
