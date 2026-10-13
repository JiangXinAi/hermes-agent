"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.
"""

from unittest.mock import patch

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, raise_in):
        self._raise_in = set(raise_in)
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        if "save_trajectory" in self._raise_in:
            raise RuntimeError("trajectory disk full")

    def _cleanup_task_resources(self, *a, **k):
        if "cleanup_task_resources" in self._raise_in:
            raise RuntimeError("docker teardown EOF")

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        if "persist_session" in self._raise_in:
            raise RuntimeError("sqlite database is locked")

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run(
    agent,
    *,
    final_response=None,
    api_call_count=3,
    turn_exit_reason="unknown",
):
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )




@pytest.mark.parametrize(
    "step", ["save_trajectory", "cleanup_task_resources", "persist_session"]
)
def test_single_cleanup_step_raises_does_not_skip_others(step):
    agent = _StubAgent(raise_in=(step,))
    result = _run(agent)
    # Response survives.
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    # Exactly the failing step is recorded; the others ran without error.
    assert result["cleanup_errors"] == [
        next(
            e
            for e in result["cleanup_errors"]
            if e.startswith(step)
        )
    ]
    assert len(result["cleanup_errors"]) == 1


def test_clean_turn_has_no_cleanup_errors_key():
    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    assert result["completed"] is False
    assert "cleanup_errors" not in result


@pytest.mark.parametrize(
    ("persist_disabled", "expected_calls"),
    [
        (True, ["transform_llm_output"]),
        (False, ["transform_llm_output", "post_llm_call", "on_session_end"]),
    ],
)
def test_persist_disabled_turn_skips_session_end_hook(
    persist_disabled, expected_calls
):
    agent = _StubAgent(raise_in=())
    agent._persist_disabled = persist_disabled
    calls = []

    def capture(name, _logger, **_kwargs):
        calls.append(name)
        return []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=capture):
        _run(
            agent,
            final_response="done",
            api_call_count=1,
            turn_exit_reason="text_response(stop)",
        )

    assert calls == expected_calls




def test_finalizer_owner_fence_failure_cannot_be_reported_as_a_join(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
    from hermes_cli.plugins_dispatch import ExecutionLifecycleError
    manager = PluginManager()
    phases = []
    def refuse(*, event):
        phases.append(event["phase"])
        raise ValueError("original fence unavailable")
    PluginContext(PluginManifest(name="bounded-owner"), manager).register_hook("execution_lifecycle", refuse)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    agent = _StubAgent(raise_in=())
    cleanups, trajectory_completion = [], []
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda task_id: cleanups.append(("cleanup", task_id)))
    monkeypatch.setattr(agent, "_persist_session", lambda *args: cleanups.append(("persist", None)))
    monkeypatch.setattr(agent, "_save_trajectory", lambda messages, user, completed: trajectory_completion.append(completed))
    with pytest.raises(ExecutionLifecycleError):
        _run(agent, final_response="inert finite result", api_call_count=2)
    assert phases == ["fence"]
    assert cleanups == [("cleanup", "task-1"), ("persist", None)]
    assert trajectory_completion == [False]



def test_join_observation_refusal_preserves_owned_cleanup_and_persistence(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
    from hermes_cli.plugins_dispatch import ExecutionLifecycleError, reserve_execution_resource, execution_lifecycle_scope
    from tools.daemon_pool import DaemonThreadPoolExecutor
    manager, phases, cleanups = PluginManager(), [], []
    refuse_join = [True]
    def callback(*, event):
        phases.append(event["phase"])
        if event["phase"] == "joined" and refuse_join[0]:
            raise ValueError("original join observation unavailable")
        return ({"admitted": True, "ticket": object()} if event["phase"] == "reserve" else {"recorded": True})
    PluginContext(PluginManifest(name="bounded-owner"), manager).register_hook("execution_lifecycle", callback)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    original = reserve_execution_resource("tool", "original-call", context={})
    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        with execution_lifecycle_scope(original):
            assert pool.submit(lambda: "inert-component-result").result(5) == "inert-component-result"
        pool.shutdown(wait=False)
        agent = _StubAgent(raise_in=())
        agent._tool_lifecycle_pools = [pool]
        monkeypatch.setattr(agent, "_cleanup_task_resources", lambda task_id: cleanups.append("cleanup"))
        monkeypatch.setattr(agent, "_persist_session", lambda *args: cleanups.append("persist"))
        with pytest.raises(ExecutionLifecycleError):
            _run(agent, final_response="inert finite result", api_call_count=2)
        assert phases.index("worker_finally") < phases.index("joined")
        assert cleanups == ["cleanup", "persist"]
        assert agent._tool_lifecycle_pools == [pool]
    finally:
        refuse_join[0] = False
        pool.shutdown(wait=True)


def test_finalizer_reconciles_original_background_worker_without_joining_its_caller(monkeypatch):
    import threading
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_cli.plugins_dispatch import (execution_lifecycle_scope, execution_worker_target,
        initial_execution_scope, reserve_execution_resource, retain_execution_worker)
    manager, phases = PluginManager(), []
    def callback(*, event):
        phases.append((event["phase"], event["resource_kind"]))
        return {"admitted": True, "ticket": object()} if event["phase"] == "reserve" else {"recorded": True}
    PluginContext(PluginManifest(name="component-owner"), manager).register_hook("execution_lifecycle", callback)
    monkeypatch.setattr(plugins_mod, "_delivery_manager", lambda: manager)
    initial = initial_execution_scope("inert-runtime", {"session_key": "inert-stored"})
    handle = reserve_execution_resource("background_review", "inert-original-review", parent=initial)
    finished = threading.Event()
    original = threading.Thread(target=execution_worker_target(handle, finished.set))
    retain_execution_worker(handle, original)
    original.start()
    try:
        assert finished.wait(2)
        with execution_lifecycle_scope(initial):
            _run(_StubAgent(raise_in=()), final_response="inert finite component result", api_call_count=1)
        assert not original.is_alive()
        assert phases.index(("worker_finally", "background_review")) < phases.index(("joined", "background_review"))
        assert not any(phase == "joined" and kind in {"initial", "turn", "conversation_worker"} for phase, kind in phases)
    finally:
        original.join(3)
