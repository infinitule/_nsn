"""
Auto-mode test suite — no API key required.

All tests mock NoesisLoop.run to avoid LLM calls. The MockResult factory
produces deterministic NoesisResult objects so assertions are stable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noesis.evaluator import EvalScore, _COHERENCE_FLOOR, score
from noesis.loop import NoesisLoop, NoesisResult
from noesis.persistence import SessionPersistence, exists, load, save
from noesis.scheduler import ScheduleConfig, schedule_config
from noesis.self_model import SelfModel
from noesis.auto_mode import AutoMode, CycleRecord
from noesis.agent import NOESISAgent
from prism_bridge import PRISMBridge


# ── Shared fixtures ────────────────────────────────────────────────────────

def make_self_model(confidence: float = 0.90, depth: int = 0) -> SelfModel:
    bridge = PRISMBridge(seed_dim=64)
    m = SelfModel(identity="TEST", attention_vector=bridge.seed_attention_vector())
    m.confidence = confidence
    m.metacognitive_depth = depth
    return m


def make_result(
    output: str = "Response text.",
    depth: int = 1,
    self_model: SelfModel | None = None,
) -> NoesisResult:
    if self_model is None:
        self_model = make_self_model()
    return NoesisResult(output=output, depth_used=depth, self_model=self_model, raw_turns=[])


@pytest.fixture
def agent():
    with patch("anthropic.Anthropic"):
        a = NOESISAgent(api_key="test-key")
    return a


# ── SessionPersistence ─────────────────────────────────────────────────────

class TestSessionPersistence:

    def test_save_load_roundtrip(self, tmp_path):
        m = make_self_model(confidence=0.77)
        m.session_wisdom = ["insight one", "insight two"]
        m.current_intent = "explore photonics"
        path = tmp_path / "state.json"

        save(m, path, cycles_completed=3)
        loaded, cycles = load(path)

        assert cycles == 3
        assert loaded.identity == m.identity
        assert loaded.confidence == pytest.approx(m.confidence)
        assert loaded.current_intent == m.current_intent
        assert loaded.session_wisdom == m.session_wisdom
        np.testing.assert_allclose(loaded.attention_vector, m.attention_vector)

    def test_session_id_preserved(self, tmp_path):
        m = make_self_model()
        path = tmp_path / "state.json"
        save(m, path)
        loaded, _ = load(path)
        assert loaded.session_id == m.session_id

    def test_cycles_completed_roundtrip(self, tmp_path):
        m = make_self_model()
        path = tmp_path / "state.json"
        save(m, path, cycles_completed=7)
        _, cycles = load(path)
        assert cycles == 7

    def test_exists_returns_true_after_save(self, tmp_path):
        m = make_self_model()
        path = tmp_path / "state.json"
        assert not exists(path)
        save(m, path)
        assert exists(path)

    def test_exists_returns_false_for_missing(self, tmp_path):
        assert not exists(tmp_path / "nonexistent.json")

    def test_exists_handles_oserror(self):
        with patch("noesis.persistence.Path") as MockPath:
            MockPath.return_value.exists.side_effect = OSError("permission denied")
            assert not exists("/some/path")

    def test_atomic_write_preserves_original(self, tmp_path):
        m1 = make_self_model(confidence=0.55)
        m1._replace = None  # make it distinct
        path = tmp_path / "state.json"
        save(m1, path, cycles_completed=1)

        m2 = make_self_model(confidence=0.99)
        tmp_file = Path(str(path) + ".tmp")

        # Simulate a crash between write and rename by patching Path.replace
        with patch.object(Path, "replace", side_effect=OSError("simulated crash")):
            with pytest.raises(OSError):
                save(m2, path, cycles_completed=2)

        # Original file must still be readable as V1
        loaded, cycles = load(path)
        assert cycles == 1
        assert loaded.confidence == pytest.approx(0.55)

    def test_session_persistence_static_methods(self, tmp_path):
        m = make_self_model()
        path = tmp_path / "sp.json"
        SessionPersistence.save(m, path, cycles_completed=5)
        assert SessionPersistence.exists(path)
        loaded, cycles = SessionPersistence.load(path)
        assert cycles == 5
        assert loaded.identity == m.identity


# ── Evaluator ──────────────────────────────────────────────────────────────

class TestEvaluator:

    def _model(self, confidence: float = 0.90, depth: int = 0, coherence: float = 0.90) -> SelfModel:
        m = make_self_model(confidence=confidence, depth=depth)
        m.coherence = coherence
        m.confidence = confidence
        m.metacognitive_depth = depth
        return m

    def test_high_confidence_should_advance(self):
        m = self._model(confidence=0.95, depth=0, coherence=0.90)
        result = score("What is light?", "Light is electromagnetic radiation " * 20, 0.95, m)
        assert result.should_advance is True
        assert result.depth_worthy is False

    def test_low_confidence_depth_worthy(self):
        m = self._model(confidence=0.50, depth=1, coherence=0.80)
        result = score("Explain quantum entanglement.", "Some response " * 10, 0.50, m,
                       threshold=0.80, max_depth=3)
        assert result.depth_worthy is True
        assert result.should_advance is False

    def test_at_max_depth_not_depth_worthy(self):
        m = self._model(confidence=0.50, depth=3, coherence=0.80)
        result = score("task", "response " * 10, 0.50, m, threshold=0.80, max_depth=3)
        assert result.depth_worthy is False

    def test_below_max_depth_is_depth_worthy(self):
        m = self._model(confidence=0.50, depth=2, coherence=0.80)
        result = score("task", "response " * 10, 0.50, m, threshold=0.80, max_depth=3)
        assert result.depth_worthy is True

    def test_below_coherence_floor_not_depth_worthy(self):
        m = self._model(confidence=0.50, depth=1, coherence=_COHERENCE_FLOOR - 0.01)
        result = score("task", "response " * 10, 0.50, m, threshold=0.80, max_depth=3)
        assert result.depth_worthy is False

    def test_empty_task_no_zerodivision(self):
        m = self._model()
        result = score("", "some response", 0.9, m)
        assert 0.0 <= result.task_completion <= 1.0

    def test_threshold_parameter_respected(self):
        m = self._model(confidence=0.75, depth=0, coherence=0.90)
        # With threshold=0.70, confidence=0.75 → not depth_worthy
        result = score("task", "response " * 30, 0.75, m, threshold=0.70)
        assert result.depth_worthy is False

    def test_task_completion_in_unit_interval(self):
        m = self._model()
        for conf in [0.0, 0.5, 1.0]:
            r = score("task", "response", conf, m)
            assert 0.0 <= r.task_completion <= 1.0

    def test_coherence_score_mirrors_model(self):
        m = self._model(coherence=0.73)
        r = score("task", "response", 0.9, m)
        assert r.coherence_score == pytest.approx(0.73, abs=1e-3)


# ── Scheduler ─────────────────────────────────────────────────────────────

class TestScheduleConfig:

    @pytest.fixture
    def bridge(self):
        return PRISMBridge(seed_dim=64)

    def test_deterministic_same_inputs(self, bridge):
        m = make_self_model()
        c1 = schedule_config("What is coherent detection?", m, bridge)
        c2 = schedule_config("What is coherent detection?", m, bridge)
        assert c1 == c2

    def test_threshold_in_range(self, bridge):
        m = make_self_model()
        for task in ["short", "?" * 20, "A very long question " * 10 + "?"]:
            c = schedule_config(task, m, bridge)
            assert 0.60 <= c.threshold <= 0.92, f"threshold {c.threshold} out of range for task: {task!r}"

    def test_max_tokens_capped(self, bridge):
        m = make_self_model()
        task = ("long question " * 20) + "?" * 20
        c = schedule_config(task, m, bridge)
        assert c.max_tokens <= 4096

    def test_question_count_capped(self, bridge):
        # Same word count, question chars differ but are capped at 5 for both
        m = make_self_model()
        base = "hello world " * 10
        c5  = schedule_config(base + "?" * 5,  m, bridge)
        c20 = schedule_config(base + "?" * 20, m, bridge)
        assert c5.max_tokens == c20.max_tokens
        assert c5.threshold == pytest.approx(c20.threshold)

    def test_mi_contributes_to_depth(self, bridge):
        m = make_self_model()
        c = schedule_config("simple task", m, bridge)
        # Fresh seed has known MI ~3.9 → depth_bonus = int(clip(3.9/2,0,2)) = 1
        # so max_depth should be base + at least some bonus
        assert c.max_depth >= 3

    def test_schedule_config_namedtuple(self, bridge):
        m = make_self_model()
        c = schedule_config("task", m, bridge)
        assert isinstance(c, ScheduleConfig)
        assert hasattr(c, "max_depth")
        assert hasattr(c, "threshold")
        assert hasattr(c, "max_tokens")


# ── AutoMode ───────────────────────────────────────────────────────────────

class TestAutoMode:
    """All tests patch NoesisLoop.run to avoid LLM calls."""

    @pytest.fixture
    def mock_result(self, agent):
        return make_result(output="Test response.", depth=1, self_model=agent.self_model)

    def _run_auto(self, agent, tasks, max_cycles=10, mock_depth=1,
                  mock_confidence=0.92, auto_generate=False,
                  persist_path=None, stop_on_error=False,
                  on_cycle=None):
        """Helper: run AutoMode with a mocked NoesisLoop.run."""
        result_model = agent.self_model
        result_model.confidence = mock_confidence
        mock_res = make_result(
            output="Mocked response " * 20,
            depth=mock_depth,
            self_model=result_model,
        )
        auto = AutoMode(agent, auto_generate=auto_generate, on_cycle=on_cycle)
        with patch.object(NoesisLoop, "run", return_value=mock_res):
            records = auto.run(
                initial_tasks=tasks,
                max_cycles=max_cycles,
                persist_path=persist_path,
                stop_on_error=stop_on_error,
            )
        return auto, records

    def test_runs_queue_to_completion(self, agent):
        _, records = self._run_auto(agent, ["t1", "t2", "t3"], max_cycles=10)
        assert len(records) == 3

    def test_stops_at_max_cycles(self, agent):
        tasks = [f"task-{i}" for i in range(10)]
        _, records = self._run_auto(agent, tasks, max_cycles=2)
        assert len(records) == 2

    def test_warm_restart_respects_prior_cycles(self, agent, tmp_path):
        path = tmp_path / "state.json"
        save(agent.self_model, path, cycles_completed=8)
        tasks = [f"task-{i}" for i in range(10)]
        _, records = self._run_auto(agent, tasks, max_cycles=10, persist_path=path)
        # 10 - 8 = 2 cycles allowed
        assert len(records) <= 2

    def test_persists_after_each_cycle(self, agent, tmp_path):
        path = tmp_path / "state.json"
        seen = []

        def on_cycle(r):
            seen.append(exists(path))

        _, records = self._run_auto(agent, ["t1", "t2"], persist_path=path, on_cycle=on_cycle)
        assert all(seen), "state file must exist after every cycle"

    def test_generates_tasks_from_wisdom(self, agent):
        agent.self_model.session_wisdom = ["coherent detection is fast", "phase matters"]
        _, records = self._run_auto(
            agent, [], max_cycles=3, auto_generate=True
        )
        assert len(records) > 0

    def test_generates_no_task_without_wisdom(self, agent):
        agent.self_model.session_wisdom = []
        _, records = self._run_auto(agent, [], max_cycles=5, auto_generate=True)
        assert len(records) == 0

    def test_on_cycle_callback_called(self, agent):
        called = []
        _, records = self._run_auto(agent, ["t1", "t2"], on_cycle=lambda r: called.append(r))
        assert len(called) == 2
        assert all(isinstance(r, CycleRecord) for r in called)

    def test_agent_loop_not_mutated(self, agent):
        original_threshold = agent.loop.threshold
        original_max_depth = agent.loop.max_depth

        mock_res = make_result(self_model=agent.self_model)
        with patch.object(NoesisLoop, "run", return_value=mock_res):
            auto = AutoMode(agent)
            auto._run_with_schedule(
                "test task",
                ScheduleConfig(max_depth=5, threshold=0.60, max_tokens=1024),
            )

        assert agent.loop.threshold == original_threshold
        assert agent.loop.max_depth == original_max_depth

    def test_exception_from_agent_continues(self, agent):
        call_count = 0

        def raising_run(self_loop, task, self_model, depth=0, raw_turns=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated API error")
            return make_result(self_model=self_model)

        auto = AutoMode(agent)
        with patch.object(NoesisLoop, "run", raising_run):
            records = auto.run(initial_tasks=["t1", "t2"], max_cycles=10)

        assert len(records) == 2
        assert records[0].error is not None
        assert records[1].error is None

    def test_exception_stop_on_error(self, agent):
        def always_raise(self_loop, task, self_model, depth=0, raw_turns=None):
            raise RuntimeError("always fails")

        auto = AutoMode(agent)
        with patch.object(NoesisLoop, "run", always_raise):
            with pytest.raises(RuntimeError):
                auto.run(initial_tasks=["t1"], stop_on_error=True)

    def test_summary_empty_records(self, agent):
        auto = AutoMode(agent)
        s = auto.summary()
        assert s["cycles"] == 0
        assert s["avg_depth"] == 0.0
        assert s["tasks_errored"] == 0

    def test_summary_aggregates_correctly(self, agent):
        _, records = self._run_auto(agent, ["t1", "t2", "t3"], mock_depth=2)
        auto = AutoMode.__new__(AutoMode)
        auto._records = records
        s = auto.summary()
        assert s["cycles"] == 3
        assert s["avg_depth"] == pytest.approx(2.0)
        assert s["tasks_completed"] == 3

    def test_sigint_handler_restored(self, agent):
        import signal as _signal
        original = _signal.getsignal(_signal.SIGINT)

        mock_res = make_result(self_model=agent.self_model)
        auto = AutoMode(agent)
        with patch.object(NoesisLoop, "run", return_value=mock_res):
            auto.run(initial_tasks=["t1"], max_cycles=1)

        restored = _signal.getsignal(_signal.SIGINT)
        assert restored == original

    def test_cycle_record_has_expected_fields(self, agent):
        _, records = self._run_auto(agent, ["task one"])
        r = records[0]
        assert r.task == "task one"
        assert isinstance(r.output, str)
        assert isinstance(r.depth_used, int)
        assert isinstance(r.eval_score, EvalScore)
        assert isinstance(r.schedule, ScheduleConfig)
        assert isinstance(r.self_model_snapshot, dict)
        assert r.error is None
