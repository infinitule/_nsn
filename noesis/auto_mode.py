"""
AutoMode — self-driving NOESIS loop.

Runs tasks continuously from a queue, self-evaluates each result, applies
adaptive scheduling via PRISM MI entropy, and optionally generates its own
next tasks from accumulated session wisdom.

Sessions can be interrupted (Ctrl+C) and resumed from a JSON checkpoint.
The agent's persistent loop settings are never mutated — each cycle uses a
temporary NoesisLoop configured per the ScheduleConfig for that task.
"""

from __future__ import annotations

import random
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .agent import NOESISAgent
from .evaluator import EvalScore, score as _eval_score
from .loop import NoesisLoop, NoesisResult
from .persistence import exists, load, save
from .scheduler import ScheduleConfig, schedule_config
from .self_model import SelfModel


_GENERATION_TEMPLATES = [
    "Explore the following insight further: {}",
    "What are the deeper implications of: {}",
    "How does this connect to broader patterns: {}",
    "Critically examine the assumption behind: {}",
]


@dataclass
class CycleRecord:
    """Compact record of one AutoMode cycle. Stores a snapshot, not the raw LLM turns."""

    cycle: int
    task: str
    output: str
    depth_used: int
    self_model_snapshot: dict
    eval_score: EvalScore
    schedule: ScheduleConfig
    deepened: bool
    error: str | None = None


class AutoMode:
    """
    Self-driving NOESIS loop.

    Usage
    -----
    auto = AutoMode(agent, auto_generate=True)
    records = auto.run(
        initial_tasks=["What is photonic computing?"],
        max_cycles=10,
        persist_path="/tmp/noesis_session.json",
    )
    print(auto.summary())
    """

    def __init__(
        self,
        agent: NOESISAgent,
        auto_generate: bool = False,
        on_cycle: Callable[[CycleRecord], None] | None = None,
    ) -> None:
        self._agent = agent
        self._auto_generate = auto_generate
        self._on_cycle = on_cycle
        self._records: list[CycleRecord] = []

    def run(
        self,
        initial_tasks: list[str],
        max_cycles: int = 10,
        persist_path: str | Path | None = None,
        stop_on_error: bool = False,
    ) -> list[CycleRecord]:
        """
        Execute the self-driving loop.

        Loads a persisted checkpoint at persist_path if it exists, restoring
        the SelfModel and prior cycle count for correct budget enforcement.
        Runs until the task queue empties, max_cycles is reached, or Ctrl+C.
        """
        prior_cycles = 0
        if persist_path and exists(persist_path):
            loaded_model, prior_cycles = load(persist_path)
            self._agent.self_model = loaded_model

        task_queue: list[str] = list(initial_tasks)

        _stop = False

        def _sigint_handler(sig, frame: object) -> None:
            nonlocal _stop
            _stop = True

        _old_handler = signal.signal(signal.SIGINT, _sigint_handler)

        try:
            cycle = 0
            while not _stop and (prior_cycles + cycle) < max_cycles:
                task = self._next_task(task_queue, cycle)
                if task is None:
                    break

                config = schedule_config(task, self._agent.self_model, self._agent.bridge)
                error: str | None = None
                deepened = False
                result: NoesisResult | None = None

                try:
                    result = self._run_with_schedule(task, config)
                except Exception as exc:
                    error = str(exc)
                    if stop_on_error:
                        raise

                if error is not None or result is None:
                    record = CycleRecord(
                        cycle=cycle,
                        task=task,
                        output="",
                        depth_used=0,
                        self_model_snapshot=self._agent.state_snapshot(),
                        eval_score=_eval_score(
                            task, "", 0.0, self._agent.self_model,
                            threshold=config.threshold,
                            max_depth=config.max_depth,
                        ),
                        schedule=config,
                        deepened=False,
                        error=error,
                    )
                else:
                    eval_result = _eval_score(
                        task,
                        result.output,
                        result.self_model.confidence,
                        result.self_model,
                        threshold=config.threshold,
                        max_depth=config.max_depth,
                    )

                    if eval_result.depth_worthy:
                        deepened_config = ScheduleConfig(
                            max_depth=config.max_depth + 1,
                            threshold=config.threshold,
                            max_tokens=config.max_tokens,
                        )
                        try:
                            result = self._run_with_schedule(task, deepened_config)
                            deepened = True
                        except Exception:
                            pass  # accept first result if the deepen run fails

                    record = CycleRecord(
                        cycle=cycle,
                        task=task,
                        output=result.output,
                        depth_used=result.depth_used,
                        self_model_snapshot=result.self_model.snapshot(),
                        eval_score=eval_result,
                        schedule=config,
                        deepened=deepened,
                    )

                self._records.append(record)

                if persist_path:
                    save(
                        self._agent.self_model,
                        persist_path,
                        cycles_completed=prior_cycles + cycle + 1,
                    )

                if self._on_cycle:
                    self._on_cycle(record)

                cycle += 1

        finally:
            signal.signal(signal.SIGINT, _old_handler)

        return list(self._records)

    def _run_with_schedule(self, task: str, config: ScheduleConfig) -> NoesisResult:
        """
        Run one task via a temporary NoesisLoop configured per ScheduleConfig.

        Never mutates agent.loop — constructs a fresh loop for each call.
        Forwards and restores _turn_counter so session continuity is preserved.
        """
        temp_loop = NoesisLoop(
            client=self._agent.client,
            model=self._agent.model,
            bridge=self._agent.bridge,
            threshold=config.threshold,
            max_depth=config.max_depth,
            max_tokens=config.max_tokens,
        )
        temp_loop._turn_counter = self._agent.loop._turn_counter
        result = temp_loop.run(task, self._agent.self_model)
        self._agent.loop._turn_counter = temp_loop._turn_counter
        self._agent.self_model = result.self_model
        return result

    def _next_task(self, queue: list[str], cycle: int) -> str | None:
        """
        Pop from queue; generate from session wisdom if queue is empty and
        auto_generate is enabled. Returns None to stop the loop.
        """
        if queue:
            return queue.pop(0)
        if self._auto_generate:
            wisdom = self._agent.self_model.session_wisdom
            if not wisdom:
                return None
            template = _GENERATION_TEMPLATES[cycle % len(_GENERATION_TEMPLATES)]
            return template.format(random.choice(wisdom))
        return None

    @property
    def records(self) -> list[CycleRecord]:
        return list(self._records)

    def summary(self) -> dict:
        """Return aggregate statistics across all completed cycles."""
        records = self._records
        if not records:
            return {
                "cycles": 0,
                "avg_depth": 0.0,
                "avg_completion": 0.0,
                "max_depth_used": 0,
                "tasks_completed": 0,
                "tasks_errored": 0,
            }
        completed = [r for r in records if r.error is None]
        return {
            "cycles": len(records),
            "avg_depth": (
                sum(r.depth_used for r in completed) / len(completed)
                if completed else 0.0
            ),
            "avg_completion": (
                sum(r.eval_score.task_completion for r in completed) / len(completed)
                if completed else 0.0
            ),
            "max_depth_used": max((r.depth_used for r in completed), default=0),
            "tasks_completed": len(completed),
            "tasks_errored": len(records) - len(completed),
        }
