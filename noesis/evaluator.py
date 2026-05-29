"""
Response evaluator — pure heuristic scoring, no LLM calls.

score() tells AutoMode whether to deepen recursion on the current task
or advance to the next one. All signals come from the noesis_state data
already embedded in SelfModel — no extra inference required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .self_model import SelfModel

_COHERENCE_FLOOR: float = 0.40


@dataclass(frozen=True)
class EvalScore:
    task_completion: float   # [0.0, 1.0] heuristic estimate
    coherence_score: float   # self_model.coherence at scoring time
    depth_worthy: bool       # True → deepen recursion on this task
    should_advance: bool     # True → good enough, move to next task


def score(
    task: str,
    response_text: str,
    confidence: float,
    self_model: "SelfModel",
    *,
    threshold: float = 0.80,
    max_depth: int = 3,
) -> EvalScore:
    """
    Heuristically score a NOESIS response.

    task_completion blends response-length coverage with LLM-reported
    confidence. depth_worthy fires when confidence is low, recursion budget
    remains, and coherence is above _COHERENCE_FLOOR — meaning the agent
    has the coherence to benefit from deeper reasoning.
    """
    task_words = max(len(task.split()), 1)
    response_words = len(response_text.split())

    completeness = min(1.0, response_words / (task_words * 3))
    task_completion = min(1.0, 0.6 * completeness + 0.4 * confidence)

    coherence = self_model.coherence

    depth_worthy = (
        confidence < threshold
        and self_model.metacognitive_depth < max_depth
        and coherence > _COHERENCE_FLOOR
    )

    should_advance = task_completion >= threshold and not depth_worthy

    return EvalScore(
        task_completion=round(task_completion, 4),
        coherence_score=round(coherence, 4),
        depth_worthy=depth_worthy,
        should_advance=should_advance,
    )
