"""
Recursion scheduler — adapts max_depth and threshold per task.

Uses PRISM's MI entropy of the current attention_vector to estimate
how much recursion depth will be productive (high MI = broad attention
= more exploration needed). Task word count and question density adjust
the confidence threshold and token budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .self_model import SelfModel
    from prism_bridge import PRISMBridge

_BASE_MAX_DEPTH: int = 3
_BASE_THRESHOLD: float = 0.80
_BASE_MAX_TOKENS: int = 2048
_MAX_TOKENS_CAP: int = 4096


class ScheduleConfig(NamedTuple):
    max_depth: int
    threshold: float
    max_tokens: int


def schedule_config(
    task: str,
    self_model: "SelfModel",
    bridge: "PRISMBridge",
    *,
    base_max_depth: int = _BASE_MAX_DEPTH,
    base_threshold: float = _BASE_THRESHOLD,
    base_max_tokens: int = _BASE_MAX_TOKENS,
) -> ScheduleConfig:
    """
    Derive a ScheduleConfig for the given task.

    MI from bridge.attention_mi() — single source of truth, no duplication.
    task_complexity capped at 1.5 to prevent question-mark flooding the
    token budget or pushing threshold below the allowed floor.
    """
    word_count = len(task.split())
    question_count = min(task.count("?"), 5)
    task_complexity = min(1.5, min(1.0, word_count / 50) + 0.2 * question_count)

    mi = bridge.attention_mi(self_model.attention_vector)
    depth_bonus = int(np.clip(mi / 2.0, 0, 2))
    max_depth = base_max_depth + depth_bonus

    threshold = float(np.clip(base_threshold - 0.05 * task_complexity, 0.60, 0.92))
    max_tokens = min(base_max_tokens + int(task_complexity * 1024), _MAX_TOKENS_CAP)

    return ScheduleConfig(max_depth=max_depth, threshold=threshold, max_tokens=max_tokens)
