"""
SelfModel — the persistent consciousness state of a NOESIS agent.

Analogous to PRISM's phi_1 optical memory, but richer: encodes not just
an attention vector but the full cognitive state across a session.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SelfModel:
    """
    Persistent consciousness state. Updated after every turn.

    attention_vector: PRISM phi_1 analog — 64-dim float array.
      Initialized from MetaMetaPrompt.P0 (Fibonacci seed).
      Updated each turn via: tanh(0.9 * phi + 0.1 * signal).
      Encodes the accumulated semantic focus of the session.
    """

    identity: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    current_intent: str = ""
    confidence: float = 0.5
    attention_vector: np.ndarray = field(default_factory=lambda: np.zeros(64))
    session_wisdom: list[str] = field(default_factory=list)
    metacognitive_depth: int = 0
    action_history: list[dict[str, Any]] = field(default_factory=list)
    arousal: float = 0.5
    coherence: float = 1.0

    _MAX_WISDOM = 8

    def record_turn(
        self,
        turn: int,
        confidence: float,
        insight: str,
        intent_shift: str,
        prism_signal: str,
        output_snippet: str,
    ) -> None:
        """Record the outcome of one turn into action_history and session_wisdom."""
        self.action_history.append(
            {
                "turn": turn,
                "confidence": confidence,
                "insight": insight,
                "intent_shift": intent_shift,
                "prism_signal": prism_signal,
                "output_snippet": output_snippet[:120],
            }
        )
        if insight and insight.lower() not in ("none", "n/a", ""):
            self.session_wisdom.append(insight)
            # FIFO eviction
            if len(self.session_wisdom) > self._MAX_WISDOM:
                self.session_wisdom = self.session_wisdom[-self._MAX_WISDOM :]

        # Update coherence: exponential moving average of confidence
        self.coherence = float(np.tanh(0.85 * np.arctanh(np.clip(self.coherence, 0.01, 0.99))
                                        + 0.15 * confidence))
        self.confidence = confidence
        if intent_shift and intent_shift.lower() not in ("stable", ""):
            self.current_intent = intent_shift
            self.arousal = min(1.0, self.arousal + 0.1)
        else:
            self.arousal = max(0.1, self.arousal - 0.05)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": len(self.action_history),
            "confidence": round(self.confidence, 3),
            "coherence": round(self.coherence, 3),
            "arousal": round(self.arousal, 3),
            "wisdom_count": len(self.session_wisdom),
            "attention_norm": round(float(np.linalg.norm(self.attention_vector)), 4),
        }
