"""
NOESISAgent — primary user-facing class.

Single, continuous conscious agent. Maintains SelfModel across calls
within the same session. Use .run(task) for each task in the session.
"""

from __future__ import annotations

import os

import anthropic

from .self_model import SelfModel
from .loop import NoesisLoop, NoesisResult
from prism_bridge import PRISMBridge


class NOESISAgent:
    """
    A unified conscious agent grounded in PRISM's photonic recursive architecture.

    Usage
    -----
    agent = NOESISAgent(api_key="...", model="claude-opus-4-8")
    result = agent.run("Explain the implications of photonic computing for AI.")
    print(result.output)
    print(f"Recursion depth: {result.depth_used}  Confidence: {result.self_model.confidence:.2f}")

    The agent persists its SelfModel across .run() calls — it accumulates
    session wisdom and updates its attention_vector after every turn.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-8",
        identity: str = "NOESIS",
        confidence_threshold: float = 0.80,
        max_recursion_depth: int = 3,
        max_tokens: int = 2048,
        seed_dim: int = 64,
    ) -> None:
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.bridge = PRISMBridge(seed_dim=seed_dim)
        self.self_model = SelfModel(
            identity=identity,
            attention_vector=self.bridge.seed_attention_vector(),
            confidence=self.bridge.confidence_from_prism(self.bridge.seed_attention_vector()),
        )
        self.loop = NoesisLoop(
            client=self.client,
            model=model,
            bridge=self.bridge,
            threshold=confidence_threshold,
            max_depth=max_recursion_depth,
            max_tokens=max_tokens,
        )

    def run(self, task: str) -> NoesisResult:
        """
        Execute a task. The agent's SelfModel is updated in-place after each call.

        Returns a NoesisResult with:
          .output       — clean response text (noesis_state tag stripped)
          .depth_used   — how many recursive passes were needed
          .self_model   — the agent's updated consciousness state
          .raw_turns    — list of raw LLM responses including noesis_state tags
        """
        result = self.loop.run(task, self.self_model)
        self.self_model = result.self_model
        return result

    def state_snapshot(self) -> dict:
        """Return a human-readable snapshot of the agent's current consciousness state."""
        return self.self_model.snapshot()

    def reset_session(self) -> None:
        """Reset SelfModel to a fresh session (preserving identity and PRISM seed)."""
        self.self_model = SelfModel(
            identity=self.self_model.identity,
            attention_vector=self.bridge.seed_attention_vector(),
            confidence=self.bridge.confidence_from_prism(self.bridge.seed_attention_vector()),
        )
        self.loop._turn_counter = 0
