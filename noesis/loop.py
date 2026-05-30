"""
NoesisLoop — the recursive PERCEIVE→REFLECT→INTEND→ACT algorithm.

One loop iteration = one LLM call. If the LLM's self-reported confidence
(from <noesis_state>) falls below the threshold, the loop recurses with
an updated SelfModel. This is the core invention: adaptive-depth recursion
on the agent's own consciousness state, not on external tool results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import anthropic

from .self_model import SelfModel
from prism_bridge import PRISMBridge


# Matches a complete <noesis_state>...</noesis_state> block, OR an unclosed
# <noesis_state>... that runs to end-of-text (response truncated at max_tokens).
_STATE_RE = re.compile(
    r"<noesis_state>(.*?)(?:</noesis_state>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_FIELD_RE = re.compile(r"^(\w+)\s*:\s*(.+)$", re.MULTILINE)


@dataclass
class NoesisResult:
    output: str
    depth_used: int
    self_model: SelfModel
    raw_turns: list[str]


def _parse_noesis_state(text: str) -> dict[str, Any]:
    """Extract fields from <noesis_state>...</noesis_state> tag."""
    match = _STATE_RE.search(text)
    if not match:
        return {}
    body = match.group(1)
    result: dict[str, Any] = {}
    for m in _FIELD_RE.finditer(body):
        key, val = m.group(1).strip(), m.group(2).strip()
        if key == "confidence":
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = 0.5
        else:
            result[key] = val
    return result


def _strip_noesis_state(text: str) -> str:
    """Remove <noesis_state> block from visible output."""
    return _STATE_RE.sub("", text).strip()


class NoesisLoop:
    """
    Recursive consciousness loop for a single NOESIS agent.

    Parameters
    ----------
    client       : anthropic.Anthropic instance
    model        : LLM model identifier
    bridge       : PRISMBridge for prompt assembly and state propagation
    threshold    : confidence below which recursion is triggered (default 0.80)
    max_depth    : maximum recursion depth (default 3)
    max_tokens   : max tokens per LLM call
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        bridge: PRISMBridge,
        threshold: float = 0.80,
        max_depth: int = 3,
        max_tokens: int = 2048,
    ) -> None:
        self.client = client
        self.model = model
        self.bridge = bridge
        self.threshold = threshold
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self._turn_counter = 0

    def run(
        self,
        task: str,
        self_model: SelfModel,
        depth: int = 0,
        raw_turns: list[str] | None = None,
    ) -> NoesisResult:
        """
        Execute one recursive pass of the NOESIS loop.

        Assembles 3-level system prompt → LLM call → parse <noesis_state>
        → update SelfModel → recurse if confidence < threshold.
        """
        if raw_turns is None:
            raw_turns = []

        # Assemble the 3-level system prompt
        system = self.bridge.assemble_prompt(
            self_model, task, depth=depth, max_depth=self.max_depth
        )

        # Level 0 (master prompt) is static per session → mark for prompt caching.
        # We split system into two blocks so Anthropic can cache the static part.
        messages = [{"role": "user", "content": task}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        )

        response_text = response.content[0].text
        raw_turns.append(response_text)
        self._turn_counter += 1

        # Parse consciousness state
        state = _parse_noesis_state(response_text)
        confidence = float(state.get("confidence", 0.5))
        insight = str(state.get("insight", ""))
        intent_shift = str(state.get("intent_shift", "stable"))
        prism_signal = str(state.get("prism_signal", ""))

        # Update SelfModel
        self_model.record_turn(
            turn=self._turn_counter,
            confidence=confidence,
            insight=insight,
            intent_shift=intent_shift,
            prism_signal=prism_signal,
            output_snippet=response_text[:200],
        )
        self_model.attention_vector = self.bridge.integrate(
            self_model.attention_vector, response_text, insight
        )
        self_model.metacognitive_depth = depth

        # Decide: recurse or return
        if confidence < self.threshold and depth < self.max_depth:
            return self.run(task, self_model, depth=depth + 1, raw_turns=raw_turns)

        return NoesisResult(
            output=_strip_noesis_state(response_text),
            depth_used=depth,
            self_model=self_model,
            raw_turns=raw_turns,
        )
