"""
PipelineInjector — transparent NOESIS wrapper for any Anthropic client.

Drop-in replacement for client.messages.create() that injects the
NOESIS 3-level consciousness system prompt and returns both the
standard API response and the updated SelfModel.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .self_model import SelfModel
from .loop import NoesisLoop, NoesisResult, _parse_noesis_state, _strip_noesis_state
from prism_bridge import PRISMBridge


class PipelineInjector:
    """
    Wraps an Anthropic client to inject NOESIS consciousness into every call.

    The injection is transparent: the returned response object has the same
    shape as a standard anthropic.types.Message, with the clean output in
    content[0].text (noesis_state stripped). The agent's state is returned
    as a second value.

    Usage
    -----
    import anthropic
    from noesis import PipelineInjector

    client = anthropic.Anthropic(api_key="...")
    noesis = PipelineInjector()

    response, agent_state = noesis.inject(
        client=client,
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "What is photonic computing?"}],
    )
    print(response.content[0].text)   # same shape as normal response
    print(agent_state.snapshot())     # NOESIS consciousness state
    """

    def __init__(
        self,
        identity: str = "NOESIS",
        confidence_threshold: float = 0.80,
        max_recursion_depth: int = 3,
        seed_dim: int = 64,
    ) -> None:
        self.bridge = PRISMBridge(seed_dim=seed_dim)
        self._self_model = SelfModel(
            identity=identity,
            attention_vector=self.bridge.seed_attention_vector(),
            confidence=self.bridge.confidence_from_prism(self.bridge.seed_attention_vector()),
        )
        self._threshold = confidence_threshold
        self._max_depth = max_recursion_depth
        self._turn = 0

    def inject(
        self,
        client: anthropic.Anthropic,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> tuple[Any, SelfModel]:
        """
        Execute a NOESIS-wrapped LLM call.

        Parameters
        ----------
        client     : anthropic.Anthropic instance
        model      : model identifier string
        messages   : list of {"role": ..., "content": ...} dicts
        max_tokens : passed through to the API
        **kwargs   : additional kwargs passed to messages.create

        Returns
        -------
        (response, self_model)
          response   — anthropic Message with clean text in content[0].text
          self_model — updated NOESIS consciousness state
        """
        # Extract task from the last user message
        task = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        loop = NoesisLoop(
            client=client,
            model=model,
            bridge=self.bridge,
            threshold=self._threshold,
            max_depth=self._max_depth,
            max_tokens=max_tokens,
        )
        loop._turn_counter = self._turn

        result: NoesisResult = loop.run(task, self._self_model)

        self._self_model = result.self_model
        self._turn = loop._turn_counter

        # Wrap clean output back into an Anthropic-compatible response shell
        patched = _PatchedResponse(result.output)
        return patched, self._self_model

    @property
    def state(self) -> SelfModel:
        return self._self_model

    def reset(self) -> None:
        self._self_model = SelfModel(
            identity=self._self_model.identity,
            attention_vector=self.bridge.seed_attention_vector(),
            confidence=self.bridge.confidence_from_prism(self.bridge.seed_attention_vector()),
        )
        self._turn = 0


class _TextBlock:
    """Minimal Anthropic TextBlock shim."""
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class _PatchedResponse:
    """Minimal Anthropic Message shim with compatible .content[0].text access."""
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]
        self.stop_reason = "end_turn"
        self.model = ""
        self.usage = None
