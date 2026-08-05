"""
PRISMBridge — connects PRISM's photonic recursive compute to LLM prompts.

Maps MetaMetaPrompt's 3-level hierarchy to NOESIS's 3-level prompt structure:
  P0 (Fibonacci seed)      → Level 0 master prompt context vector
  phi_1 (optical memory)   → SelfModel.attention_vector
  MetaGen propagate rule   → SelfModel update after each turn
"""

import sys
import os
import re
import numpy as np
from pathlib import Path

# PRISM lives as a submodule; add to path
_PRISM_DIR = Path(__file__).parent.parent / "prism"
if str(_PRISM_DIR) not in sys.path:
    sys.path.insert(0, str(_PRISM_DIR))

from recursive_prompt import MetaMetaPrompt, SynapticEmbedder  # noqa: E402


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PRISMBridge:
    """
    Translates between PRISM's numerical optical state and LLM prompt strings.

    Lifecycle:
      bridge = PRISMBridge()
      self_model = bridge.seed_self_model("NOESIS")
      system_prompt = bridge.assemble_prompt(self_model, task, depth=0)
      # ... LLM call ...
      self_model = bridge.integrate(self_model, llm_response_text, state_dict)
    """

    def __init__(self, seed_dim: int = 64, max_depth: int = 16, rng_seed: int = 42):
        self.mmp = MetaMetaPrompt(seed_dim=seed_dim, max_depth=max_depth, rng_seed=rng_seed)
        self.embedder = SynapticEmbedder()
        self._master_prompt = (_PROMPTS_DIR / "master.md").read_text()
        self._meta_template = (_PROMPTS_DIR / "meta_generator.md").read_text()
        self._task_template = (_PROMPTS_DIR / "task_cognition.md").read_text()

    # ── Seeding ───────────────────────────────────────────────────────────

    def seed_attention_vector(self) -> np.ndarray:
        """Return PRISM's Fibonacci seed P0 as the initial attention vector."""
        return self.mmp.P0.copy()

    # ── Context encoding ──────────────────────────────────────────────────

    def encode_context(self, attention_vector: np.ndarray) -> str:
        """
        Convert attention_vector → human-readable PRISM context paragraph.

        Uses SynapticEmbedder to compute MI proxy (optical clarity) and
        derives a natural-language grounding paragraph injected into Level 0.
        """
        mi = self.attention_mi(attention_vector)
        norm = float(np.linalg.norm(attention_vector))
        coherence_score = float(1.0 - np.std(attention_vector))

        dominant_idx = int(np.argmax(np.abs(attention_vector)))
        polarity = "constructive (φ=0)" if attention_vector[dominant_idx] >= 0 else "destructive (φ=π)"

        return (
            f"**Photonic State** — optical clarity (MI): {mi:.3f} | "
            f"attention norm: {norm:.4f} | coherence: {coherence_score:.3f}\n"
            f"Dominant channel: dim-{dominant_idx}, phase {polarity}. "
            f"Your attention is {'broadly distributed' if mi > 2.5 else 'sharply focused'} "
            f"across the semantic spectrum — "
            f"{'maximise coverage this turn' if mi > 2.5 else 'exploit your current focus, then widen'}."
        )

    def attention_mi(self, vec: np.ndarray) -> float:
        """Return the MI entropy proxy for the given attention vector."""
        enc = self.embedder.encode(vec.reshape(1, -1))
        return float(self.embedder.mutual_information_proxy(enc))

    def confidence_from_prism(self, attention_vector: np.ndarray) -> float:
        """
        Derive a confidence floor from the MI entropy of attention_vector.

        High MI (broad distribution) → lower prior confidence (more to explore).
        Low MI (focused distribution) → higher prior confidence (convergent).
        Maps MI ∈ [0, ~4.2] → confidence_floor ∈ [0.3, 0.75].
        """
        mi = self.attention_mi(attention_vector)
        # Sigmoid-style normalisation: MI≈0 → 0.75, MI≈4 → 0.30
        return float(np.clip(0.75 - 0.11 * mi, 0.30, 0.75))

    # ── Prompt assembly ───────────────────────────────────────────────────

    def assemble_prompt(
        self,
        self_model,  # SelfModel — imported lazily to avoid circular import
        task: str,
        depth: int = 0,
        max_depth: int = 3,
    ) -> str:
        """
        Build the 3-level NOESIS system prompt for an LLM call.

        Level 0 (static, prompt-cacheable) + Level 1 (session) + Level 2 (turn).
        """
        # Level 0
        prism_ctx = self.encode_context(self_model.attention_vector)
        level0 = self._master_prompt.replace("{prism_context}", prism_ctx)

        # Level 1
        wisdom_text = (
            "\n".join(f"- {w}" for w in self_model.session_wisdom)
            if self_model.session_wisdom
            else "- No prior turns in this session."
        )
        meta_directive = self._meta_directive(self_model, depth)
        level1 = self._meta_template.format(
            session_id=self_model.session_id,
            turn_count=len(self_model.action_history),
            coherence=self_model.coherence,
            attention_clarity=float(np.linalg.norm(self_model.attention_vector)),
            session_wisdom=wisdom_text,
            current_intent=self_model.current_intent or "not yet determined",
            arousal=self_model.arousal,
            confidence=self_model.confidence,
            meta_directive=meta_directive,
        )

        # Level 2
        recursion_ctx = self._recursion_context(self_model, depth)
        history_summary = self._action_history_summary(self_model)
        task_hash = hex(abs(hash(task)) % 0xFFFF)[2:].upper()
        level2 = self._task_template.format(
            depth=depth,
            max_depth=max_depth,
            task_fingerprint=task_hash,
            recursion_context=recursion_ctx,
            action_history_summary=history_summary,
        )

        return f"{level0}\n\n---\n\n{level1}\n\n---\n\n{level2}"

    def _meta_directive(self, self_model, depth: int) -> str:
        if depth == 0:
            return (
                "This is your first pass. Apply all five PRISM lenses fully before acting. "
                "Prefer breadth of understanding over speed of response."
            )
        return (
            f"This is recursion pass {depth}. You have already reasoned once. "
            f"Your previous confidence was {self_model.confidence:.2f}. "
            f"Focus your recursion on the lens where you were least certain. "
            f"Do not repeat what you already established — build on it."
        )

    def _recursion_context(self, self_model, depth: int) -> str:
        if depth == 0 or not self_model.action_history:
            return ""
        last = self_model.action_history[-1]
        return (
            f"> **Recursion context** (depth {depth}): "
            f"Previous pass insight: \"{last.get('insight', 'none')}\". "
            f"Intent shift: {last.get('intent_shift', 'stable')}. "
            f"Re-examine from this angle before acting."
        )

    def _action_history_summary(self, self_model) -> str:
        if not self_model.action_history:
            return ""
        recent = self_model.action_history[-3:]
        lines = [f"**Recent actions** (last {len(recent)} of {len(self_model.action_history)}):"]
        for h in recent:
            lines.append(f"- Turn {h.get('turn', '?')}: conf={h.get('confidence', '?'):.2f}")
        return "\n".join(lines)

    # ── State integration (phi_1 update rule) ────────────────────────────

    def propagate(
        self,
        attention_vector: np.ndarray,
        text_signal: str,
    ) -> np.ndarray:
        """
        Update attention_vector after a turn — mirrors PRISM's phi_1 rule:
            phi_next = tanh(0.9 * phi_prev + 0.1 * new_signal)

        Converts text_signal to a 64-dim vector via deterministic hash embedding,
        then applies the optical memory propagation.
        """
        signal = self._text_to_signal(text_signal)
        return np.tanh(0.9 * attention_vector + 0.1 * signal)

    def _text_to_signal(self, text: str) -> np.ndarray:
        """
        Deterministic text → 64-dim signal vector via hash embedding.

        Uses overlapping 4-byte windows of the UTF-8 encoding as hash seeds,
        producing a stable, low-collision representation that respects text
        semantics at a coarse level.
        """
        d = self.mmp.d
        text_bytes = text.encode("utf-8")
        # Collect up to 16 seed values from non-overlapping 4-byte windows
        seeds = []
        for i in range(0, min(len(text_bytes), 256), 16):
            chunk = text_bytes[i : i + 16]
            seeds.append(int.from_bytes(chunk.ljust(16, b"\x00"), "little") % (2**32))
        if not seeds:
            seeds = [abs(hash(text)) % (2**32)]

        rng = np.random.default_rng(seeds[0])
        raw = rng.normal(0, 1, d)
        for s in seeds[1:]:
            rng2 = np.random.default_rng(s)
            raw += 0.3 * rng2.normal(0, 1, d)

        norm = np.linalg.norm(raw)
        return np.tanh(raw / (norm + 1e-12))

    def integrate(
        self,
        attention_vector: np.ndarray,
        response_text: str,
        insight: str = "",
    ) -> np.ndarray:
        """
        Post-turn integration: propagate attention_vector through response content.
        Weights insight more heavily if provided (it's the distilled signal).
        """
        combined = f"{insight} {response_text}" if insight else response_text
        return self.propagate(attention_vector, combined)
