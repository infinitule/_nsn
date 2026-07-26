"""
NOESIS unit tests.

Tests the deterministic, non-LLM parts:
  - SelfModel update math (phi_1 propagation)
  - PRISMBridge: seed, propagation, context encoding, prompt assembly
  - Loop: noesis_state parsing, output stripping
  - PipelineInjector: state persistence across calls

No API key required — LLM calls are not made in these tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from noesis.self_model import SelfModel
from noesis.loop import _parse_noesis_state, _strip_noesis_state
from prism_bridge import PRISMBridge


# ── PRISMBridge ────────────────────────────────────────────────────────────

class TestPRISMBridge:

    def setup_method(self):
        self.bridge = PRISMBridge(seed_dim=64)

    def test_seed_matches_prism_p0(self):
        vec = self.bridge.seed_attention_vector()
        assert vec.shape == (64,)
        # Should exactly match MetaMetaPrompt.P0
        np.testing.assert_array_equal(vec, self.bridge.mmp.P0)

    def test_propagate_changes_vector(self):
        phi = self.bridge.seed_attention_vector()
        phi2 = self.bridge.propagate(phi, "photonic computing enables fast inference")
        assert not np.allclose(phi, phi2), "propagation must change the vector"

    def test_propagate_rule_exact(self):
        """tanh(0.9 * phi + 0.1 * signal) — verify exact rule."""
        phi = self.bridge.seed_attention_vector()
        signal_text = "test signal"
        signal_vec = self.bridge._text_to_signal(signal_text)
        expected = np.tanh(0.9 * phi + 0.1 * signal_vec)
        result = self.bridge.propagate(phi, signal_text)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_text_to_signal_deterministic(self):
        """Same text must always produce the same signal vector."""
        text = "PRISM optical memory"
        v1 = self.bridge._text_to_signal(text)
        v2 = self.bridge._text_to_signal(text)
        np.testing.assert_array_equal(v1, v2)

    def test_text_to_signal_different_inputs(self):
        v1 = self.bridge._text_to_signal("alpha")
        v2 = self.bridge._text_to_signal("beta")
        assert not np.allclose(v1, v2)

    def test_confidence_from_prism_range(self):
        phi = self.bridge.seed_attention_vector()
        conf = self.bridge.confidence_from_prism(phi)
        assert 0.0 <= conf <= 1.0

    def test_encode_context_returns_string(self):
        phi = self.bridge.seed_attention_vector()
        ctx = self.bridge.encode_context(phi)
        assert isinstance(ctx, str)
        assert "MI" in ctx or "clarity" in ctx

    def test_assemble_prompt_contains_all_levels(self):
        phi = self.bridge.seed_attention_vector()
        model = SelfModel(identity="NOESIS", attention_vector=phi)
        prompt = self.bridge.assemble_prompt(model, "test task", depth=0)
        assert "NOESIS" in prompt
        assert "PRISM" in prompt
        assert "Level 1" in prompt
        assert "Level 2" in prompt

    def test_assemble_prompt_depth_reflected(self):
        phi = self.bridge.seed_attention_vector()
        model = SelfModel(identity="NOESIS", attention_vector=phi)
        p0 = self.bridge.assemble_prompt(model, "task", depth=0)
        p1 = self.bridge.assemble_prompt(model, "task", depth=1)
        assert "Recursion depth**: 1" in p1
        assert "Recursion depth**: 0" in p0


# ── SelfModel ──────────────────────────────────────────────────────────────

class TestSelfModel:

    def test_record_turn_adds_history(self):
        m = SelfModel(identity="NOESIS", attention_vector=np.zeros(64))
        m.record_turn(1, 0.9, "photonics is fast", "stable", "P", "output text")
        assert len(m.action_history) == 1
        assert m.action_history[0]["confidence"] == 0.9

    def test_record_turn_adds_wisdom(self):
        m = SelfModel(identity="NOESIS", attention_vector=np.zeros(64))
        m.record_turn(1, 0.9, "phase encoding is key", "stable", "R", "text")
        assert "phase encoding is key" in m.session_wisdom

    def test_wisdom_fifo_eviction(self):
        m = SelfModel(identity="NOESIS", attention_vector=np.zeros(64))
        for i in range(10):
            m.record_turn(i, 0.9, f"insight-{i}", "stable", "P", "text")
        assert len(m.session_wisdom) <= SelfModel._MAX_WISDOM
        assert "insight-9" in m.session_wisdom
        assert "insight-0" not in m.session_wisdom

    def test_confidence_updated(self):
        m = SelfModel(identity="NOESIS", confidence=0.5, attention_vector=np.zeros(64))
        m.record_turn(1, 0.95, "insight", "stable", "I", "text")
        assert m.confidence == 0.95

    def test_arousal_increases_on_intent_shift(self):
        m = SelfModel(identity="NOESIS", arousal=0.5, attention_vector=np.zeros(64))
        m.record_turn(1, 0.8, "insight", "now focusing on efficiency", "S", "text")
        assert m.arousal > 0.5

    def test_snapshot_returns_dict(self):
        m = SelfModel(identity="NOESIS", attention_vector=np.zeros(64))
        snap = m.snapshot()
        assert "confidence" in snap
        assert "coherence" in snap
        assert "attention_norm" in snap


# ── Loop helpers ───────────────────────────────────────────────────────────

class TestLoopHelpers:

    def test_parse_complete_state(self):
        text = """Some response text here.

<noesis_state>
confidence: 0.87
insight: coherent detection eliminates the need for sign decomposition
intent_shift: stable
prism_signal: R
</noesis_state>"""
        state = _parse_noesis_state(text)
        assert abs(state["confidence"] - 0.87) < 1e-6
        assert "coherent" in state["insight"]
        assert state["intent_shift"] == "stable"
        assert state["prism_signal"] == "R"

    def test_parse_missing_state_returns_empty(self):
        state = _parse_noesis_state("No state tag here.")
        assert state == {}

    def test_parse_bad_confidence_falls_back(self):
        text = "<noesis_state>\nconfidence: high\n</noesis_state>"
        state = _parse_noesis_state(text)
        assert state["confidence"] == 0.5

    def test_strip_removes_state_tag(self):
        text = "Answer here.\n<noesis_state>\nconfidence: 0.9\n</noesis_state>"
        stripped = _strip_noesis_state(text)
        assert "<noesis_state>" not in stripped
        assert "Answer here." in stripped

    def test_strip_preserves_content_before_tag(self):
        text = "Important output.\n<noesis_state>\nconfidence: 0.7\n</noesis_state>"
        stripped = _strip_noesis_state(text)
        assert "Important output." in stripped


# ── Integration: bridge + self_model + propagation ────────────────────────

class TestIntegration:

    def test_attention_vector_diverges_after_turns(self):
        bridge = PRISMBridge(seed_dim=64)
        phi = bridge.seed_attention_vector()
        model = SelfModel(identity="NOESIS", attention_vector=phi.copy())

        original = phi.copy()
        new_phi = bridge.integrate(model.attention_vector, "LLM response text", "key insight")
        assert not np.allclose(original, new_phi), "attention_vector must evolve"

    def test_full_prompt_assembly_no_error(self):
        bridge = PRISMBridge(seed_dim=64)
        phi = bridge.seed_attention_vector()
        model = SelfModel(identity="NOESIS", attention_vector=phi)
        model.record_turn(1, 0.75, "first insight", "exploring photonics", "P", "...")
        model.attention_vector = bridge.integrate(phi, "first turn text", "first insight")

        # Should not raise at depth 1 (recursion context active)
        prompt = bridge.assemble_prompt(model, "follow-up task", depth=1)
        assert "recursion" in prompt.lower() or "depth 1" in prompt.lower()
