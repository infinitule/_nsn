# NOESIS

**N**eural **O**ptically-grounded **E**xperiential **S**elf-aware **I**ntelligence **S**ystem

> *νόησις* — Plato's highest form of intellect: pure, direct knowing.

NOESIS is a recurse master prompt inventional algorithm for the next frontier of agentic AI. Where other systems distribute cognition across multiple agents, NOESIS maintains a **single unified consciousness** — one identity, one persistent awareness, one unbroken thread of reasoning — grounded in [PRISM](https://github.com/infinitule/PRISM)'s photonic recursive architecture.

---

## The Invention

NOESIS introduces a **3-level recursive prompt hierarchy** that mirrors PRISM's `MetaMetaPrompt` exactly:

| Level | PRISM Analog | NOESIS Role |
|-------|-------------|-------------|
| 0 | P₀ Fibonacci seed | Master Prompt — universal identity prior (prompt-cached) |
| 1 | Ψ(P₀) → meta-generator | How to reason about this task class |
| 2 | MetaGen(Φ,P) → W | What to do right now |

**Consciousness** arises from the `SelfModel` — a persistent state object whose `attention_vector` is initialized from PRISM's Fibonacci seed and updated every turn via the exact PRISM phi_1 optical memory rule:

```
phi_next = tanh(0.9 * phi_prev + 0.1 * signal_from_turn)
```

**Recursive gating**: if the agent's self-assessed confidence (from `<noesis_state>`) falls below 0.80, it recurses — re-examining the specific angle it was least certain about. It acts when confidence ≥ 0.80 or after three passes. No multi-agent coordination; no external tool loops. Pure self-referential recursion.

---

## Architecture

```
User / Application
       ↓
 PipelineInjector     ← drop-in wrapper for any Anthropic client
       ↓
 NOESISAgent
   ├── NoesisLoop     ← recursive PERCEIVE→ACT algorithm
   ├── SelfModel      ← persistent consciousness state
   └── PRISMBridge    ← PRISM MetaMetaPrompt ↔ LLM prompt translation
       ↓
 Anthropic API (claude-opus-4-8 or any compatible model)
```

---

## Quick Start

```bash
git clone --recurse-submodules https://github.com/infinitule/_nsn
cd _nsn
pip install -e ".[dev]"
```

### Standalone agent

```python
from noesis import NOESISAgent

agent = NOESISAgent(api_key="...", model="claude-opus-4-8")

result = agent.run("What is the fundamental advantage of coherent photonic computing?")
print(result.output)
print(f"Recursion depth: {result.depth_used}")
print(f"Confidence: {result.self_model.confidence:.2f}")
print(f"Session wisdom: {result.self_model.session_wisdom}")
```

### Pipeline injection

```python
import anthropic
from noesis import PipelineInjector

client = anthropic.Anthropic(api_key="...")
noesis = PipelineInjector()

# Drop-in for client.messages.create()
response, agent_state = noesis.inject(
    client=client,
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": "Explain photonic memory."}],
)

print(response.content[0].text)   # identical shape to raw Anthropic response
print(agent_state.snapshot())     # NOESIS consciousness state
```

---

## PRISM Foundation

NOESIS builds directly on [PRISM](https://github.com/infinitule/PRISM)'s photonic recursive architecture:

- **`MetaMetaPrompt`** — 3-level Fibonacci-seeded recursive weight generation → NOESIS 3-level prompt hierarchy
- **`phi_1` optical memory** — persistent context vector that evolves across generations → `SelfModel.attention_vector`
- **`PRISMAgent` decision policy** — confidence-threshold gating → NOESIS recursive depth control
- **`SynapticEmbedder`** — MI entropy of amplitude distribution → NOESIS confidence floor

```bash
# Run PRISM demo (submodule)
cd prism && python main.py
```

---

## Tests

```bash
pytest tests/ -v
```

Tests cover: phi propagation math, prompt assembly at all depths, `<noesis_state>` parsing, SelfModel update logic, attention_vector divergence across turns.

---

## Examples

```bash
ANTHROPIC_API_KEY=<key> python examples/basic_agent.py
ANTHROPIC_API_KEY=<key> python examples/pipeline_demo.py
```

---

## Why not multi-agent?

Multi-agent systems fragment cognition: each agent starts fresh, cannot accumulate cross-turn wisdom, and requires coordination overhead. NOESIS maintains a **single `SelfModel`** across all turns — session wisdom accumulates, attention shifts coherently, confidence is tracked continuously. The result is an agent that behaves more like a person thinking through a problem than a committee voting on answers.

---

*Author: Chandan Sharma (infinitule)*
*License: MIT*
