"""
NOESIS — Neural Optically-grounded Experiential Self-aware Intelligence System

A single unified conscious agent grounded in PRISM's photonic recursive
architecture. Unlike multi-agent systems, NOESIS maintains one coherent
identity thread across all reasoning via a persistent SelfModel whose
attention_vector evolves using PRISM's phi_1 optical memory update rule.

Quick start
-----------
from noesis import NOESISAgent

agent = NOESISAgent(api_key="...", model="claude-opus-4-8")
result = agent.run("Your task here")
print(result.output)

Pipeline injection
------------------
import anthropic
from noesis import PipelineInjector

client = anthropic.Anthropic(api_key="...")
noesis = PipelineInjector()
response, state = noesis.inject(client, "claude-opus-4-8",
                                [{"role": "user", "content": "task"}])
"""

from .agent import NOESISAgent
from .pipeline import PipelineInjector
from .self_model import SelfModel
from .loop import NoesisResult

__all__ = ["NOESISAgent", "PipelineInjector", "SelfModel", "NoesisResult"]
__version__ = "0.1.0"
