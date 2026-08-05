"""
PipelineInjector demo — transparent NOESIS wrapping of a standard Anthropic client.

Usage:
    ANTHROPIC_API_KEY=<key> python examples/pipeline_demo.py

Shows that the response shape is identical to a raw Anthropic call,
with a NOESIS SelfModel returned as a bonus second value.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from rich.console import Console
from rich.panel import Panel

from noesis import PipelineInjector

console = Console()

TASK = (
    "Design a minimal Python class that simulates a single coherent fiber "
    "neuron performing a dot product via IQ modulation. The class should "
    "encode weights as optical phase (0 for positive, π for negative) and "
    "decode via homodyne detection. Keep it under 40 lines."
)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Exiting.[/red]")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    console.print(Panel.fit(
        "[bold cyan]NOESIS PipelineInjector Demo[/bold cyan]\n"
        "Drop-in replacement for client.messages.create()",
        border_style="cyan",
    ))

    console.print(f"\n[bold yellow]Task:[/bold yellow] {TASK}\n")

    # Standard Anthropic call shape — with NOESIS injected
    noesis = PipelineInjector(confidence_threshold=0.80, max_recursion_depth=3)

    response, agent_state = noesis.inject(
        client=client,
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": TASK}],
        max_tokens=1024,
    )

    # response.content[0].text — identical interface to anthropic.types.Message
    console.print(Panel(
        response.content[0].text,
        title="[green]Response (same shape as raw Anthropic response)[/green]",
        border_style="green",
    ))

    console.print("\n[bold]NOESIS agent state:[/bold]")
    for k, v in agent_state.snapshot().items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")


if __name__ == "__main__":
    main()
