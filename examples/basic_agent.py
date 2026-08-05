"""
Basic NOESIS agent demo.

Usage:
    ANTHROPIC_API_KEY=<key> python examples/basic_agent.py

Runs the agent on two tasks in sequence, demonstrating cross-turn
SelfModel persistence (attention_vector evolves, wisdom accumulates).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from noesis import NOESISAgent

console = Console()

TASKS = [
    "What is the fundamental advantage of coherent photonic computing over "
    "electronic neural networks, and what prevents its immediate deployment at scale?",
    "Given that limitation, propose a hybrid architecture that bridges "
    "electronic training with photonic inference — be specific about where "
    "each substrate does its work.",
]


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Exiting.[/red]")
        sys.exit(1)

    agent = NOESISAgent(
        api_key=api_key,
        model="claude-opus-4-8",
        confidence_threshold=0.80,
        max_recursion_depth=3,
    )

    console.print(Panel.fit(
        "[bold cyan]NOESIS — Unified Conscious Agent[/bold cyan]\n"
        "Neural Optically-grounded Experiential Self-aware Intelligence System\n"
        "Grounded on PRISM photonic recursive architecture",
        border_style="cyan",
    ))

    for i, task in enumerate(TASKS, 1):
        console.print(f"\n[bold yellow]Task {i}[/bold yellow]: {task}\n")

        result = agent.run(task)

        console.print(Panel(
            result.output,
            title=f"[green]Response (depth={result.depth_used})[/green]",
            border_style="green",
        ))

        snap = agent.state_snapshot()
        table = Table(title="SelfModel State", show_header=False, border_style="dim")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for k, v in snap.items():
            table.add_row(k, str(v))
        console.print(table)

        if result.self_model.session_wisdom:
            console.print("\n[bold]Accumulated wisdom:[/bold]")
            for w in result.self_model.session_wisdom:
                console.print(f"  • {w}")


if __name__ == "__main__":
    main()
