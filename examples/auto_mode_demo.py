"""
AutoMode demo — 3 seeded tasks + template auto-generation, 6 cycles, persistence.

Usage:
    ANTHROPIC_API_KEY=<key> python examples/auto_mode_demo.py
    ANTHROPIC_API_KEY=<key> python examples/auto_mode_demo.py --resume

On the first run, the agent works through 3 photonic-computing tasks and
auto-generates 3 more from its accumulated wisdom. The session is checkpointed
to /tmp/noesis_auto.json after each cycle.

On --resume, the saved session is loaded and additional cycles run from the
agent's prior state (attention_vector, session_wisdom, action_history intact).
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from noesis import NOESISAgent
from noesis.auto_mode import AutoMode

PERSIST_PATH = "/tmp/noesis_auto.json"

SEED_TASKS = [
    "What is the fundamental advantage of coherent photonic neural networks "
    "over electronic ones for matrix-vector multiplication?",
    "Explain how PRISM's three-level MetaMetaPrompt (Fibonacci seed → "
    "meta-generator → weight matrix) mirrors a living organism's "
    "instinct → learning → cognition hierarchy.",
    "What are the two most likely engineering barriers to deploying photonic "
    "neural networks in production AI inference systems by 2030?",
]

console = Console()


def on_cycle(record):
    status = "✓" if record.error is None else "✗"
    depth_str = f"depth={record.depth_used}" + (" [deepened]" if record.deepened else "")
    conf = record.self_model_snapshot.get("confidence", "?")
    console.print(
        f"  [{status}] Cycle {record.cycle}  {depth_str}  conf={conf}  "
        f"task: {record.task[:60]}..."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Exiting.[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]NOESIS AutoMode — Self-Driving Recursive Agent[/bold cyan]\n"
        "Continuous loop · PRISM MI scheduling · Session persistence",
        border_style="cyan",
    ))

    agent = NOESISAgent(
        api_key=api_key,
        model="claude-opus-4-8",
        confidence_threshold=0.80,
        max_recursion_depth=3,
    )

    auto = AutoMode(agent, auto_generate=True, on_cycle=on_cycle)

    initial = [] if args.resume else SEED_TASKS
    action = "Resuming" if args.resume else "Starting"
    console.print(f"\n[bold]{action}[/bold] auto mode session  persist_path={PERSIST_PATH}\n")

    records = auto.run(
        initial_tasks=initial,
        max_cycles=6,
        persist_path=PERSIST_PATH,
    )

    console.print()
    summary = auto.summary()
    table = Table(title="AutoMode Summary", show_header=False, border_style="cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    for k, v in summary.items():
        table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    console.print(table)

    if agent.self_model.session_wisdom:
        console.print("\n[bold]Final session wisdom:[/bold]")
        for w in agent.self_model.session_wisdom:
            console.print(f"  • {w}")

    console.print(f"\n[dim]Session saved to {PERSIST_PATH}. Run with --resume to continue.[/dim]")


if __name__ == "__main__":
    main()
