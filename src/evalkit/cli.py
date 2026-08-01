# CLI entry point
import asyncio
from datetime import datetime
from pathlib import Path

import typer

from evalkit.client import (
    CachingCompletionProvider,
    CompletionProvider,
    OpenAICompletionProvider,
)
from evalkit.loaders import TaskLoadError, load_task
from evalkit.runner import run_task

app = typer.Typer()


@app.command()
def run(
    task_path: Path = typer.Argument(
        Path("tasks/ticket_extraction.yaml"), help="Path to a task YAML file."
    ),
    model: str = typer.Option("gpt-5.4-mini", help="OpenAI model to run completions against."),
    concurrency: int = typer.Option(5, help="Max concurrent completions in flight."),
    output_dir: Path = typer.Option(
        Path("runs"), help="Directory to write run results under."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass the completion cache and always call the API."
    ),
) -> None:
    try:
        task = load_task(task_path)
    except (FileNotFoundError, TaskLoadError) as e:
        typer.echo(f"Error loading task: {e}", err=True)
        raise typer.Exit(code=1) from e

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{task.name}-{timestamp}"

    # --no-cache bypasses the cache entirely (no reads, no writes)
    provider: CompletionProvider = OpenAICompletionProvider()
    if not no_cache:
        provider = CachingCompletionProvider(provider)
    run = asyncio.run(
        run_task(task, provider, model=model, output_dir=run_dir, concurrency=concurrency)
    )

    total = len(task.test_cases)
    typer.echo(f"Run {run.run_id}: completed {len(run.completions)}/{total} cases for '{task.name}'.")
    typer.echo(f"Results written to {run_dir}")
    for result in run.completions:
        typer.echo(
            f"  {result.case_id}: {result.output_tokens} out tokens, "
            f"{result.latency_ms:.0f}ms"
        )


def main() -> None:
    app()
