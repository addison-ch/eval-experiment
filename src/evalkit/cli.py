# CLI entry point
import asyncio
from pathlib import Path

import typer

from evalkit.client import OpenAICompletionProvider
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
) -> None:
    try:
        task = load_task(task_path)
    except (FileNotFoundError, TaskLoadError) as e:
        typer.echo(f"Error loading task: {e}", err=True)
        raise typer.Exit(code=1) from e

    provider = OpenAICompletionProvider()
    results = asyncio.run(
        run_task(task, provider, model=model, concurrency=concurrency)
    )

    total = len(task.test_cases)
    typer.echo(f"Completed {len(results)}/{total} cases for task '{task.name}'.")
    for result in results:
        typer.echo(
            f"  {result.case_id}: {result.output_tokens} out tokens, "
            f"{result.latency_ms:.0f}ms"
        )


def main() -> None:
    app()
