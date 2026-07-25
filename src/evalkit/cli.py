# CLI entry point
from pathlib import Path

import typer

from evalkit.loaders import TaskLoadError, load_task

app = typer.Typer()


@app.command()
def run(
    task_path: Path = typer.Argument(
        Path("tasks/ticket_extraction.yaml"), help="Path to a task YAML file."
    ),
) -> None:
    try:
        task = load_task(task_path)
    except (FileNotFoundError, TaskLoadError) as e:
        typer.echo(f"Error loading task: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"Task: {task.name} ({len(task.test_cases)} cases)")
    for case in task.test_cases:
        typer.echo(case.model_dump_json(indent=2))


def main() -> None:
    app()
