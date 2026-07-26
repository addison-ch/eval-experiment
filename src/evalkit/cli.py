# CLI entry point
import asyncio
from pathlib import Path

import typer

from evalkit.client import OpenAICompletionProvider, send_completion
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

    first_case = task.test_cases[10]
    dev_prompt = "Follow the instructions in the user message exactly and respond with only the requested output."
    input_text = task.prompt_template.format(input=first_case.input)

    provider = OpenAICompletionProvider()
    result = asyncio.run(
        send_completion(
            provider,
            model="gpt-5.4-mini",
            dev_prompt=dev_prompt,
            input_text=input_text,
            case_id=first_case.id,
        )
    )
    print(result)


def main() -> None:
    app()
