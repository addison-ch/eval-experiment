import asyncio
from pathlib import Path

from evalkit.client import CompletionProvider, send_completion
from evalkit.models import Case, CompletionResult, Task
from evalkit.store import ResultWriter

# Assembled here because the runner owns turning a Task + Case into a full
# prompt; providers only see the finished dev/user text.
DEV_PROMPT = (
    "Follow the instructions in the user message exactly and respond with "
    "only the requested output."
)


async def _run_case(
    provider: CompletionProvider,
    model: str,
    prompt_template: str,
    case: Case,
) -> CompletionResult:
    """Render one case's prompt and get its completion. Plain async: knows
    nothing about concurrency limits — that is the runner's concern."""
    input_text = prompt_template.format(input=case.input)
    return await send_completion(
        provider,
        model=model,
        dev_prompt=DEV_PROMPT,
        input_text=input_text,
        case_id=case.id,
    )


async def run_task(
    task: Task,
    provider: CompletionProvider,
    model: str,
    output_path: Path,
    *,
    concurrency: int = 5,
) -> list[CompletionResult]:
    """Run every case in a task concurrently, bounded by a semaphore.

    Each successful CompletionResult is written to `output_path` (JSONL) the
    moment its case finishes, so a crash mid-run leaves everything completed so
    far on disk.

    A case that fails (retries exhausted or a non-retryable error) is logged
    and dropped so one bad case can't kill the whole batch — the returned list
    holds only the successes, in case order.
    """
    semaphore = asyncio.Semaphore(concurrency)

    with ResultWriter(output_path) as writer:

        async def _guarded(case: Case) -> CompletionResult | None:
            async with semaphore:
                try:
                    result = await _run_case(provider, model, task.prompt_template, case)
                except Exception as e:  # noqa: BLE001 -- batch isolation boundary
                    # Intentional broad catch: this is the failure-isolation seam.
                    # Anything a single case raises is contained here so the rest
                    # of the batch survives.
                    print(f"[runner] case {case.id!r} failed, skipping: {e!r}")
                    return None
                # Persist outside the try: a completion we can't write to disk is
                # a real error, not a per-case skip — let it propagate.
                writer.write(result)
                return result

        results = await asyncio.gather(*(_guarded(case) for case in task.test_cases))

    return [r for r in results if r is not None]
