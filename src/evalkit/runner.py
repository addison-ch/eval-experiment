import asyncio
import uuid
from datetime import datetime
from pathlib import Path

from evalkit.client import CompletionProvider, send_completion
from evalkit.models import Case, CompletionResult, RunResult, Task
from evalkit.store import ResultWriter, write_run

# Assembled here because the runner owns turning a Task + Case into a full
# prompt; providers only see the finished dev/user text.
DEV_PROMPT = (
    "Follow the instructions in the user message exactly and respond with "
    "only the requested output."
)

COMPLETIONS_FILENAME = "completions.jsonl"
RUN_MANIFEST_FILENAME = "run.json"


async def _run_case(
    provider: CompletionProvider,
    model: str,
    prompt_template: str,
    case: Case,
) -> CompletionResult:
    """Render one case's prompt and get its completion. Plain async: knows
    nothing about concurrency limits — that is the runner's concern."""
    input_text = prompt_template.format(input=case.input)

    # Compute cache key...
    
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
    output_dir: Path,
    *,
    concurrency: int = 5,
) -> RunResult:
    """Run every case in a task concurrently, bounded by a semaphore.

    Writes two files under `output_dir`:
      - completions.jsonl: one CompletionResult per successful case, written the
        moment that case finishes (crash mid-run leaves the successes so far).
      - run.json: the run manifest (run-level metadata), written at start with
        completed_at=None and rewritten at the end to stamp completed_at.

    A case that fails (retries exhausted or a non-retryable error) is logged and
    dropped so one bad case can't kill the whole batch — the returned RunResult's
    completions hold only the successes, in case order.
    """
    completions_path = output_dir / COMPLETIONS_FILENAME
    manifest_path = output_dir / RUN_MANIFEST_FILENAME

    run = RunResult(
        run_id=uuid.uuid4().hex,
        task=task.name,
        model=model,
        started_at=datetime.now(),
    )
    # Write the manifest up front so the run is self-identifying and durable
    # immediately — a crash still leaves a record of what produced the
    # completions, with completed_at=None marking it unfinished.
    write_run(manifest_path, run)

    semaphore = asyncio.Semaphore(concurrency)

    with ResultWriter(completions_path) as writer:

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

    run.completions = [r for r in results if r is not None]
    run.completed_at = datetime.now()
    # Finalize: rewrite the manifest with completed_at now set.
    write_run(manifest_path, run)

    return run
