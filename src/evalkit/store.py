from pathlib import Path
from types import TracebackType

from evalkit.models import CompletionResult, RunResult


class ResultWriter:
    """Append-only JSONL sink for CompletionResults.

    One result per line via pydantic's model_dump_json. Each write is flushed
    immediately so a crash mid-run leaves a valid file of everything completed
    so far. Used as a context manager to own the file handle's lifetime.

    No lock is taken around writes: asyncio is cooperatively single-threaded, so
    a synchronous write with no `await` inside it cannot interleave with another
    task. If writes ever move to threads / run_in_executor, this stops holding
    and a lock is required.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __enter__(self) -> "ResultWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._file.close()

    def write(self, result: CompletionResult) -> None:
        self._file.write(result.model_dump_json() + "\n")
        self._file.flush()


def read_results(path: str | Path) -> list[CompletionResult]:
    """Read a completions JSONL file back into CompletionResult models."""
    results_path = Path(path)
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results: list[CompletionResult] = []
    with results_path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            results.append(CompletionResult.model_validate_json(stripped))
    return results


def write_run(path: str | Path, run: RunResult) -> None:
    """Write the run manifest (run-level metadata) as a single JSON object.

    The per-case results live in the completions JSONL stream, so they are
    excluded here rather than duplicated into the manifest. Called once at run
    start (completed_at=None) and again at the end to stamp completed_at.
    """
    run_path = Path(path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(run.model_dump_json(exclude={"completions", "judgments"}))


def read_run(path: str | Path) -> RunResult:
    """Read a run manifest back into a RunResult (completions/judgments empty —
    those are read separately from the completions stream)."""
    run_path = Path(path)
    if not run_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {run_path}")
    return RunResult.model_validate_json(run_path.read_text())
