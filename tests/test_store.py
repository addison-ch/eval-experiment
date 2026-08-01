from datetime import datetime
from pathlib import Path

import pytest

from evalkit.models import CompletionResult, RunResult
from evalkit.store import ResultWriter, read_results, read_run, write_run


def _make_result(completion_id: str, case_id: str) -> CompletionResult:
    return CompletionResult(
        id=completion_id,
        case_id=case_id,
        model="gpt-5.4-mini",
        input="some input",
        output="some output",
        latency_ms=123.4,
        input_tokens=10,
        output_tokens=5,
        cost_estimate=None,
        timestamp=datetime(2026, 8, 1, 12, 0, 0),
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "completions.jsonl"
    result = _make_result("completion-1", "case-1")

    with ResultWriter(path) as writer:
        writer.write(result)

    read_back = read_results(path)
    assert read_back == [result]


def test_write_multiple_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "completions.jsonl"
    results = [_make_result(f"completion-{i}", f"case-{i}") for i in range(3)]

    with ResultWriter(path) as writer:
        for result in results:
            writer.write(result)

    assert read_results(path) == results


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "task-123" / "completions.jsonl"

    with ResultWriter(path) as writer:
        writer.write(_make_result("completion-1", "case-1"))

    assert path.is_file()


def test_read_results_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "completions.jsonl"
    result = _make_result("completion-1", "case-1")
    path.write_text(f"\n{result.model_dump_json()}\n\n")

    assert read_results(path) == [result]


def test_read_results_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_results(tmp_path / "nope.jsonl")


def test_write_flushes_incrementally(tmp_path: Path) -> None:
    # Each write should be readable before the writer is closed — proves results
    # aren't buffered until the end (the crash-safety property).
    path = tmp_path / "completions.jsonl"

    with ResultWriter(path) as writer:
        writer.write(_make_result("completion-1", "case-1"))
        assert len(read_results(path)) == 1
        writer.write(_make_result("completion-2", "case-2"))
        assert len(read_results(path)) == 2


def test_write_run_then_read_run_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    run = RunResult(
        run_id="run-1",
        task="ticket_extraction",
        model="gpt-5.4-mini",
        started_at=datetime(2026, 8, 1, 12, 0, 0),
        completed_at=datetime(2026, 8, 1, 12, 5, 0),
    )

    write_run(path, run)

    assert read_run(path) == run


def test_write_run_excludes_completions(tmp_path: Path) -> None:
    # The manifest is metadata only — completions live in the JSONL stream and
    # must not be duplicated into run.json.
    path = tmp_path / "run.json"
    run = RunResult(
        run_id="run-1",
        task="t",
        model="m",
        started_at=datetime(2026, 8, 1, 12, 0, 0),
        completions=[_make_result("completion-1", "case-1")],
    )

    write_run(path, run)

    assert "completion-1" not in path.read_text()
    assert read_run(path).completions == []


def test_read_run_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_run(tmp_path / "nope.json")
