import asyncio
from pathlib import Path

import pytest

from evalkit.models import Case, ProviderResponse, Task
from evalkit.runner import COMPLETIONS_FILENAME, RUN_MANIFEST_FILENAME, run_task
from evalkit.store import read_results, read_run


def _make_task(num_cases: int) -> Task:
    return Task(
        name="t",
        prompt_template="Extract from: {input}",
        rubric="r",
        test_cases=[
            Case(id=f"case-{i}", input=f"input {i}", answer=None) for i in range(num_cases)
        ],
    )


class _RecordingProvider:
    """Fake provider that records the input_text it was called with and tracks
    peak concurrency so tests can assert the semaphore bound is respected."""

    def __init__(self, *, delay: float = 0.0) -> None:
        self._delay = delay
        self.inputs: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, model: str, dev_prompt: str, input_text: str) -> ProviderResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.inputs.append(input_text)
            if self._delay:
                await asyncio.sleep(self._delay)
            return ProviderResponse(output="ok", input_tokens=1, output_tokens=1)
        finally:
            self.in_flight -= 1


class _FailingProvider:
    """Fails for one specific case id; succeeds for all others."""

    def __init__(self, fail_for_input: str) -> None:
        self._fail_for_input = fail_for_input

    async def complete(self, model: str, dev_prompt: str, input_text: str) -> ProviderResponse:
        if self._fail_for_input in input_text:
            raise RuntimeError("boom")
        return ProviderResponse(output="ok", input_tokens=1, output_tokens=1)


@pytest.mark.asyncio
async def test_run_task_returns_result_per_case(tmp_path: Path) -> None:
    task = _make_task(3)
    provider = _RecordingProvider()

    run = await run_task(task, provider, model="m", output_dir=tmp_path)

    assert len(run.completions) == 3
    assert {r.case_id for r in run.completions} == {"case-0", "case-1", "case-2"}


@pytest.mark.asyncio
async def test_run_task_renders_prompt_template(tmp_path: Path) -> None:
    task = _make_task(1)
    provider = _RecordingProvider()

    await run_task(task, provider, model="m", output_dir=tmp_path)

    assert provider.inputs == ["Extract from: input 0"]


@pytest.mark.asyncio
async def test_run_task_respects_concurrency_limit(tmp_path: Path) -> None:
    task = _make_task(10)
    provider = _RecordingProvider(delay=0.01)

    await run_task(task, provider, model="m", output_dir=tmp_path, concurrency=3)

    assert provider.max_in_flight <= 3


@pytest.mark.asyncio
async def test_run_task_skips_failed_case_and_keeps_survivors(tmp_path: Path) -> None:
    task = _make_task(3)
    provider = _FailingProvider(fail_for_input="input 1")

    run = await run_task(task, provider, model="m", output_dir=tmp_path)

    assert len(run.completions) == 2
    assert {r.case_id for r in run.completions} == {"case-0", "case-2"}


@pytest.mark.asyncio
async def test_run_task_persists_only_successful_cases(tmp_path: Path) -> None:
    task = _make_task(3)
    provider = _FailingProvider(fail_for_input="input 1")

    await run_task(task, provider, model="m", output_dir=tmp_path)

    persisted = read_results(tmp_path / COMPLETIONS_FILENAME)
    assert {r.case_id for r in persisted} == {"case-0", "case-2"}


@pytest.mark.asyncio
async def test_run_task_writes_finalized_manifest(tmp_path: Path) -> None:
    task = _make_task(2)
    provider = _RecordingProvider()

    run = await run_task(task, provider, model="m", output_dir=tmp_path)

    manifest = read_run(tmp_path / RUN_MANIFEST_FILENAME)
    assert manifest.run_id == run.run_id
    assert manifest.task == "t"
    assert manifest.model == "m"
    assert manifest.completed_at is not None
    # Manifest holds run-level metadata only; per-case results live in the
    # completions stream, not duplicated here.
    assert manifest.completions == []


@pytest.mark.asyncio
async def test_run_task_writes_manifest_before_completion(tmp_path: Path) -> None:
    # The manifest is written at start with completed_at=None, so it exists (and
    # reads as unfinished) even while cases are still running.
    task = _make_task(1)
    manifest_path = tmp_path / RUN_MANIFEST_FILENAME

    class _CheckingProvider:
        async def complete(
            self, model: str, dev_prompt: str, input_text: str
        ) -> ProviderResponse:
            assert read_run(manifest_path).completed_at is None
            return ProviderResponse(output="ok", input_tokens=1, output_tokens=1)

    run = await run_task(task, _CheckingProvider(), model="m", output_dir=tmp_path)

    assert run.completed_at is not None
