import asyncio
from pathlib import Path

import pytest

from evalkit.client import ProviderResponse
from evalkit.models import Case, Task
from evalkit.runner import run_task
from evalkit.store import read_results


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

    results = await run_task(task, provider, model="m", output_path=tmp_path / "out.jsonl")

    assert len(results) == 3
    assert {r.case_id for r in results} == {"case-0", "case-1", "case-2"}


@pytest.mark.asyncio
async def test_run_task_renders_prompt_template(tmp_path: Path) -> None:
    task = _make_task(1)
    provider = _RecordingProvider()

    await run_task(task, provider, model="m", output_path=tmp_path / "out.jsonl")

    assert provider.inputs == ["Extract from: input 0"]


@pytest.mark.asyncio
async def test_run_task_respects_concurrency_limit(tmp_path: Path) -> None:
    task = _make_task(10)
    provider = _RecordingProvider(delay=0.01)

    await run_task(task, provider, model="m", output_path=tmp_path / "out.jsonl", concurrency=3)

    assert provider.max_in_flight <= 3


@pytest.mark.asyncio
async def test_run_task_skips_failed_case_and_keeps_survivors(tmp_path: Path) -> None:
    task = _make_task(3)
    provider = _FailingProvider(fail_for_input="input 1")

    results = await run_task(task, provider, model="m", output_path=tmp_path / "out.jsonl")

    assert len(results) == 2
    assert {r.case_id for r in results} == {"case-0", "case-2"}


@pytest.mark.asyncio
async def test_run_task_persists_only_successful_cases(tmp_path: Path) -> None:
    task = _make_task(3)
    provider = _FailingProvider(fail_for_input="input 1")
    output_path = tmp_path / "out.jsonl"

    await run_task(task, provider, model="m", output_path=output_path)

    persisted = read_results(output_path)
    assert {r.case_id for r in persisted} == {"case-0", "case-2"}
