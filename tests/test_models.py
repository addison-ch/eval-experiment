from datetime import datetime

import pytest
from pydantic import ValidationError

from evalkit.models import Case, CompletionResult, JudgementResult, RunResult, Task


def test_test_case_parses_valid_data() -> None:
    case = Case(id="case-1", input="summarize this", answer="a summary", metadata={"lang": "en"})

    assert case.id == "case-1"
    assert case.input == "summarize this"
    assert case.answer == "a summary"
    assert case.metadata == {"lang": "en"}


def test_test_case_metadata_defaults_to_empty_dict() -> None:
    case = Case(id="case-1", input="summarize this", answer=None)

    assert case.metadata == {}


def test_test_case_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate({"id": "case-1", "input": "summarize this"})  # missing `answer`


def test_test_case_wrong_type_raises() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate({"id": 123, "input": "summarize this", "answer": None})


def test_test_case_invalid_metadata_value_raises() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(
            {
                "id": "case-1",
                "input": "summarize this",
                "answer": None,
                "metadata": {"tags": ["a", "b"]},
            }
        )


def test_task_parses_valid_data() -> None:
    task = Task(
        name="summarization",
        prompt_template="Summarize: {input}",
        rubric="Penalize hallucinated facts.",
        test_cases=[Case(id="case-1", input="summarize this", answer=None)],
    )

    assert task.name == "summarization"
    assert len(task.test_cases) == 1
    assert task.test_cases[0].id == "case-1"


def test_task_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {"name": "summarization", "prompt_template": "Summarize: {input}", "test_cases": []}
        )


def test_task_invalid_test_cases_raises() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "name": "summarization",
                "prompt_template": "Summarize: {input}",
                "rubric": "Penalize hallucinated facts.",
                "test_cases": [{"id": "case-1", "answer": None}],  # missing `input`
            }
        )


def test_task_prompt_template_missing_placeholder_raises() -> None:
    with pytest.raises(ValidationError):
        Task(
            name="summarization",
            prompt_template="Summarize this document.",  # missing `{input}`
            rubric="Penalize hallucinated facts.",
            test_cases=[],
        )


def test_completion_result_parses_valid_data() -> None:
    completion = CompletionResult(
        id="completion-1",
        case_id="case-1",
        model="gpt-5",
        output="a summary",
        latency_ms=123.4,
        input_tokens=10,
        output_tokens=20,
        cost_estimate=0.001,
        timestamp=datetime(2026, 7, 24, 12, 0, 0),
    )

    assert completion.id == "completion-1"
    assert completion.output == "a summary"
    assert completion.timestamp == datetime(2026, 7, 24, 12, 0, 0)


def test_completion_result_parses_iso_timestamp_string() -> None:
    completion = CompletionResult.model_validate(
        {
            "id": "completion-1",
            "case_id": "case-1",
            "model": "gpt-5",
            "output": "a summary",
            "latency_ms": 123.4,
            "input_tokens": 10,
            "output_tokens": 20,
            "cost_estimate": 0.001,
            "timestamp": "2026-07-24T12:00:00",
        }
    )

    assert completion.timestamp == datetime(2026, 7, 24, 12, 0, 0)


def test_completion_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        CompletionResult.model_validate(
            {
                "id": "completion-1",
                "case_id": "case-1",
                "model": "gpt-5",
                "output": "a summary",
                "latency_ms": 123.4,
                "input_tokens": 10,
                "output_tokens": 20,
                # missing `cost_estimate` and `timestamp`
            }
        )


def test_completion_result_wrong_type_raises() -> None:
    with pytest.raises(ValidationError):
        CompletionResult.model_validate(
            {
                "id": "completion-1",
                "case_id": "case-1",
                "model": "gpt-5",
                "output": "a summary",
                "latency_ms": "fast",  # not a float
                "input_tokens": 10,
                "output_tokens": 20,
                "cost_estimate": 0.001,
                "timestamp": "2026-07-24T12:00:00",
            }
        )


def test_judgement_result_parses_valid_data() -> None:
    judgement = JudgementResult(
        id="judgement-1",
        completion_id="completion-1",
        score=4,
        is_pass=True,
        reasoning="Accurate and concise.",
        judge_model="gpt-5",
    )

    assert judgement.score == 4
    assert judgement.is_pass is True


def test_judgement_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        JudgementResult.model_validate(
            {
                "id": "judgement-1",
                "completion_id": "completion-1",
                "score": 4,
                "is_pass": True,
                # missing `reasoning` and `judge_model`
            }
        )


def test_judgement_result_wrong_type_raises() -> None:
    with pytest.raises(ValidationError):
        JudgementResult.model_validate(
            {
                "id": "judgement-1",
                "completion_id": "completion-1",
                "score": "five",  # not an int
                "is_pass": True,
                "reasoning": "Accurate and concise.",
                "judge_model": "gpt-5",
            }
        )


def test_judgement_result_score_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        JudgementResult.model_validate(
            {
                "id": "judgement-1",
                "completion_id": "completion-1",
                "score": 6,  # outside the 1-5 range
                "is_pass": True,
                "reasoning": "Accurate and concise.",
                "judge_model": "gpt-5",
            }
        )


def test_run_result_parses_valid_data_with_defaults() -> None:
    run = RunResult(
        task="summarization",
        model="gpt-5",
        started_at=datetime(2026, 7, 24, 12, 0, 0),
    )

    assert run.completed_at is None
    assert run.completions == []
    assert run.judgments == []


def test_run_result_parses_nested_results() -> None:
    completion = CompletionResult(
        id="completion-1",
        case_id="case-1",
        model="gpt-5",
        output="a summary",
        latency_ms=123.4,
        input_tokens=10,
        output_tokens=20,
        cost_estimate=0.001,
        timestamp=datetime(2026, 7, 24, 12, 0, 0),
    )
    judgement = JudgementResult(
        id="judgement-1",
        completion_id="completion-1",
        score=4,
        is_pass=True,
        reasoning="Accurate and concise.",
        judge_model="gpt-5",
    )

    run = RunResult(
        task="summarization",
        model="gpt-5",
        started_at=datetime(2026, 7, 24, 12, 0, 0),
        completions=[completion],
        judgments=[judgement],
    )

    assert run.completions == [completion]
    assert run.judgments == [judgement]


def test_run_result_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        RunResult.model_validate({"task": "summarization", "model": "gpt-5"})  # missing `started_at`


def test_run_result_invalid_nested_completion_raises() -> None:
    with pytest.raises(ValidationError):
        RunResult.model_validate(
            {
                "task": "summarization",
                "model": "gpt-5",
                "started_at": "2026-07-24T12:00:00",
                "completions": [{"id": "completion-1"}],  # missing required fields
            }
        )
