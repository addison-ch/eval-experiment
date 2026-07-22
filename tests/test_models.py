import pytest
from pydantic import ValidationError

from evalkit.models import Task, Case


def test_test_case_parses_valid_data() -> None:
    case = Case(id=1, input="summarize this", answer="a summary", metadata={"lang": "en"})

    assert case.id == 1
    assert case.input == "summarize this"
    assert case.answer == "a summary"
    assert case.metadata == {"lang": "en"}


def test_test_case_metadata_defaults_to_empty_dict() -> None:
    case = Case(id=1, input="summarize this", answer=None)

    assert case.metadata == {}


def test_test_case_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate({"id": 1, "input": "summarize this"})  # missing `answer`


def test_test_case_wrong_type_raises() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate({"id": "not-an-int", "input": "summarize this", "answer": None})


def test_test_case_invalid_metadata_value_raises() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(
            {"id": 1, "input": "summarize this", "answer": None, "metadata": {"tags": ["a", "b"]}}
        )


def test_task_parses_valid_data() -> None:
    task = Task(
        name="summarization",
        prompt_template="Summarize: {input}",
        rubric="Penalize hallucinated facts.",
        test_cases=[Case(id=1, input="summarize this", answer=None)],
    )

    assert task.name == "summarization"
    assert len(task.test_cases) == 1
    assert task.test_cases[0].id == 1


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
                "test_cases": [{"id": "bad", "input": "x", "answer": None}],
            }
        )
