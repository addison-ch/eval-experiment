from pathlib import Path

import pytest

from evalkit.loaders import TaskLoadError, load_cases, load_task


def _write_task_yaml(task_dir: Path, cases_file: str = "cases.jsonl") -> Path:
    yaml_path = task_dir / "task.yaml"
    yaml_path.write_text(
        f"name: summarization\n"
        f"prompt_template: 'Summarize: {{input}}'\n"
        f"rubric: Penalize hallucinated facts.\n"
        f"cases_file: {cases_file}\n"
    )
    return yaml_path


def test_load_task_parses_valid_yaml_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "cases.jsonl").write_text(
        '{"id": "case-1", "input": "hi", "answer": null}\n'
        '{"id": "case-2", "input": "bye", "answer": "goodbye"}\n'
    )
    yaml_path = _write_task_yaml(tmp_path)

    task = load_task(yaml_path)

    assert task.name == "summarization"
    assert task.prompt_template == "Summarize: {input}"
    assert len(task.test_cases) == 2
    assert task.test_cases[0].id == "case-1"
    assert task.test_cases[1].answer == "goodbye"


def test_load_task_missing_yaml_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        load_task(missing_path)


def test_load_task_missing_cases_file_raises(tmp_path: Path) -> None:
    yaml_path = _write_task_yaml(tmp_path, cases_file="missing.jsonl")

    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        load_task(yaml_path)


def test_load_cases_malformed_json_line_raises(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"id": "case-1", "input": "hi", "answer": null}\n'
        "not valid json\n"
    )

    with pytest.raises(TaskLoadError, match=r"line 2: invalid JSON"):
        load_cases(cases_path)


def test_load_cases_validation_error_raises(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"id": "case-1", "answer": null}\n'  # missing required `input`
    )

    with pytest.raises(TaskLoadError, match="line 1"):
        load_cases(cases_path)


def test_load_cases_empty_file_raises(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("")

    with pytest.raises(TaskLoadError, match="empty"):
        load_cases(cases_path)


def test_load_cases_duplicate_ids_raises(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"id": "case-1", "input": "hi", "answer": null}\n'
        '{"id": "case-1", "input": "bye", "answer": null}\n'
    )

    with pytest.raises(TaskLoadError, match="duplicate case id 'case-1'"):
        load_cases(cases_path)


def test_load_task_cases_file_resolved_relative_to_yaml_dir(tmp_path: Path) -> None:
    task_dir = tmp_path / "nested" / "task_dir"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "data" / "cases.jsonl").write_text(
        '{"id": "case-1", "input": "hi", "answer": null}\n'
    )
    yaml_path = _write_task_yaml(task_dir, cases_file="data/cases.jsonl")

    task = load_task(yaml_path)

    assert len(task.test_cases) == 1
