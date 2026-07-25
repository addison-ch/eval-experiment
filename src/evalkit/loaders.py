import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from evalkit.models import Case, Task, TaskYamlSpec


class TaskLoadError(ValueError):
    """Raised when a task definition or its cases file is malformed or invalid."""


def load_cases(path: str | Path) -> list[Case]:
    cases_path = Path(path)
    if not cases_path.is_file():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    cases: list[Case] = []
    seen_ids: set[str] = set()

    with cases_path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                raw_case = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise TaskLoadError(
                    f"{cases_path} line {line_number}: invalid JSON: {e}"
                ) from e

            try:
                case = Case.model_validate(raw_case)
            except ValidationError as e:
                raise TaskLoadError(f"{cases_path} line {line_number}: {e}") from e

            if case.id in seen_ids:
                raise TaskLoadError(
                    f"{cases_path} line {line_number}: duplicate case id '{case.id}'"
                )
            seen_ids.add(case.id)
            cases.append(case)

    if not cases:
        raise TaskLoadError(f"{cases_path}: cases file is empty")

    return cases


def load_task(path: str | Path) -> Task:
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Task YAML file not found: {yaml_path}")

    with yaml_path.open("r") as f:
        raw = yaml.safe_load(f)

    try:
        spec = TaskYamlSpec.model_validate(raw)
    except ValidationError as e:
        raise TaskLoadError(f"{yaml_path}: invalid task definition: {e}") from e

    cases_path = (yaml_path.parent / spec.cases_file).resolve()
    test_cases = load_cases(cases_path)

    try:
        return Task(
            name=spec.name,
            prompt_template=spec.prompt_template,
            rubric=spec.rubric,
            test_cases=test_cases,
        )
    except ValidationError as e:
        raise TaskLoadError(f"{yaml_path}: invalid task definition: {e}") from e
