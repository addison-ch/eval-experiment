from pydantic import BaseModel, Field


# Data models

# Tasks (essentially the prompts)
# Tasks should have test cases which you use to fill in the prompt templates
MetadataValue = str | int | float | bool | None

class Case(BaseModel):
    id: int
    input: str
    answer: str | None # optional reference answer
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

class Task(BaseModel):
    name: str
    prompt_template: str # Must have {input} as placeholder
    rubric: str
    test_cases: list[Case]

