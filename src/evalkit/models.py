from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime


# Data models

# Tasks (essentially the prompts)
# Tasks should have test cases which you use to fill in the prompt templates
MetadataValue = str | int | float | bool | None

class Case(BaseModel):
    id: str
    input: str
    answer: str | None # optional reference answer
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

class Task(BaseModel):
    name: str
    prompt_template: str # Must have {input} as placeholder
    rubric: str
    test_cases: list[Case]

    @field_validator("prompt_template")
    @classmethod
    def prompt_template_has_input_placeholder(cls, v: str) -> str:
        if "{input}" not in v:
            raise ValueError("prompt_template must contain '{input}'")
        return v


class TaskYamlSpec(BaseModel):
    name: str
    prompt_template: str
    rubric: str
    cases_file: str


class ProviderResponse(BaseModel):
    # Persisted when caching, so it crosses the disk trust boundary on read-back
    # — hence a validated (frozen) pydantic model rather than a plain dataclass.
    model_config = ConfigDict(frozen=True)

    output: str
    input_tokens: int
    output_tokens: int


class CompletionResult(BaseModel):
    id: str
    case_id: str
    model: str
    input: str  # rendered input text sent to the model (not the dev prompt)
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_estimate: float | None
    timestamp: datetime

class JudgementResult(BaseModel):
    id: str
    completion_id: str
    score: int = Field(ge=1, le=5)
    is_pass: bool
    reasoning: str
    judge_model: str

class RunResult(BaseModel):
    run_id: str
    task: str
    model: str
    started_at: datetime
    completed_at: datetime | None = None
    completions: list[CompletionResult] = Field(default_factory=list[CompletionResult])
    judgments: list[JudgementResult] = Field(default_factory=list[JudgementResult])