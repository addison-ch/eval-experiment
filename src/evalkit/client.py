import datetime
import itertools
import time
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI

from evalkit.models import CompletionResult

load_dotenv()

_completion_ids = itertools.count(1)


@dataclass(frozen=True)
class ProviderResponse:
    output: str
    input_tokens: int
    output_tokens: int


class CompletionProvider(Protocol):
    async def complete(self, model: str, dev_prompt: str, input_text: str) -> ProviderResponse: ...


class OpenAICompletionProvider:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(max_retries=0)

    async def complete(self, model: str, dev_prompt: str, input_text: str) -> ProviderResponse:
        completion = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": dev_prompt},
                {"role": "user", "content": input_text},
            ],
        )
        usage = completion.usage
        return ProviderResponse(
            output=str(completion.choices[0].message.content),
            input_tokens=usage.prompt_tokens if usage else -1,
            output_tokens=usage.completion_tokens if usage else -1,
        )


async def send_completion(
    provider: CompletionProvider,
    model: str,
    dev_prompt: str,
    input_text: str,
    case_id: str,
) -> CompletionResult:
    start = time.perf_counter()
    response = await provider.complete(model=model, dev_prompt=dev_prompt, input_text=input_text)
    latency_ms = (time.perf_counter() - start) * 1000

    return CompletionResult(
        id=f"completion-{next(_completion_ids)}",
        case_id=case_id,
        model=model,
        output=response.output,
        latency_ms=latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_estimate=None,
        timestamp=datetime.datetime.now(),
    )
