import asyncio
import datetime
import itertools
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError

from evalkit.models import CompletionResult

load_dotenv()

_completion_ids = itertools.count(1)

T = TypeVar("T")

_RETRYABLE_OPENAI_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,  # covers APITimeoutError too (it's a subclass)
    RateLimitError,
    InternalServerError,
)


async def _retry_with_backoff(
    call: Callable[[], Awaitable[T]],
    *,
    retryable_exceptions: tuple[type[Exception], ...],
    max_attempts: int = 5,
    base_delay_s: float = 1.0,
) -> T:
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except retryable_exceptions as e:
            if attempt == max_attempts:
                raise
            delay = base_delay_s * (2 ** (attempt - 1)) * (1 + random.uniform(-0.25, 0.25))
            print(f"[retry] attempt {attempt}/{max_attempts} failed ({e!r}); retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # loop always returns or raises


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
        completion = await _retry_with_backoff(
            lambda: self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": dev_prompt},
                    {"role": "user", "content": input_text},
                ],
            ),
            retryable_exceptions=_RETRYABLE_OPENAI_ERRORS,
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
