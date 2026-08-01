from dataclasses import dataclass

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from evalkit.client import (
    OpenAICompletionProvider,
    _retry_with_backoff,  # pyright: ignore[reportPrivateUsage]
    send_completion,
)
from evalkit.models import ProviderResponse


class RetryableError(Exception):
    pass


class NonRetryableError(Exception):
    pass


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage | None


def _make_fake_completion(
    content: str, prompt_tokens: int, completion_tokens: int
) -> _FakeCompletion:
    return _FakeCompletion(
        choices=[_FakeChoice(message=_FakeMessage(content=content))],
        usage=_FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _make_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status_code, request=request)


def _make_provider(monkeypatch: pytest.MonkeyPatch) -> OpenAICompletionProvider:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAICompletionProvider()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("evalkit.client.asyncio.sleep", instant_sleep)


@pytest.mark.asyncio
async def test_retry_with_backoff_succeeds_on_first_try() -> None:
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await _retry_with_backoff(call, retryable_exceptions=(RetryableError,))

    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_with_backoff_recovers_after_retryable_failures() -> None:
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableError("transient")
        return "ok"

    result = await _retry_with_backoff(
        call, retryable_exceptions=(RetryableError,), max_attempts=5
    )

    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_raises_after_exhausting_attempts() -> None:
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise RetryableError("always fails")

    with pytest.raises(RetryableError):
        await _retry_with_backoff(call, retryable_exceptions=(RetryableError,), max_attempts=3)

    assert calls == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_does_not_retry_non_retryable_errors() -> None:
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise NonRetryableError("permanent")

    with pytest.raises(NonRetryableError):
        await _retry_with_backoff(call, retryable_exceptions=(RetryableError,), max_attempts=5)

    assert calls == 1


@pytest.mark.asyncio
async def test_openai_provider_complete_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(monkeypatch)

    async def fake_create(**_kwargs: object) -> _FakeCompletion:
        return _make_fake_completion("hello", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(
        provider._client.chat.completions,  # pyright: ignore[reportPrivateUsage]
        "create",
        fake_create,
    )

    response = await provider.complete(model="gpt-5.4-mini", dev_prompt="dp", input_text="it")

    assert response == ProviderResponse(output="hello", input_tokens=10, output_tokens=5)


@pytest.mark.asyncio
async def test_openai_provider_missing_usage_falls_back_to_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider(monkeypatch)

    async def fake_create(**_kwargs: object) -> _FakeCompletion:
        return _FakeCompletion(choices=[_FakeChoice(message=_FakeMessage(content="hi"))], usage=None)

    monkeypatch.setattr(
        provider._client.chat.completions,  # pyright: ignore[reportPrivateUsage]
        "create",
        fake_create,
    )

    response = await provider.complete(model="gpt-5.4-mini", dev_prompt="dp", input_text="it")

    assert response.input_tokens == -1
    assert response.output_tokens == -1


@pytest.mark.asyncio
async def test_openai_provider_retries_on_rate_limit_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider(monkeypatch)
    calls = 0

    async def flaky_create(**_kwargs: object) -> _FakeCompletion:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitError("rate limited", response=_make_response(429), body=None)
        return _make_fake_completion("ok", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(
        provider._client.chat.completions,  # pyright: ignore[reportPrivateUsage]
        "create",
        flaky_create,
    )

    response = await provider.complete(model="gpt-5.4-mini", dev_prompt="dp", input_text="it")

    assert response.output == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_openai_provider_does_not_retry_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(monkeypatch)
    calls = 0

    async def failing_create(**_kwargs: object) -> _FakeCompletion:
        nonlocal calls
        calls += 1
        raise AuthenticationError("bad key", response=_make_response(401), body=None)

    monkeypatch.setattr(
        provider._client.chat.completions,  # pyright: ignore[reportPrivateUsage]
        "create",
        failing_create,
    )

    with pytest.raises(AuthenticationError):
        await provider.complete(model="gpt-5.4-mini", dev_prompt="dp", input_text="it")

    assert calls == 1


class _FakeProvider:
    def __init__(
        self, response: ProviderResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, model: str, dev_prompt: str, input_text: str) -> ProviderResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


@pytest.mark.asyncio
async def test_send_completion_builds_completion_result() -> None:
    provider = _FakeProvider(
        response=ProviderResponse(output="answer", input_tokens=10, output_tokens=5)
    )

    result = await send_completion(
        provider, model="gpt-5.4-mini", dev_prompt="dp", input_text="it", case_id="case-1"
    )

    assert result.case_id == "case-1"
    assert result.model == "gpt-5.4-mini"
    assert result.input == "it"
    assert result.output == "answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.latency_ms >= 0
    assert result.id.startswith("completion-")


@pytest.mark.asyncio
async def test_send_completion_generates_sequential_ids() -> None:
    provider = _FakeProvider(response=ProviderResponse(output="a", input_tokens=1, output_tokens=1))

    first = await send_completion(
        provider, model="m", dev_prompt="dp", input_text="it", case_id="c1"
    )
    second = await send_completion(
        provider, model="m", dev_prompt="dp", input_text="it", case_id="c2"
    )

    first_n = int(first.id.split("-")[1])
    second_n = int(second.id.split("-")[1])
    assert second_n == first_n + 1


@pytest.mark.asyncio
async def test_send_completion_propagates_provider_failure() -> None:
    provider = _FakeProvider(error=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await send_completion(
            provider, model="m", dev_prompt="dp", input_text="it", case_id="c1"
        )

    assert provider.calls == 1
