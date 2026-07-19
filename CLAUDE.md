# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an LLM evaluation harness: it runs LLM completions against test cases concurrently, judges outputs with a second model via LLM-as-judge, and produces comparison reports. Built as a learning project mirroring production eval systems.

## Tech stack and tooling

- Python 3.12, dependencies managed with `uv` only — never `pip install` directly, always `uv add`
- Pydantic v2 for all data models — v2 API only, not v1 (`model_validate`, `field_validator`, `model_dump`)
- `ruff` for linting/formatting
- `pyright` in strict mode for type checking
- `pytest` with `pytest-asyncio` for tests

## Architecture rules

- All Pydantic models live in `src/evalkit/models.py` — never define models inline in other modules
- All LLM API calls go through the wrapper in `src/evalkit/client.py` — no raw HTTP or SDK calls anywhere else
- Concurrency is controlled in `runner.py` via a semaphore; individual functions should be plain async functions unaware of concurrency limits
- Raw API responses are always persisted to disk before any parsing or judging — never a pipeline where re-analysis requires re-calling APIs
- Judge parsing must handle malformed JSON with a repair path, never a bare `json.loads` that can crash a run

## Code style

- Type hints on every function signature; no `Any` unless unavoidable and commented
- No bare `except:` — catch specific exceptions; API errors must distinguish retryable (rate limit, timeout) from non-retryable (auth, bad request)
- Prefer small pure functions; side effects (disk, network) isolated at module edges
- Docstrings only where the "why" isn't obvious — no boilerplate docstrings restating the signature

## Testing rules

- Every module gets tests; all LLM API calls in tests are mocked — tests must never make real network calls or cost money
- Test the failure paths explicitly: malformed judge JSON, rate-limit retries, cache hits/misses
- Run `pytest` and `ruff check` before declaring any task complete

## Workflow rules for Claude

- Before implementing anything non-trivial, state your plan in 2-3 bullets and wait for approval
- When making a design choice (library, pattern, structure), state why in one sentence — this project is for learning
- Never mark a milestone done without running the test suite
- Do not add dependencies without asking first
- If uncertain whether something matches these conventions, ask rather than guess

## Commands

No `pyproject.toml` exists yet — these are the standard invocations for this stack once it's set up with `uv init` / `uv add`:

- Install/sync deps: `uv sync`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run pyright`
- Tests: `uv run pytest`
- Single test: `uv run pytest path/to/test_file.py::test_name`
- CLI entry point: TBD — depends on the `[project.scripts]` name chosen in `pyproject.toml`
