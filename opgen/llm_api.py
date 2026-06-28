"""Standalone LLM wrappers for the agents.

Two backends, both via the OpenAI SDK:
- query_llm:      OpenRouter (OPENROUTER_API_KEY), model ids like
                  ``deepseek/deepseek-v4-pro``.
- query_deepseek: DeepSeek API direct (DEEPSEEK_API_KEY), bare model ids like
                  ``deepseek-v4-pro``, native ``thinking``/``reasoning_effort``.

Kept independent of MoKA/prompt so the agents stay self-contained.

Backend selection via env:
- LLM_BACKEND=deepseek  → uses query_deepseek (needs DEEPSEEK_API_KEY)
- LLM_BACKEND=openrouter (or unset) → uses query_llm (needs OPENROUTER_API_KEY)
- DEEPSEEK_MODEL        → override default model for direct DeepSeek calls
"""

from __future__ import annotations

import os
import time
from typing import Callable

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

_MAX_RETRIES = 5
_RETRY_SLEEP = 2


def query_llm(prompt: str, model: str = "anthropic/claude-sonnet-4.5") -> str:
    """Send a single-user-message completion and return the text content.

    Uses streaming. OpenRouter sends ``: OPENROUTER PROCESSING`` SSE keep-alive
    comments while a slow (reasoning) model generates; a non-streaming request
    then fails to JSON-parse the body. Streaming consumes those comments
    correctly and lets us accumulate content from reasoning models whose budget
    is split between reasoning and answer tokens.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed; `pip install openai`.")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "100000"))

    # Thinking/reasoning models (kimi-k2.x, glm-5.x, ...) can burn the whole
    # token budget on reasoning and return EMPTY answer content. Disable
    # reasoning by default for reliability; override with GRAPH_REASONING=on.
    extra_body: dict = {}
    if os.environ.get("GRAPH_REASONING", "off").lower() not in ("on", "1", "true"):
        extra_body["reasoning"] = {"enabled": False}

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
                stream=True,
                extra_body=extra_body,
            )
            chunks: list[str] = []
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    chunks.append(piece)
            content = "".join(chunks).strip()
            if content:
                return content
            last_exc = RuntimeError("empty content (model returned no answer tokens)")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if attempt < _MAX_RETRIES:
            print(f"[llm retry {attempt}/{_MAX_RETRIES}] {last_exc}")
            time.sleep(_RETRY_SLEEP)
    raise RuntimeError(f"LLM query failed after {_MAX_RETRIES} attempts: {last_exc}")


def query_deepseek(prompt: str, model: str = "deepseek-v4-pro") -> str:
    """Single-message completion against the DeepSeek API directly.

    Unlike query_llm (OpenRouter), this targets https://api.deepseek.com with
    DeepSeek's native param names: ``thinking`` + ``reasoning_effort`` instead of
    OpenRouter's ``reasoning``. The model id carries NO ``deepseek/`` prefix.

    Reads the key from DEEPSEEK_API_KEY and max tokens from DEEPSEEK_MAX_TOKENS.
    Thinking is ON by default (matching the documented usage); disable it with
    DEEPSEEK_THINKING=off. Streaming accumulates only answer ``content`` and
    ignores ``reasoning_content``.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed; `pip install openai`.")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
    max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "40000"))

    thinking_on = os.environ.get("DEEPSEEK_THINKING", "on").lower() in ("on", "1", "true")
    extra_body: dict = {}
    if thinking_on:
        extra_body["thinking"] = {"type": "enabled"}
        extra_body["reasoning_effort"] = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
    else:
        extra_body["thinking"] = {"type": "disabled"}

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=max_tokens,
                stream=True,
                extra_body=extra_body,
            )
            chunks: list[str] = []
            for event in stream:
                if not event.choices:
                    continue
                piece = getattr(event.choices[0].delta, "content", None)
                if piece:
                    chunks.append(piece)
            content = "".join(chunks).strip()
            if content:
                return content
            last_exc = RuntimeError("empty content (model returned no answer tokens)")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if attempt < _MAX_RETRIES:
            print(f"[deepseek retry {attempt}/{_MAX_RETRIES}] {last_exc}")
            time.sleep(_RETRY_SLEEP)
    raise RuntimeError(f"DeepSeek query failed after {_MAX_RETRIES} attempts: {last_exc}")


def get_llm_query() -> Callable[[str, str], str]:
    """Return the active LLM query function based on environment.

    LLM_BACKEND=deepseek  → query_deepseek (direct DeepSeek API)
    anything else / unset → query_llm     (OpenRouter)

    Model overrides:
    - DEEPSEEK_MODEL env var sets the default model for deepseek backend
    """
    backend = os.environ.get("LLM_BACKEND", "openrouter").lower()

    if backend == "deepseek":
        deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

        def _query(prompt: str, model: str = deepseek_model) -> str:
            return query_deepseek(prompt, model=model)

        return _query

    # Default: OpenRouter
    return query_llm
