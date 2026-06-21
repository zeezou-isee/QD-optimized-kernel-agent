"""Standalone LLM wrapper (OpenRouter) for graph_agent.

Kept independent of MoKA/prompt so the agent is self-contained. Reads the API
key from OPENROUTER_API_KEY and max tokens from OPENROUTER_MAX_TOKENS.
"""

from __future__ import annotations

import os
import time

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
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "40000"))

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
