"""Standalone LLM wrapper for graph_agent / kernel_agent / optimize_agent.

Routes by model name:
  - "deepseek/*" or "deepseek-*" (e.g. "deepseek-chat", "deepseek-reasoner")
                                  → DeepSeek (DEEPSEEK_API_KEY, https://api.deepseek.com)
  - any other prefix (e.g. "anthropic/...", "z-ai/...", "openai/...")
                                  → OpenRouter (OPENROUTER_API_KEY)

Env:
  DEEPSEEK_API_KEY       — DeepSeek key (required for the deepseek route)
  OPENROUTER_API_KEY     — OpenRouter key (required for the openrouter route)
  OPENROUTER_MAX_TOKENS  — token cap (default 40000); also used as DEEPSEEK_MAX_TOKENS fallback
  DEEPSEEK_MAX_TOKENS    — override for deepseek output cap. Defaults depend on the model:
                           V4 (deepseek-v4-pro / -flash) → 384000 (the V4 hard cap);
                           V3 (deepseek-chat / -reasoner) → 8000 (V3 hard cap is 8192).
  GRAPH_REASONING=on/off — toggle the OpenRouter `reasoning` extra (DeepSeek ignores it)
"""

from __future__ import annotations

import os
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "8"))
_RETRY_SLEEP = float(os.environ.get("LLM_RETRY_SLEEP", "2"))
# Cap for the exponential backoff between retries (seconds).
_RETRY_SLEEP_MAX = float(os.environ.get("LLM_RETRY_SLEEP_MAX", "30"))


def _route(model: str) -> tuple[str, str, str, str]:
    """Return (provider, base_url, api_key, model_for_api).

    DeepSeek expects bare model ids ('deepseek-chat' / 'deepseek-reasoner'); strip a
    leading 'deepseek/' if present. OpenRouter accepts vendor-prefixed ids as-is.
    """
    m = (model or "").strip()
    is_ds = m.startswith("deepseek/") or m.startswith("deepseek-") or m == "deepseek"
    if is_ds:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (required for deepseek/* models).")
        api_model = m.split("/", 1)[1] if m.startswith("deepseek/") else m
        if api_model in ("deepseek", ""):
            api_model = "deepseek-chat"
        return "deepseek", "https://api.deepseek.com", key, api_model
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (required for non-deepseek models).")
    return "openrouter", "https://openrouter.ai/api/v1", key, m


def query_llm(prompt: str, model: str = "deepseek-v4-pro") -> str:
    """Send a single-user-message completion and return the text content.

    Uses streaming. OpenRouter sends ``: OPENROUTER PROCESSING`` SSE keep-alive
    comments while a slow (reasoning) model generates; a non-streaming request
    then fails to JSON-parse the body. Streaming consumes those comments
    correctly and lets us accumulate content from reasoning models whose budget
    is split between reasoning and answer tokens. DeepSeek streams plain SSE
    deltas (same openai-compatible shape) so the same loop works for both.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed; `pip install openai`.")

    provider, base_url, api_key, api_model = _route(model)
    client = OpenAI(base_url=base_url, api_key=api_key)
    if provider == "deepseek":
        # DeepSeek-V4 series caps output at 384K; older deepseek-chat caps at 8192.
        # Default to V4's max; users can override via DEEPSEEK_MAX_TOKENS for V3 ids.
        is_v3 = api_model in ("deepseek-chat", "deepseek-reasoner")
        default_max = "8000" if is_v3 else "384000"
        max_tokens = int(os.environ.get("DEEPSEEK_MAX_TOKENS", default_max))
    else:
        max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "40000"))

    # Thinking/reasoning models (kimi-k2.x, glm-5.x, ...) can burn the whole
    # token budget on reasoning and return EMPTY answer content. Disable
    # reasoning by default for reliability; override with GRAPH_REASONING=on.
    # DeepSeek doesn't accept OpenRouter's `reasoning` extra — keep it empty there.
    extra_body: dict = {}
    if provider == "openrouter" and \
       os.environ.get("GRAPH_REASONING", "off").lower() not in ("on", "1", "true"):
        extra_body["reasoning"] = {"enabled": False}

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=api_model,
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
            # Exponential backoff (capped). Transient failures like
            # "Connection error" recover far better with a growing delay than a
            # flat 2s — a brief API/network blip used to kill the whole op (and,
            # in batch mode, every op after it) after only ~10s of retries.
            sleep_s = min(_RETRY_SLEEP * (2 ** (attempt - 1)), _RETRY_SLEEP_MAX)
            print(f"[llm retry {attempt}/{_MAX_RETRIES}] {last_exc} (sleep {sleep_s:.0f}s)")
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM query failed after {_MAX_RETRIES} attempts: {last_exc}")


def get_llm_query():
    """Return the active LLM query callable: ``(prompt, model) -> str``.

    Compatibility shim: several callers (graph_agent, operator_agent,
    run_optimize, run_arm_batch) import ``get_llm_query`` from the pre-sync API.
    The post-sync API routes by model name inside ``query_llm`` itself, so the
    "active backend" is simply ``query_llm``. Kept so those imports resolve
    without each caller needing to change.
    """
    return query_llm
