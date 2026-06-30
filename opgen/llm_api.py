"""Standalone LLM wrapper for graph_agent / kernel_agent / optimize_agent.

Routes by model name:
  - "claude-opus-4-8" / "idealab/<model>"
                                  → IdeaLab Anthropic-compatible proxy
                                    (IDEALAB_API_KEY, https://idealab.alibaba-inc.com/api/anthropic)
  - "deepseek/*" or "deepseek-*" (e.g. "deepseek-chat", "deepseek-reasoner")
                                  → DeepSeek (DEEPSEEK_API_KEY, https://api.deepseek.com)
  - any other prefix (e.g. "anthropic/...", "z-ai/...", "openai/...")
                                  → OpenRouter (OPENROUTER_API_KEY)

Env:
  IDEALAB_API_KEY        — IdeaLab proxy key (required for the idealab route, e.g.
                           claude-opus-4-8). Export it; do not hardcode it in source.
  IDEALAB_BASE_URL       — override the proxy base (default https://idealab.alibaba-inc.com/api/anthropic)
  IDEALAB_MAX_TOKENS     — output cap for the idealab route (default 40000)
  DEEPSEEK_API_KEY       — DeepSeek key (required for the deepseek route)
  OPENROUTER_API_KEY     — OpenRouter key (required for the openrouter route)
  OPENROUTER_MAX_TOKENS  — token cap (default 40000); also used as DEEPSEEK_MAX_TOKENS fallback
  DEEPSEEK_MAX_TOKENS    — override for deepseek output cap. Defaults depend on the model:
                           V4 (deepseek-v4-pro / -flash) → 384000 (the V4 hard cap);
                           V3 (deepseek-chat / -reasoner) → 8000 (V3 hard cap is 8192).
  GRAPH_REASONING=on/off — toggle the OpenRouter `reasoning` extra (DeepSeek ignores it)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

_MAX_RETRIES = 5
_RETRY_SLEEP = 2

# IdeaLab Anthropic-compatible proxy. NOTE: the proxy validates the model name against its
# own catalog *before* authenticating upstream, so only ids it knows work (e.g.
# "claude-opus-4-8"; "claude-opus-4.8" / dated suffixes are rejected as CE-001). The auth
# header may be either `x-api-key` or `Authorization: Bearer` — both are accepted.
_IDEALAB_DEFAULT_BASE = "https://idealab.alibaba-inc.com/api/anthropic"
_IDEALAB_MODELS = {"claude-opus-4-8"}


def _is_idealab(model: str) -> bool:
    m = (model or "").strip()
    return m.startswith("idealab/") or m in _IDEALAB_MODELS


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


def _query_idealab(prompt: str, model: str) -> str:
    """Single-message completion against the IdeaLab Anthropic Messages API proxy.

    Uses the native Anthropic ``/v1/messages`` shape (not OpenAI chat-completions), via
    stdlib urllib so the ``anthropic`` package is not required. Non-streaming: the proxy
    returns the whole message in one JSON body.
    """
    m = model.strip()
    api_model = m.split("/", 1)[1] if m.startswith("idealab/") else m
    base = os.environ.get("IDEALAB_BASE_URL", _IDEALAB_DEFAULT_BASE).rstrip("/")
    key = os.environ.get("IDEALAB_API_KEY")
    if not key:
        raise RuntimeError("IDEALAB_API_KEY is not set (required for idealab/claude-opus-4-8).")
    max_tokens = int(os.environ.get("IDEALAB_MAX_TOKENS", "40000"))

    url = f"{base}/v1/messages"
    # NOTE: `temperature` is deprecated/rejected for claude-opus-4-8 on this proxy, so it
    # is intentionally omitted.
    payload = json.dumps({
        "model": api_model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            # Anthropic success shape: {"content": [{"type":"text","text": "..."}], ...}
            parts = [b.get("text", "") for b in body.get("content", [])
                     if b.get("type") == "text"]
            content = "".join(parts).strip()
            if content:
                return content
            last_exc = RuntimeError(f"empty content (stop_reason={body.get('stop_reason')})")
        except urllib.error.HTTPError as exc:  # 4xx/5xx — read the proxy's error envelope
            detail = exc.read().decode("utf-8", "replace")
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if attempt < _MAX_RETRIES:
            print(f"[llm retry {attempt}/{_MAX_RETRIES}] {last_exc}")
            time.sleep(_RETRY_SLEEP)
    raise RuntimeError(f"IdeaLab query failed after {_MAX_RETRIES} attempts: {last_exc}")


def query_llm(prompt: str, model: str = "deepseek-v4-pro") -> str:
    """Send a single-user-message completion and return the text content.

    Uses streaming. OpenRouter sends ``: OPENROUTER PROCESSING`` SSE keep-alive
    comments while a slow (reasoning) model generates; a non-streaming request
    then fails to JSON-parse the body. Streaming consumes those comments
    correctly and lets us accumulate content from reasoning models whose budget
    is split between reasoning and answer tokens. DeepSeek streams plain SSE
    deltas (same openai-compatible shape) so the same loop works for both.
    """
    # IdeaLab proxy speaks the Anthropic Messages API (not OpenAI chat-completions);
    # handle it on its own path before touching the OpenAI client.
    if _is_idealab(model):
        return _query_idealab(prompt, model)

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
            print(f"[llm retry {attempt}/{_MAX_RETRIES}] {last_exc}")
            time.sleep(_RETRY_SLEEP)
    raise RuntimeError(f"LLM query failed after {_MAX_RETRIES} attempts: {last_exc}")
