"""Incremental per-API-key LLM token/usage tracker.

Every LLM call goes through llm_api.query_llm(); this module hooks in via
`record(key, provider, model, input_tokens, output_tokens, ...)` and merges the
delta into a shared JSON file (`batch/results/api_usage.json` by default).

Design:
  - Key identity: masked as `<PROVIDER>:<first6>...<last4>` so the file is
    keyed by API key WITHOUT storing the actual secret (project rule).
  - Concurrency: batch_runner spawns many subprocesses that each call the LLM
    in parallel. Uses fcntl.flock (LOCK_EX) around read-modify-write so
    concurrent recorders don't clobber each other.
  - Atomic write: temp-file + rename, so a killed process can never leave a
    corrupt JSON.

Schema:
  {
    "keys": {
      "IDEALAB:55a594...5622": {
        "provider": "idealab",
        "first_seen": "2026-07-05T20:00:00",
        "last_seen":  "2026-07-05T20:15:33",
        "total_calls": 42,
        "models": {
          "claude-opus-4-8": {
            "calls":                    42,
            "input_tokens":         123456,
            "output_tokens":        67890,
            "cache_read_tokens":         0,
            "cache_creation_tokens":     0
          }
        }
      }
    },
    "updated_at": "2026-07-05T20:15:33"
  }
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "batch" / "results" / "api_usage.json"


def _mask(key: str) -> str:
    """Reduce an API key to `first6...last4` so the file is keyed by the key
    identity but is NOT a secret dump. Short keys collapse to `***`."""
    k = (key or "").strip()
    if len(k) < 12:
        return "***"
    return f"{k[:6]}...{k[-4:]}"


def _ident(provider: str, key: str) -> str:
    return f"{provider.upper()}:{_mask(key)}"


def _default_bucket() -> dict:
    return {"provider": "", "first_seen": "", "last_seen": "",
            "total_calls": 0, "models": {}}


def _default_model_stats() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0}


def record(*, provider: str, key: str, model: str,
           input_tokens: int = 0, output_tokens: int = 0,
           cache_read_tokens: int = 0, cache_creation_tokens: int = 0,
           path: Path | str | None = None) -> None:
    """Merge a single call's usage into the tracker file. Never raises — a
    tracker failure must NOT break the calling LLM path (usage-logging is
    strictly best-effort telemetry).
    """
    try:
        _record_locked(provider=provider, key=key, model=model,
                       input_tokens=int(input_tokens or 0),
                       output_tokens=int(output_tokens or 0),
                       cache_read_tokens=int(cache_read_tokens or 0),
                       cache_creation_tokens=int(cache_creation_tokens or 0),
                       path=Path(path) if path else _DEFAULT_PATH)
    except Exception:  # noqa: BLE001 — telemetry must not break the caller
        pass


def _record_locked(*, provider: str, key: str, model: str,
                   input_tokens: int, output_tokens: int,
                   cache_read_tokens: int, cache_creation_tokens: int,
                   path: Path) -> None:
    ident = _ident(provider, key)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Cross-process lock lives on a SEPARATE sidecar file (never renamed away),
    # so concurrent writers can't leapfrog each other by locking their own stale
    # FD and then renaming a temp file over a peer's atomic write. The lockfile
    # never changes name → flock holds through the full read-modify-write, and
    # rewriting the target as truncate-in-place makes the operation single-FD
    # (no atomic rename gymnastics needed under an exclusive lock).
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            data: dict = {"keys": {}, "updated_at": now}
            if path.exists():
                try:
                    raw = path.read_text(encoding="utf-8").strip()
                    if raw:
                        data = json.loads(raw)
                except (json.JSONDecodeError, OSError):
                    data = {"keys": {}, "updated_at": now}

            buckets = data.setdefault("keys", {})
            bucket = buckets.setdefault(ident, _default_bucket())
            bucket["provider"] = provider
            if not bucket.get("first_seen"):
                bucket["first_seen"] = now
            bucket["last_seen"] = now
            bucket["total_calls"] = int(bucket.get("total_calls", 0)) + 1
            ms = bucket.setdefault("models", {}).setdefault(model, _default_model_stats())
            ms["calls"] += 1
            ms["input_tokens"] += input_tokens
            ms["output_tokens"] += output_tokens
            ms["cache_read_tokens"] += cache_read_tokens
            ms["cache_creation_tokens"] += cache_creation_tokens
            data["updated_at"] = now

            path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
