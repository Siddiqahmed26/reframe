"""Provider-pluggable LLM wrapper with auto-fallback.

LLM_PROVIDER values:
  auto       Try Groq -> xAI -> Anthropic (whichever has a key set), in that
             order. On 429 or connection errors, the failing provider cools
             down for 30 seconds and the next is tried.
  anthropic  Anthropic only.
  xai / grok xAI Grok only.
  groq       Groq only.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, List, Optional

from backend.config import get_settings


logger = logging.getLogger(__name__)


COOLDOWN_SECONDS = 30.0
# Permanent-ish errors (no credits, bad key, account disabled): don't ping
# this provider again for the rest of the process lifetime — there's no point
# burning a round-trip per LLM call when the answer won't change.
LONG_COOLDOWN_SECONDS = 60 * 60 * 24


class _Provider:
    def __init__(self, name: str, client: Any, model: str, complete_fn: Callable):
        self.name = name
        self.client = client
        self.model = model
        self._complete_fn = complete_fn
        self.cool_until: float = 0.0

    def available(self) -> bool:
        return time.time() >= self.cool_until

    def cool(self, seconds: float = COOLDOWN_SECONDS) -> None:
        self.cool_until = time.time() + seconds

    def complete(self, system, user, max_tokens, temperature):
        return self._complete_fn(self.client, self.model, system, user, max_tokens, temperature)


def _anthropic_complete(client, model, system, user, max_tokens, temperature):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _openai_compat_complete(client, model, system, user, max_tokens, temperature):
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message or not choice.message.content:
        return ""
    return choice.message.content.strip()


def _build_anthropic(settings, fast):
    if not settings.anthropic_api_key:
        return None
    from anthropic import Anthropic
    client = Anthropic(api_key=settings.anthropic_api_key)
    model = settings.anthropic_fast_model if fast else settings.anthropic_model
    return _Provider("anthropic", client, model, _anthropic_complete)


def _build_xai(settings, fast):
    if not settings.xai_api_key:
        return None
    from openai import OpenAI
    # max_retries=2 (SDK default): respects Retry-After headers so 429s on a
    # provider with no working fallback don't fail the whole request. For
    # permanent errors (401/403), the SDK doesn't retry and we hit our
    # auto-fallback immediately.
    client = OpenAI(api_key=settings.xai_api_key, base_url=settings.xai_base_url, max_retries=2)
    model = settings.xai_fast_model if fast else settings.xai_model
    return _Provider("xai", client, model, _openai_compat_complete)


def _build_groq(settings, fast):
    if not settings.groq_api_key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url, max_retries=2)
    model = settings.groq_fast_model if fast else settings.groq_model
    return _Provider("groq", client, model, _openai_compat_complete)


_BUILDERS = {
    "anthropic": _build_anthropic,
    "xai": _build_xai,
    "grok": _build_xai,
    "groq": _build_groq,
}

_AUTO_ORDER = ["groq", "xai", "anthropic"]


def _classify_error(err) -> float:
    """Return a cooldown duration in seconds for this provider after the
    given error. 0 means don't cool (try again on next call).

    - Transient (429 / rate limit / 5xx / connection): COOLDOWN_SECONDS
    - Permanent-ish (401 invalid key, 403 no credits / disabled team,
      404 model not found): LONG_COOLDOWN_SECONDS — pinging the same
      provider on every call just wastes a round-trip each time, so we
      effectively disable it for the rest of the process lifetime.
    - Anything else: 0 (transient, but no explicit backoff)
    """
    msg = str(err).lower()

    # Permanent: auth, permissions, billing, model not available
    permanent_signals = (
        "401", "403", "404",
        "invalid_api_key", "invalid api key", "incorrect api key",
        "no permission", "does not have permission",
        "no credits", "insufficient credits", "insufficient_quota",
        "doesn't have any credits", "no licenses", "license",
        "account is disabled", "team is disabled",
        "model_not_found", "model not found",
    )
    if any(sig in msg for sig in permanent_signals):
        return LONG_COOLDOWN_SECONDS

    # Transient: rate limits + 5xx
    transient_signals = (
        "429", "rate limit", "too many",
        "500", "502", "503", "504",
        "connection", "timeout", "timed out",
    )
    if any(sig in msg for sig in transient_signals):
        return COOLDOWN_SECONDS

    try:
        from openai import RateLimitError
        if isinstance(err, RateLimitError):
            return COOLDOWN_SECONDS
    except Exception:
        pass

    return 0.0


class LLM:
    def __init__(self, model: Optional[str] = None, fast: bool = False) -> None:
        settings = get_settings()
        self.settings = settings
        provider = (settings.llm_provider or "anthropic").lower().strip()
        self.providers: List[_Provider] = []

        if provider == "auto":
            for name in _AUTO_ORDER:
                p = _BUILDERS[name](settings, fast)
                if p is not None:
                    if model:
                        p.model = model
                    self.providers.append(p)
        elif provider in _BUILDERS:
            p = _BUILDERS[provider](settings, fast)
            if p is not None:
                if model:
                    p.model = model
                self.providers.append(p)
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. Use 'auto', 'anthropic', 'xai', or 'groq'."
            )

        if not self.providers:
            raise RuntimeError(
                "No API keys configured. Set ANTHROPIC_API_KEY, XAI_API_KEY, or GROQ_API_KEY in .env."
            )

    def complete(self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.2) -> str:
        last_err: Optional[Exception] = None
        for p in self.providers:
            if not p.available():
                continue
            try:
                return p.complete(system, user, max_tokens, temperature)
            except Exception as e:
                cooldown = _classify_error(e)
                if cooldown > 0:
                    p.cool(cooldown)
                    logger.warning(
                        "Provider %s failed (%.0fs cooldown): %s",
                        p.name, cooldown, str(e)[:200],
                    )
                last_err = e
                continue

        # All providers were unavailable or failed. Pick the one whose
        # cooldown ends soonest (excluding ones with very long cooldowns
        # like 24h permanent fails — no point waiting hours), wait for it,
        # and retry a few times with progressive backoff. This is the
        # critical path for users with only one working provider.
        candidates = [p for p in self.providers if p.cool_until - time.time() < 300]
        if not candidates:
            raise RuntimeError(
                f"All providers are unavailable for an extended period. Last error: {last_err}"
            )
        soonest = min(candidates, key=lambda x: x.cool_until)
        for attempt in range(3):
            wait = max(0.0, soonest.cool_until - time.time())
            if wait > 0:
                logger.info("Waiting %.0fs for %s cooldown (attempt %d)", wait, soonest.name, attempt + 1)
                time.sleep(min(wait, 60.0))
            try:
                return soonest.complete(system, user, max_tokens, temperature)
            except Exception as e:
                last_err = e
                cooldown = _classify_error(e)
                if cooldown > 0:
                    soonest.cool(cooldown)
                logger.warning(
                    "Retry %d on %s failed: %s", attempt + 1, soonest.name, str(e)[:200]
                )
        raise RuntimeError(f"All providers failed after retries. Last error: {last_err}")

    def complete_json(self, system: str, user: str, max_tokens: int = 3000, temperature: float = 0.1) -> Any:
        raw = self.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        return parse_json_block(raw)


def parse_json_block(raw):
    """Parse a JSON object or array from an LLM response. Returns None on
    failure so callers don't have to wrap every call in try/except.

    LLMs sometimes truncate at max_tokens leaving an unterminated string,
    or emit trailing commas / prose around the block. We strip code fences,
    extract the first {...} or [...] span, drop trailing commas, and finally
    try to amputate at the last well-formed brace.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Repair attempt 1: drop trailing commas
    repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Repair attempt 2: model probably hit max_tokens and produced an
    # unterminated string. Try to close at the last complete top-level
    # element by truncating to the last balanced brace.
    salvaged = _truncate_to_balanced(repaired)
    if salvaged:
        try:
            return json.loads(salvaged)
        except json.JSONDecodeError:
            pass
    return None


def _truncate_to_balanced(s: str):
    """Find the longest prefix of s that ends with a fully-closed element,
    then append the closers needed to balance the remaining open containers.

    Example:
        '{"instructions": [{"a": 1}, {"a": 2}, {"a": "unfinished'
      -> '{"instructions": [{"a": 1}, {"a": 2}]}'
    """
    if not s or s[0] not in "{[":
        return None
    open_stack: List[str] = []
    in_string = False
    escape = False
    last_close_pos = -1
    last_open_snapshot: List[str] = []
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            open_stack.append(ch)
        elif ch in "}]":
            if open_stack:
                open_stack.pop()
            # Any close brace/bracket is a safe truncation point — at this
            # position the element just finished and we can append closers
            # for whatever's still open above us.
            last_close_pos = i + 1
            last_open_snapshot = list(open_stack)
    if last_close_pos < 0:
        return None
    truncated = s[:last_close_pos].rstrip(", \n\t")
    for opener in reversed(last_open_snapshot):
        truncated += "}" if opener == "{" else "]"
    return truncated
