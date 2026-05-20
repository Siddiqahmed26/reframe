"""Provider-pluggable LLM wrapper with production-grade auto-fallback.

LLM_PROVIDER=auto rotates through every configured provider key. Per-process
shared health tracks each provider's cooldown so a quota-exhausted Gemini
stays cold across requests, not just one. Failure categorization picks the
right cooldown: a 429 burns 60s, a quota-exhausted account burns 6 hours,
a bad key burns 24 hours.

Healthy providers serve in round-robin. Cooling providers move to the tail
of the chain — they're tried only if everything healthy has already failed.

See `provider_health_snapshot()` for diagnostic state used by /health.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from backend.config import get_settings


logger = logging.getLogger(__name__)


# ── Module-level provider health ──────────────────────────────────────────
# Per-provider state SHARED across all LLM() instances. This is what lets
# a Gemini quota-exhaustion in one request affect the next request's
# routing — without this, every new LLM() would re-test all providers.

ErrorCategory = Literal["quota", "transient", "auth", "bad_request", "unknown"]


@dataclass
class ProviderHealth:
    cool_until: float = 0.0
    last_error: str = ""
    last_error_category: str = ""
    consecutive_failures: int = 0
    last_success: float = 0.0
    total_calls: int = 0
    total_failures: int = 0


_PROVIDER_HEALTH: Dict[str, ProviderHealth] = {}
_HEALTH_LOCK = threading.Lock()
# Round-robin index across healthy providers. Bumped each LLM.complete()
# call to spread load when the operator has multiple healthy keys.
_RR_INDEX = 0


def _health_for(name: str) -> ProviderHealth:
    """Get or lazily create the ProviderHealth for `name`."""
    with _HEALTH_LOCK:
        h = _PROVIDER_HEALTH.get(name)
        if h is None:
            h = ProviderHealth()
            _PROVIDER_HEALTH[name] = h
        return h


def _record_success(name: str) -> None:
    h = _health_for(name)
    with _HEALTH_LOCK:
        h.consecutive_failures = 0
        h.last_success = time.time()
        h.total_calls += 1


def _record_failure(name: str, err: Exception, cooldown: float, category: str) -> None:
    h = _health_for(name)
    with _HEALTH_LOCK:
        h.total_calls += 1
        h.total_failures += 1
        h.consecutive_failures += 1
        h.last_error = str(err)[:200]
        h.last_error_category = category
        if cooldown > 0:
            # Only EXTEND the cooldown — a fresh transient failure shouldn't
            # shorten an existing quota cooldown.
            h.cool_until = max(h.cool_until, time.time() + cooldown)


def provider_health_snapshot() -> Dict[str, Dict[str, Any]]:
    """Public read-only view of the health registry. Used by /health."""
    out: Dict[str, Dict[str, Any]] = {}
    now = time.time()
    with _HEALTH_LOCK:
        for name, h in _PROVIDER_HEALTH.items():
            failure_rate = (h.total_failures / h.total_calls) if h.total_calls else 0.0
            out[name] = {
                "healthy": h.cool_until <= now,
                "cool_until_epoch": h.cool_until if h.cool_until > now else None,
                "last_error": h.last_error or None,
                "last_error_category": h.last_error_category or None,
                "consecutive_failures": h.consecutive_failures,
                "last_success_epoch": h.last_success or None,
                "total_calls": h.total_calls,
                "total_failures": h.total_failures,
                "failure_rate": round(failure_rate, 3),
            }
    return out


# ── Provider definition ──────────────────────────────────────────────────


class _Provider:
    def __init__(self, name: str, client: Any, model: str, complete_fn: Callable):
        self.name = name
        self.client = client
        self.model = model
        self._complete_fn = complete_fn

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


def _build_gemini(settings, fast):
    if not settings.gemini_api_key:
        return None
    from openai import OpenAI
    # Gemini exposes an OpenAI-compatible endpoint at /v1beta/openai/.
    client = OpenAI(api_key=settings.gemini_api_key, base_url=settings.gemini_base_url, max_retries=2)
    model = settings.gemini_fast_model if fast else settings.gemini_model
    return _Provider("gemini", client, model, _openai_compat_complete)


def _build_openai(settings, fast):
    if not settings.openai_api_key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, max_retries=2)
    model = settings.openai_fast_model if fast else settings.openai_model
    return _Provider("openai", client, model, _openai_compat_complete)


_BUILDERS = {
    "anthropic": _build_anthropic,
    "xai": _build_xai,
    "grok": _build_xai,
    "groq": _build_groq,
    "gemini": _build_gemini,
    "openai": _build_openai,
}

# Free, no-card providers first.
_AUTO_ORDER = ["groq", "gemini", "xai", "openai", "anthropic"]


def _categorize_error(err: Exception) -> ErrorCategory:
    """Classify an error so we can pick the right cooldown.

    Returned categories and intended treatment:
      "quota"       — daily/monthly limit hit. Long cooldown (~6h).
      "transient"   — rate limit / 5xx / network blip. Short cooldown (60s).
      "auth"        — bad key, no credits, account disabled. Cold for 24h.
      "bad_request" — our prompt is wrong; do NOT cool the provider.
      "unknown"     — anything else; mild cooldown (30s).
    """
    msg = str(err).lower()
    status = getattr(err, "status_code", None) or getattr(err, "code", None)

    # Quota — long cooldown so we don't burn round-trips for hours.
    quota_signals = (
        "quota exceeded", "quota_exhausted", "resource_exhausted",
        "insufficient_quota", "exceeded your current quota", "billing",
        "tokens per day", "requests per day",
    )
    if any(t in msg for t in quota_signals):
        return "quota"

    # Auth / permission / no credits — effectively permanent for this process.
    auth_signals = (
        "invalid_api_key", "invalid api key", "incorrect api key",
        "no permission", "does not have permission",
        "no credits", "insufficient credits",
        "doesn't have any credits", "no licenses",
        "account is disabled", "team is disabled",
        "authentication", "unauthorized",
    )
    if status == 401 or status == 403 or any(t in msg for t in auth_signals):
        return "auth"

    # Bad request — the prompt was malformed; not the provider's fault.
    if status == 400 or "bad request" in msg or "model_not_found" in msg:
        return "bad_request"

    # Transient — rate limit / 5xx / network.
    transient_signals = (
        "429", "rate limit", "too many",
        "500", "502", "503", "504",
        "connection", "timeout", "timed out",
        "overloaded", "unavailable",
    )
    if status in (429, 502, 503, 504) or any(t in msg for t in transient_signals):
        return "transient"

    try:
        from openai import RateLimitError
        if isinstance(err, RateLimitError):
            return "transient"
    except Exception:
        pass

    return "unknown"


def _cooldown_for_category(cat: ErrorCategory) -> float:
    """Translate a category to a cooldown duration in seconds. Env-tunable
    via PROVIDER_COOLDOWN_*_S settings."""
    settings = get_settings()
    if cat == "quota":
        return float(settings.provider_cooldown_quota_s)
    if cat == "transient":
        return float(settings.provider_cooldown_transient_s)
    if cat == "auth":
        return float(settings.provider_cooldown_auth_s)
    if cat == "bad_request":
        return 0.0  # do not cool the provider — the next call has a different prompt
    return float(settings.provider_cooldown_unknown_s)


def _has_key(settings, name: str) -> bool:
    key_attr = {
        "groq": "groq_api_key",
        "gemini": "gemini_api_key",
        "xai": "xai_api_key",
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
    }.get(name)
    if not key_attr:
        return False
    return bool(getattr(settings, key_attr, None))


def _ordered_providers(settings) -> List[str]:
    """Return the provider chain for LLM_PROVIDER=auto: configured providers
    ordered healthy-first (within base preference), cooling-last. Cooling
    providers stay in the list as a last-resort fallback."""
    base = ["groq", "gemini", "xai", "openai", "anthropic"]
    configured = [p for p in base if _has_key(settings, p)]
    now = time.time()
    healthy = [p for p in configured if _health_for(p).cool_until <= now]
    cooling = [p for p in configured if p not in healthy]
    return healthy + cooling


def _pick_starting_index(healthy_count: int) -> int:
    """Round-robin starting index across healthy providers. Spreads load
    when the operator has multiple healthy keys configured."""
    global _RR_INDEX
    if healthy_count <= 0:
        return 0
    with _HEALTH_LOCK:
        idx = _RR_INDEX % healthy_count
        _RR_INDEX = (_RR_INDEX + 1) % max(1, healthy_count)
    return idx


class LLM:
    def __init__(self, model: Optional[str] = None, fast: bool = False) -> None:
        settings = get_settings()
        self.settings = settings
        provider = (settings.llm_provider or "anthropic").lower().strip()
        self.providers: List[_Provider] = []

        if provider == "auto":
            for name in _ordered_providers(settings):
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
                f"Unknown LLM_PROVIDER '{provider}'. Use 'auto' or one of: "
                "anthropic, xai, grok, groq, gemini, openai."
            )

        if not self.providers:
            raise RuntimeError(
                "No API keys configured. Set ANTHROPIC_API_KEY, XAI_API_KEY, "
                "GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in .env."
            )

    @classmethod
    def from_user_key(cls, provider: str, api_key: str, fast: bool = False) -> "LLM":
        """Build an LLM using a single user-supplied key, no fallback chain.

        The key is held only as the OpenAI/Anthropic client's internal state
        (in-memory, per request) and never written to settings, env, logs, or
        disk. Errors raised here deliberately do NOT interpolate the key.
        """
        allowed = ("anthropic", "xai", "grok", "groq", "gemini", "openai")
        prov = (provider or "").lower().strip()
        if prov not in allowed:
            # Note: no key in the error message.
            raise ValueError(f"Unsupported BYOK provider: {prov!r}")
        if not api_key or not api_key.strip():
            raise ValueError("BYOK API key is empty.")

        settings = get_settings()
        # Build a one-off _Provider that ignores the configured settings
        # for the key field. We re-use the existing complete functions.
        if prov == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            model = settings.anthropic_fast_model if fast else settings.anthropic_model
            p = _Provider("anthropic", client, model, _anthropic_complete)
        else:
            from openai import OpenAI
            if prov in ("xai", "grok"):
                base = settings.xai_base_url
                model = settings.xai_fast_model if fast else settings.xai_model
                name = "xai"
            elif prov == "groq":
                base = settings.groq_base_url
                model = settings.groq_fast_model if fast else settings.groq_model
                name = "groq"
            elif prov == "gemini":
                base = settings.gemini_base_url
                model = settings.gemini_fast_model if fast else settings.gemini_model
                name = "gemini"
            else:  # openai
                base = settings.openai_base_url
                model = settings.openai_fast_model if fast else settings.openai_model
                name = "openai"
            client = OpenAI(api_key=api_key, base_url=base, max_retries=2)
            p = _Provider(name, client, model, _openai_compat_complete)

        inst = cls.__new__(cls)
        inst.settings = settings
        inst.providers = [p]
        return inst

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        *,
        deadline: Optional[float] = None,
    ) -> str:
        """Try each provider in this LLM's chain. Healthy providers are tried
        in round-robin order so load spreads. Cooling providers are at the
        tail of the chain and only tried if everything healthy has failed.

        `deadline` is an absolute epoch time past which we abort the
        fallback walk and raise. The agents pass the orchestrator's
        request-wide deadline so a slow chain of failures never burns more
        than MAX_REQUEST_SECONDS on a single agent call.
        """
        last_err: Optional[Exception] = None
        last_category: ErrorCategory = "unknown"

        # Partition this chain into healthy/cooling using the module-level
        # health registry. Healthy providers go first (rotated by RR),
        # cooling stays as a tail fallback.
        now = time.time()
        healthy = [p for p in self.providers if _health_for(p.name).cool_until <= now]
        cooling = [p for p in self.providers if _health_for(p.name).cool_until > now]

        ordered: List[_Provider] = []
        if healthy:
            start = _pick_starting_index(len(healthy))
            ordered.extend(healthy[start:] + healthy[:start])
        ordered.extend(cooling)

        for p in ordered:
            if deadline is not None and time.time() >= deadline:
                # We're out of time-budget before even trying the next
                # provider. Don't burn more wall-clock — bubble the
                # last_err up so the orchestrator can decide.
                logger.warning(
                    "LLM.complete aborting (deadline reached); attempted=%s",
                    [pp.name for pp in ordered[: ordered.index(p)]],
                )
                break
            try:
                result = p.complete(system, user, max_tokens, temperature)
                _record_success(p.name)
                return result
            except Exception as e:
                cat = _categorize_error(e)
                cooldown = _cooldown_for_category(cat)
                _record_failure(p.name, e, cooldown, cat)
                last_err = e
                last_category = cat
                logger.warning(
                    "Provider %s failed (%s, cool=%.0fs): %s",
                    p.name, cat, cooldown, str(e)[:200],
                )
                continue

        # Every provider in the chain failed (or the chain was empty).
        # Don't sleep-and-retry here — the modern auto chain has multiple
        # providers and waiting on a single cooling one risks blowing the
        # request-wide deadline. The caller decides what to do.
        if last_err is None:
            raise RuntimeError(
                "No providers available in this LLM chain. Configure at "
                "least one of GROQ_API_KEY, GEMINI_API_KEY, XAI_API_KEY, "
                "OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )
        raise RuntimeError(
            f"All providers failed (last category={last_category}). "
            f"Last error: {last_err}"
        )

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 3000,
        temperature: float = 0.1,
        *,
        deadline: Optional[float] = None,
    ) -> Any:
        raw = self.complete(
            system, user,
            max_tokens=max_tokens, temperature=temperature,
            deadline=deadline,
        )
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
