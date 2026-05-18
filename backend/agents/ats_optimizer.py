"""Agent: verify ATS keyword coverage of the rewritten resume.

Deterministic keyword-presence pass first (no LLM needed). If keywords are
still missing afterward, the LLM is asked for non-fabricating suggestions
the orchestrator can feed back into the rewriter.
"""
from __future__ import annotations

import re
from typing import List

from backend.agents.llm import LLM
from backend.models import ATSScore, JDAnalysis


_TOKEN_CHARS = r"a-z0-9+#"


def _normalize(text: str) -> str:
    """Lowercase and collapse non-token characters into single spaces."""
    s = text.lower()
    s = re.sub(rf"[^{_TOKEN_CHARS}]+", " ", s)
    return s.strip()


def _contains(haystack_norm: str, needle: str) -> bool:
    """Whole-word/phrase containment check on an already-normalized haystack."""
    n = _normalize(needle)
    if not n:
        return False
    pat = rf"(?:^|\s){re.escape(n)}(?:$|\s)"
    return bool(re.search(pat, haystack_norm))


def score(rewritten_text: str, jd: JDAnalysis, *, llm: "LLM | None" = None) -> ATSScore:
    """Compute ATS keyword coverage of the rewritten resume against the JD."""
    haystack = _normalize(rewritten_text)
    keywords = list(dict.fromkeys(jd.keywords + jd.must_have_skills))

    if not keywords:
        return ATSScore(keyword_coverage=1.0, matched_keywords=[], missing_keywords=[], suggestions=[])

    matched: List[str] = []
    missing: List[str] = []
    for kw in keywords:
        if _contains(haystack, kw):
            matched.append(kw)
        else:
            missing.append(kw)

    coverage = len(matched) / len(keywords)

    suggestions: List[str] = []
    if missing:
        suggestions = _suggest(missing, jd, llm=llm)

    return ATSScore(
        keyword_coverage=round(coverage, 3),
        matched_keywords=matched,
        missing_keywords=missing,
        suggestions=suggestions,
    )


SUGGEST_SYSTEM = (
    "You help candidates close ATS keyword gaps WITHOUT lying. You receive a "
    "list of keywords the resume is missing and the job description analysis. "
    "For each missing keyword, suggest ONE concrete way the candidate could "
    "surface it IF they already have that experience (for example, if they "
    "used Redis at any past role, mention it explicitly in the relevant "
    "bullet). Never instruct the candidate to claim experience they do not "
    "have.\n\n"
    "Return ONLY valid JSON: {\"suggestions\": [\"string\"]}. No prose, no fences."
)


def _suggest(missing: List[str], jd: JDAnalysis, *, llm: "LLM | None" = None) -> List[str]:
    if llm is None:
        try:
            llm = LLM(fast=True)
        except Exception:
            return [f"Consider surfacing '{kw}' if you have relevant experience." for kw in missing[:5]]

    user = f"Missing keywords: {missing}\n\nJD analysis:\n{jd.model_dump_json(indent=2)}\n"
    try:
        data = llm.complete_json(system=SUGGEST_SYSTEM, user=user, max_tokens=1200)
    except Exception:
        return [f"Consider surfacing '{kw}' if you have relevant experience." for kw in missing[:5]]
    if isinstance(data, dict):
        s = data.get("suggestions", [])
        return [str(x) for x in s][:10]
    return []
