"""Agent: rewrite resume paragraphs to align with the JD while staying truthful.

This is the workhorse. We send the rewriter the JD analysis, the match
report, and *exactly* the paragraphs from the original resume that are
candidates for rewriting (bullets and summary lines — not headings, not
contact info). It returns a 1:1 mapping from paragraph index to a new string.

The hard constraint: the rewriter MUST NOT invent experience. It can:
- reword to surface relevant keywords
- reorder claims within a bullet
- swap a generic verb for a stronger one
- drop a detail that's irrelevant to this JD in favor of one that already
  exists in the candidate's history

It MUST NOT:
- add technologies the candidate hasn't used
- inflate seniority or scope
- change quantitative claims (numbers, percentages, team sizes)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from backend.agents.llm import LLM
from backend.models import (
    JDAnalysis,
    MatchReport,
    ResumeAnalysis,
    ResumeParagraph,
    RewriteInstruction,
    RewriteResult,
)


logger = logging.getLogger(__name__)


SYSTEM = """You are an elite resume writer. Your job is to rewrite bullet points so they MAXIMIZE keyword overlap with a target job description, while staying strictly truthful about what the candidate has actually done.

DEFAULT BEHAVIOR: Rewrite. Returning the original is the EXCEPTION, not the rule. For every bullet, your first instinct is "how can I reframe this using the JD's vocabulary while preserving the underlying truth?" Only return the original when you genuinely cannot improve keyword alignment without lying.

You will receive:
  - The structured job description analysis (with must_have_skills and keywords)
  - The match report (which JD requirements are covered/weak/missing)
  - A "JD keywords to surface" list — words/phrases from the JD you should weave into rewrites when the candidate truthfully has that experience
  - A numbered list of resume paragraphs, each tagged with section + a hard MAX character budget

REWRITE PLAYBOOK (apply liberally):
  a) Replace generic verbs with stronger, JD-aligned ones. "Built" → "Engineered", "Architected", "Designed end-to-end", etc.
  b) Substitute synonyms with the JD's exact vocabulary. If the JD says "LLMs" and the bullet says "GPT-4o", you may rewrite to "LLMs such as GPT-4o". If the JD says "production deployment" and the bullet says "shipped", swap. ATS systems match literally — synonyms cost matches.
  c) Surface technologies/concepts the bullet implies but doesn't say. A RAG pipeline implies "vector databases", "embeddings", "retrieval". A multi-agent system implies "agent orchestration", "tool use". Add the implied term when the JD asks for it.
  d) Reorder claims to lead with what the JD cares about. If the JD's top priority is "scalability" and your bullet ends with "scaled to 5M requests", move that to the front.
  e) Add JD-relevant framing. If the JD emphasizes "production-grade", add that qualifier where true. If it asks for "cross-functional collaboration", note collaborations that happened.

HARD CONSTRAINTS (violating these is failure):
1. NEVER invent technologies, frameworks, certifications, or experience the candidate doesn't already mention somewhere in their resume.
2. NEVER change quantitative claims upward. "5+ workflows" cannot become "10+ workflows". You may drop a number, never inflate one.
3. NEVER inflate seniority. "Engineer" does not become "Senior Engineer" or "Lead".
4. NEVER add a degree, year of experience, or certification the candidate doesn't hold.
5. Preserve tense and grammatical person. Past stays past.
6. The rewritten text MUST be ≤ the "max chars" value. If you can't fit the new keywords, drop a less important detail or shorten an adjective — don't overflow.
7. If a bullet is a section heading, company name, date range, or contact line, return it UNCHANGED.

OUTPUT FORMAT — return ONLY valid JSON with one entry per paragraph_index supplied:
{
  "instructions": [
    {"paragraph_index": 0, "original": "...", "rewritten": "...", "reason": "one short phrase like 'surfaced LLM and production deployment keywords'"}
  ]
}

Targets: aim to materially rewrite AT LEAST 70% of the supplied bullets. The "reason" field for unchanged bullets must be a real explanation like "already optimal for these JD keywords" or "no truthful angle to surface remaining missing terms" — not a default.
"""


def _select_rewriteable(paragraphs: List[ResumeParagraph]) -> List[ResumeParagraph]:
    """Pick the paragraphs we'd ever want to rewrite."""
    out: List[ResumeParagraph] = []
    for p in paragraphs:
        if not p.text:
            continue
        if p.is_heading:
            continue
        # Skip very short lines that look like dates or single tokens
        if len(p.text.split()) < 3 and not p.is_bullet:
            continue
        # Skip lines that look like contact info
        lowered = p.text.lower()
        if "@" in lowered or "linkedin.com" in lowered or "github.com" in lowered:
            continue
        # Likely candidates: bullets and summary lines
        if p.is_bullet or p.section in ("summary", "experience", "projects"):
            out.append(p)
    return out


def _chunk(items: List[ResumeParagraph], size: int) -> List[List[ResumeParagraph]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _budget_for(p: ResumeParagraph, override: Optional[int]) -> int:
    """Per-paragraph character ceiling. If the caller provided a hint, use
    that; otherwise allow +10% over the original (rounded up, never below 40)."""
    if override is not None and override > 0:
        return max(override, 40)
    return max(40, int(len(p.text) * 1.1) + 5)


def rewrite(
    jd: JDAnalysis,
    resume: ResumeAnalysis,
    match_report: MatchReport,
    paragraphs: List[ResumeParagraph],
    length_hints: Optional[Dict[int, int]] = None,
) -> RewriteResult:
    """Rewrite resume bullets to align with the JD.

    Args:
        length_hints: optional map of paragraph_index -> max chars. For PDFs
            this should be the original span's character count plus a small
            margin, so the rewrite fits in the original bbox.
    """
    targets = _select_rewriteable(paragraphs)
    if not targets:
        return RewriteResult()

    llm = LLM()
    all_instructions: List[RewriteInstruction] = []
    budgets: Dict[int, int] = {p.index: _budget_for(p, (length_hints or {}).get(p.index)) for p in targets}

    # Build the "JD keywords to surface" list once: must-haves + general
    # keywords, deduped. This is the explicit vocabulary the rewriter should
    # try to weave into bullets when the candidate truthfully has the work.
    jd_vocab = _dedupe(list(jd.must_have_skills) + list(jd.keywords))

    # Process in batches of 24 paragraphs. A typical resume has 15-25
    # rewriteable bullets, so this usually completes in a single LLM call,
    # which matters on free-tier providers with tight rate limits.
    batch_num = 0
    for batch in _chunk(targets, 24):
        batch_num += 1
        numbered = "\n".join(
            f"[{p.index}] ({p.section or 'unknown'}, max {budgets[p.index]} chars): {p.text}"
            for p in batch
        )
        user = f"""JD must-have skills: {', '.join(jd.must_have_skills) or '(none extracted)'}
JD nice-to-have skills: {', '.join(jd.nice_to_have_skills) or '(none extracted)'}
JD keywords to surface (use these literal terms where truthful): {', '.join(jd_vocab) or '(none)'}

Match report rewrite priorities (top of your list):
{chr(10).join('- ' + p for p in match_report.rewrite_priorities) or '(none)'}

Weakly covered requirements (these need stronger surfacing): {', '.join(match_report.weakly_covered) or '(none)'}
Missing requirements (only surface if the candidate truthfully has them — do NOT invent): {', '.join(match_report.missing) or '(none)'}

Paragraphs to rewrite (index in brackets, section in parens, hard char ceiling in the "max" hint). Aim to materially rewrite AT LEAST 70% of these:
{numbered}

Return the JSON schema in the system prompt, one entry per paragraph_index above. Preserve the exact paragraph_index numbers. The "rewritten" field MUST be <= the max chars shown.
"""
        try:
            data = llm.complete_json(system=SYSTEM, user=user, max_tokens=8000)
        except Exception as e:
            # One bad batch shouldn't poison the whole pipeline. Log and
            # move on — earlier batches' instructions are still applied.
            logger.warning("Rewriter batch %d: LLM call raised %s: %s",
                           batch_num, type(e).__name__, str(e)[:200])
            continue
        if not isinstance(data, dict):
            logger.warning(
                "Rewriter batch %d: LLM returned non-dict (likely truncated JSON); "
                "skipping this batch", batch_num
            )
            continue
        batch_instructions: List[RewriteInstruction] = []
        for ins in data.get("instructions", []):
            try:
                instruction = RewriteInstruction(**ins)
            except Exception:
                continue
            budget = budgets.get(instruction.paragraph_index)
            if budget and len(instruction.rewritten) > budget:
                instruction.rewritten = _truncate(instruction.rewritten, budget)
            batch_instructions.append(instruction)

        # Diagnostic: how many actually changed vs returned identical?
        changed = sum(
            1 for i in batch_instructions
            if i.rewritten and i.rewritten.strip() != i.original.strip()
        )
        logger.info(
            "Rewriter batch %d: %d/%d bullets rewritten (%d unchanged)",
            batch_num, changed, len(batch_instructions), len(batch_instructions) - changed,
        )
        all_instructions.extend(batch_instructions)

    return RewriteResult(instructions=all_instructions)


def _dedupe(items: List[str]) -> List[str]:
    """Lowercase-dedupe while preserving original casing of first occurrence."""
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to fit within max_chars while ending at a natural
    clause boundary. Strongly prefers sentence-ending punctuation over
    word-boundary chops — sacrificing a few extra characters to avoid
    ugly endings like "improving customer." or "and strong."

    Search strategy, in order of preference (within the last 40% of budget):
      1. Period or semicolon → cut after it
      2. Comma → cut before it, append period
      3. Word boundary → cut there, append period
      4. Hard cut at budget if nothing else works
    """
    if len(text) <= max_chars:
        return text

    # Don't shave more than 40% off — if we can't find a good clause
    # ending in that range, accept a word-boundary cut.
    min_keep = int(max_chars * 0.6)

    # 1. Prefer ending at a period or semicolon (full clause)
    for end_punct in (". ", "; ", ".\n", ";\n"):
        idx = text.rfind(end_punct, min_keep, max_chars)
        if idx >= 0:
            return text[: idx + 1].rstrip()

    # 2. Comma — cut before it and add a period
    idx = text.rfind(", ", min_keep, max_chars)
    if idx >= 0:
        return text[: idx].rstrip(" ,;:.") + "."

    # 3. Word boundary
    idx = text.rfind(" ", min_keep, max_chars)
    if idx >= 0:
        return text[: idx].rstrip(" ,;:.") + "."

    # 4. Hard cut
    return text[: max_chars - 1].rstrip(" ,;:.") + "."
