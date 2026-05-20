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
import os
import re
from typing import Dict, List, Optional, Tuple

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
2. PRESERVE every quantitative claim verbatim. Numbers like "5+", "60%", "10K+", "95%", "$1M", "2-week sprints" MUST appear in the rewrite exactly as in the original. You may drop a soft adjective; you may NEVER drop or modify a number. NEVER inflate a number upward.
3. NEVER inflate seniority. "Engineer" does not become "Senior Engineer" or "Lead".
4. NEVER add a degree, year of experience, or certification the candidate doesn't hold.
5. Preserve tense and grammatical person. Past stays past.
6. The rewritten text MUST be ≤ the "max chars" value. This is a HARD LIMIT — going even 1 character over breaks the layout. If you can't fit the new keywords within the budget, drop a less-important phrase or replace an adjective with a shorter synonym. Prefer shorter words ("use" over "utilize", "build" over "construct"). Do not overflow.
7. If a bullet is a section heading, company name, date range, or contact line, return it UNCHANGED.
8. Every rewritten bullet MUST end with a period ("."). Never end with ";" or "," or trail off with an em dash. If you used ";" inside the bullet to join clauses, the FINAL character is still a period.
9. Use plain ASCII. No em dashes ("—"), no en dashes ("–"), no curly quotes, no Unicode ligatures. Use "-" for any dash and straight quotes only.

STRICT MODE (every output bullet is checked against the original)
You are NOT a copywriter. You are a constrained editor. Treat every original bullet as ground truth that must be preserved in meaning and detail. Specifically:
  a. Every NUMBER in the original (percentage, count, year, version, "5+", "10+", "100%", "~70%", "~80%", "8 minutes", "2-week", etc.) MUST appear unchanged in your rewrite. Never drop a number. Never round one. Never change "5+" to "five plus". Never expand "100%" to "all" or "every" or contract "40%" to "significant".
  b. Every PROPER NOUN in the original (product names, company names, tool names, frameworks, model names) MUST appear unchanged. Examples: "LangGraph", "ChromaDB", "FAISS", "LoRA/QLoRA", "GPT-4o", "Claude", "Gemini 2.0", "LangSmith", "Hugging Face Spaces", "Next.js 16", "React 19", "Spring Boot", "Stripe", "Bitbucket", "ElevenLabs", "Tavily", "Fetch", "GitHub", "Filesystem", "Postman", "Jira", "SQL", "Excel", "PC hardware", "Windows/Linux", "Deccan AI Catalyst Hackathon 2026". Preserve the spaces inside multi-word tokens ("Spring Boot" not "SpringBoot", "Gemini 2.0" not "Gemini2.0").
  c. Every CONCRETE ACHIEVEMENT clause in the original (anything with a number or a named outcome) MUST be present in the rewrite. You may rephrase it. You may move it within the bullet. You may NOT delete it.
  d. You may NOT add any new tool, framework, methodology, role, or scope that is not already mentioned in this candidate's resume. If a JD keyword is missing from the resume entirely, it stays missing. Mark it in the optional "unfulfillable_keywords" field instead.
  e. You may NOT change the candidate's role verb in a way that inflates seniority or implies a different responsibility. "Participated" cannot become "Conducted" or "Led". "Contributed" cannot become "Owned". "Helped" cannot become "Drove". A few safe upgrades are allowed only when the original verb already implies leadership: "Built" can become "Engineered", "Developed" can become "Designed and built", but "Conducted" is reserved for activities the candidate ran end-to-end.
  f. Output text must be syntactically complete. No trailing fragments. No half-sentences. If you cannot complete the rewrite within the token budget, return the original verbatim with reason "token_budget".
  g. Preserve all spacing between tokens, especially around version numbers, model names, and quantities. Do not concatenate "Spring" and "Boot". Do not concatenate "Gemini" and "2.0". Do not concatenate digit and unit ("8 minutes" not "8minutes").

OUTPUT FORMAT — return ONLY valid JSON with one entry per paragraph_index supplied:
{
  "instructions": [
    {
      "paragraph_index": 0,
      "original": "...",
      "rewritten": "...",
      "reason": "one short phrase like 'surfaced LLM and production deployment keywords'",
      "warnings": ["optional list of constraints you chose to relax and why"]
    }
  ],
  "unfulfillable_keywords": ["optional list of JD keywords the candidate truly doesn't have"]
}

Targets: aim to materially rewrite AT LEAST 70% of the supplied bullets. The "reason" field for unchanged bullets must be a real explanation like "already optimal for these JD keywords" or "no truthful angle to surface remaining missing terms" — not a default.
"""


# ── Post-LLM guards ──────────────────────────────────────────────────────
# Each guard returns (ok, reason). ok=False means the rewrite should be
# REJECTED (keep original). The caller logs the reason for audit.

# Numbers we care about preserving: percentages, "5+", "10K+", "100%",
# decimals, multi-digit counts.
_NUM_RE = re.compile(r"~?\d+(?:\.\d+)?[KkMm]?\+?%?")

# Multi-word proper nouns we must not collapse. Match is case-insensitive
# but only the spaces matter for the collapse check.
_BRAND_PAIRS: List[str] = [
    "Spring Boot",
    "Gemini 2.0",
    "Gemini 1.5",
    "Gemini 2.5",
    "GPT-4o",
    "Next.js 16",
    "Next.js 15",
    "Next.js 14",
    "React 19",
    "React 18",
    "Hugging Face Spaces",
    "Hugging Face",
    "LangGraph",
    "LangSmith",
    "LangChain",
    "ChromaDB",
    "FAISS",
    "LoRA",
    "QLoRA",
    "Arcade AI",
    "Tavily",
    "Bitbucket",
    "ElevenLabs",
    "Spring",
    "Postman",
    "Jira",
    "Deccan AI Catalyst Hackathon",
    "Deccan AI",
    "PC hardware",
    "Windows/Linux",
]

# Hanging participles that indicate a truncated bullet ("...supporting" with
# no object). Trigger only if the bullet ends in one of these without a
# following noun phrase within 30 chars (we just check the literal end).
_HANGING_PARTICIPLES = (
    "supporting",
    "delivering",
    "implementing",
    "ensuring",
    "enabling",
    "improving",
    "leveraging",
    "providing",
    "creating",
    "developing",
    "building",
    "designing",
    "managing",
    "leading",
    "driving",
    "utilizing",
    "using",
    "integrating",
    "performing",
    "executing",
    "increasing",
    "decreasing",
    "supporting,",
)

# Participatory verbs that may NOT be promoted to leadership verbs.
_PARTICIPATORY = {"contributed", "participated", "assisted", "supported", "helped"}
_LEADERSHIP = {"led", "conducted", "owned", "headed", "directed", "spearheaded"}


def check_numbers(original: str, rewritten: str) -> Tuple[bool, str]:
    """Every number in the original must reappear verbatim in the rewrite."""
    orig_nums = _NUM_RE.findall(original)
    rew_nums = _NUM_RE.findall(rewritten)
    missing = [n for n in orig_nums if n not in rew_nums]
    if missing:
        return False, f"missing_numbers={missing}"
    return True, ""


def check_proper_nouns(original: str, rewritten: str) -> Tuple[bool, str]:
    """Multi-word brand tokens present in the original (with spaces) MUST
    appear with their spaces intact in the rewrite. Catches "Spring Boot"
    → "SpringBoot", "Gemini 2.0" → "Gemini2.0", etc."""
    low_orig = original.lower()
    low_rew = rewritten.lower()
    collapsed = []
    for brand in _BRAND_PAIRS:
        if " " not in brand:
            continue
        bl = brand.lower()
        # Only check brands actually in the original
        if bl not in low_orig:
            continue
        if bl in low_rew:
            continue
        # See if the collapsed version (spaces removed) appears
        glued = bl.replace(" ", "")
        if glued in low_rew:
            collapsed.append(brand)
    if collapsed:
        return False, f"collapsed_brands={collapsed}"
    return True, ""


def check_completeness(rewritten: str) -> Tuple[bool, str]:
    """Reject rewrites that end in a hanging participle. We check the last
    word against a known list of dangling gerunds."""
    if not rewritten:
        return False, "empty"
    cleaned = rewritten.strip().rstrip(".;:,")
    # Last word, case-folded
    last_word = re.split(r"\s+", cleaned)[-1].lower() if cleaned else ""
    if last_word in _HANGING_PARTICIPLES:
        return False, f"hanging_participle={last_word!r}"
    return True, ""


def check_length(original: str, rewritten: str) -> Tuple[bool, str]:
    """Reject rewrites that lost more than 45% of the original length —
    the rewriter likely truncated and dropped achievement clauses."""
    if not rewritten:
        return False, "empty"
    if len(original) >= 20 and len(rewritten) < 0.55 * len(original):
        return False, f"too_short ({len(rewritten)} < 0.55 * {len(original)})"
    return True, ""


def check_seniority(original: str, rewritten: str) -> Tuple[bool, str]:
    """Block participatory→leadership verb swaps."""
    # Skip a leading bullet glyph + whitespace.
    def _first_word(s: str) -> str:
        s = s.lstrip("•●▪◦■□▶►–—-· *\t ")
        m = re.match(r"\w+", s)
        return m.group(0).lower() if m else ""

    orig_v = _first_word(original)
    rew_v = _first_word(rewritten)
    if orig_v in _PARTICIPATORY and rew_v in _LEADERSHIP:
        return False, f"verb_inflation: {orig_v!r}->{rew_v!r}"
    return True, ""


def _validate_rewrite(original: str, rewritten: str) -> Tuple[bool, List[str]]:
    """Run all guards on a (original, rewritten) pair. Returns (ok, reasons).
    A False ok means the caller should KEEP the original instead."""
    reasons: List[str] = []
    for fn in (check_numbers, check_proper_nouns, check_length, check_seniority):
        ok, reason = fn(original, rewritten)
        if not ok:
            reasons.append(reason)
    ok, reason = check_completeness(rewritten)
    if not ok:
        reasons.append(reason)
    return (len(reasons) == 0, reasons)


def _build_rewriter_llm() -> LLM:
    """Build the LLM the rewriter uses when no BYOK override is provided.

    Honors REWRITER_MODEL_OVERRIDE env var for forcing a specific model name
    (useful when one provider follows constraints better than the auto
    chain's default). Always uses fast=False — rewriter work is constrained
    editing, not creative writing, so the strong model is appropriate.
    """
    override = os.environ.get("REWRITER_MODEL_OVERRIDE", "").strip()
    if override:
        return LLM(model=override, fast=False)
    return LLM(fast=False)


def _select_rewriteable(paragraphs: List[ResumeParagraph]) -> List[ResumeParagraph]:
    """Pick the paragraphs we'd ever want to rewrite.

    Sections that are FACTUAL and short (certifications, education) are
    excluded outright — rewriting "Oracle Cloud Infrastructure Certified" or
    a degree line adds zero JD-match value and risks breaking the title.
    """
    NO_TOUCH_SECTIONS = {"certifications", "education"}
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
        # Hard exclude factual sections — bullet glyph in front of a cert
        # line shouldn't pull it into the rewriter.
        if p.section in NO_TOUCH_SECTIONS:
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
    *,
    llm: "LLM | None" = None,
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

    if llm is None:
        llm = _build_rewriter_llm()
    all_instructions: List[RewriteInstruction] = []
    budgets: Dict[int, int] = {p.index: _budget_for(p, (length_hints or {}).get(p.index)) for p in targets}

    # Build the "JD keywords to surface" list once: must-haves + general
    # keywords, deduped. This is the explicit vocabulary the rewriter should
    # try to weave into bullets when the candidate truthfully has the work.
    jd_vocab = _dedupe(list(jd.must_have_skills) + list(jd.keywords))

    # Originals keyed by paragraph index — needed by the post-LLM guards so
    # we can compare numbers/proper-nouns/verbs against the source.
    originals_by_idx: Dict[int, str] = {p.index: p.text for p in targets}

    # Track rejection stats for telemetry.
    total_rewrites = 0
    total_rejected = 0
    rejection_reasons: List[str] = []

    # Process in small batches of 6 paragraphs. Smaller batches give each
    # bullet more attention (less serialization drift) at the cost of more
    # LLM calls. The previous batch size of 24 caused truncation in the
    # tail of a single response on free-tier providers.
    batch_num = 0
    for batch in _chunk(targets, 6):
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
            # temperature=0.05: rewriter is constrained editing, not creative
            # writing. Lower temp tightens adherence to STRICT MODE.
            # max_tokens=6000: enough headroom for 6 bullets of structured
            # JSON without truncation risk on free-tier providers.
            data = llm.complete_json(
                system=SYSTEM,
                user=user,
                max_tokens=6000,
                temperature=0.05,
            )
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
                instruction = RewriteInstruction(**{
                    k: v for k, v in ins.items()
                    if k in {"paragraph_index", "original", "rewritten", "reason"}
                })
            except Exception:
                continue

            # Enforce the per-paragraph char budget before validation. We
            # still truncate at word boundaries here, then re-validate so a
            # truncation that breaks numbers/brands gets caught.
            budget = budgets.get(instruction.paragraph_index)
            if budget and len(instruction.rewritten) > budget:
                instruction.rewritten = _truncate(instruction.rewritten, budget)

            # Post-LLM validation. If any guard rejects, revert to original.
            original_text = originals_by_idx.get(instruction.paragraph_index, instruction.original)
            # Skip validation entirely if the LLM left this paragraph
            # unchanged — there's nothing to check, and we want to allow
            # "no truthful angle" no-ops through cleanly.
            rewritten = (instruction.rewritten or "").strip()
            if rewritten and rewritten != original_text.strip():
                total_rewrites += 1
                ok, reasons = _validate_rewrite(original_text, instruction.rewritten)
                if not ok:
                    total_rejected += 1
                    rejection_reasons.extend(reasons)
                    logger.info(
                        "Rewriter batch %d: REJECTED paragraph %d (%s). Keeping original.",
                        batch_num, instruction.paragraph_index, "; ".join(reasons),
                    )
                    # Revert to original: the writer treats rewritten == original
                    # as a no-op (no redaction, no insert).
                    instruction.rewritten = original_text
                    instruction.reason = (instruction.reason or "") + f" [reverted: {'; '.join(reasons)}]"

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

    if total_rewrites > 0:
        logger.info(
            "Rewriter rejection rate: %d/%d (%.0f%%). Top reasons: %s",
            total_rejected, total_rewrites,
            100.0 * total_rejected / total_rewrites,
            "; ".join(rejection_reasons[:8]),
        )

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
