"""Orchestrator: runs the full multi-agent pipeline end-to-end.

A post-merge sanity pass (`_sanitize_instructions`) runs after all rewriter
passes finish, before handing the instructions to the writer. It cleans up
mechanical artifacts the rewriter LLM sometimes produces: double spaces,
digit-glued-to-unit ("8minutes"), and brand pairs glued together
("SpringBoot", "Gemini2.0").

Sequence:
    1. Parse the uploaded resume (heuristic, no LLM).
       - .docx  -> ParsedDocx
       - .pdf   -> ParsedPdf (span/bbox-aware so we can edit in place)
    2. JD Analyzer    -> JDAnalysis
    3. Resume Analyzer -> ResumeAnalysis (merged with the heuristic view)
    4. Matcher        -> MatchReport
    5. Rewriter       -> RewriteResult, with per-paragraph char budgets
       so PDF rewrites fit in the original bounding box.
    6. Writer (DOCX in-place run replacement OR PDF in-place redaction)
       -> tailored file on disk in the SAME format as the input.
    7. ATS Optimizer  -> ATSScore on the rewritten text.
    8. (Optional) one re-rewrite pass if ATS coverage is below threshold.
    9. (Optional) Cover Letter agent + PDF render.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from backend.config import get_settings

from backend.agents import (
    ats_optimizer,
    cover_letter as cover_letter_agent,
    jd_analyzer,
    matcher,
    resume_analyzer,
    rewriter,
)
from backend.agents.llm import LLM
from backend.models import (
    ATSScore,
    JDAnalysis,
    MatchReport,
    ResumeAnalysis,
    ResumeParagraph,
    RewriteResult,
)
from backend.parsers.docx_parser import ParsedDocx, parse_docx, parsed_to_analysis
from backend.writers.docx_writer import apply_rewrites as apply_rewrites_docx

logger = logging.getLogger(__name__)


ATS_THRESHOLD = 0.65  # if coverage is below this, attempt one re-rewrite pass


@dataclass
class PipelineResult:
    """Bundled return type so callers don't break when fields are added."""
    output_path: Path
    output_format: str  # "docx" or "pdf"
    jd: JDAnalysis
    resume: ResumeAnalysis
    match_report: MatchReport
    ats_score: ATSScore
    cover_letter_text: Optional[str] = None
    cover_letter_pdf_path: Optional[Path] = None
    # Diagnostic message set when a non-critical step was skipped due to the
    # request-wide time budget (MAX_REQUEST_SECONDS). Empty when nothing
    # was skipped.
    cover_letter_error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _rewritten_text_for_ats(paragraphs: List[ResumeParagraph], result: RewriteResult) -> str:
    by_idx = {ins.paragraph_index: ins.rewritten for ins in result.instructions}
    parts = [by_idx.get(p.index, p.text) for p in paragraphs]
    return "\n".join(t for t in parts if t)


def _length_hints_from_pdf(parsed_pdf) -> Dict[int, int]:
    """For each PDF paragraph, the original text length is the safe upper
    bound for a rewrite that should fit in the same bbox. We add a small
    margin (10%) and floor at 60 chars to give the LLM some room to work."""
    from backend.parsers.pdf_parser import _pack_index
    hints: Dict[int, int] = {}
    for blk in parsed_pdf.blocks:
        idx = _pack_index(blk.page_idx, blk.block_idx, blk.sub_idx)
        text_len = sum(len(s.text) for s in blk.spans)
        hints[idx] = max(60, int(text_len * 1.1))
    return hints


# ── Post-merge sanity pass ───────────────────────────────────────────────
# Mechanical cleanup applied to every rewrite after the rewriter has run.
# These are NOT meaning-changing fixes; they're glyph/spacing repairs the
# LLM sometimes produces under low temperature.

# Units that should always have a space between the preceding digit and
# the unit word. Plural and abbreviated forms covered.
_DIGIT_UNIT_RE = re.compile(
    r"(\d)(?=(?:minute|minutes|second|seconds|hour|hours|day|days|week|weeks|month|months|year|years|min|mins|sec|secs|hr|hrs|yr|yrs)\b)",
    re.IGNORECASE,
)

# Brand pairs we re-space when collapsed. Keep narrowly scoped to known
# multi-word product/framework names so we don't accidentally split a
# legitimate token (e.g. "GPT-4o", "Next.js", "v2"). Order matters: longer
# patterns first so "Hugging Face Spaces" beats "Hugging Face".
_BRAND_RESPACE = [
    (re.compile(r"\bHuggingFaceSpaces\b", re.IGNORECASE), "Hugging Face Spaces"),
    (re.compile(r"\bHuggingFace\b", re.IGNORECASE), "Hugging Face"),
    (re.compile(r"\bSpringBoot\b", re.IGNORECASE), "Spring Boot"),
    (re.compile(r"\bArcadeAI\b", re.IGNORECASE), "Arcade AI"),
    (re.compile(r"\bDeccanAICatalystHackathon\b", re.IGNORECASE), "Deccan AI Catalyst Hackathon"),
    (re.compile(r"\bDeccanAI\b", re.IGNORECASE), "Deccan AI"),
    # Gemini X.Y — only when the digit is glued, never when there's
    # already a space.
    (re.compile(r"\bGemini(\d+\.\d+)\b"), r"Gemini \1"),
    (re.compile(r"\bReact(1[0-9])\b"), r"React \1"),
    (re.compile(r"\bNext\.js(1[0-9])\b"), r"Next.js \1"),
]


def _sanitize_text(text: str) -> str:
    """Mechanical post-rewrite cleanup. Idempotent — safe to run twice."""
    if not text:
        return text
    out = text
    # 1. Collapse runs of whitespace to single spaces (keeps newlines intact).
    out = re.sub(r"[ \t]{2,}", " ", out)
    # 2. Re-space digit→unit pairs ("8minutes" → "8 minutes").
    out = _DIGIT_UNIT_RE.sub(r"\1 ", out)
    # 3. Re-space collapsed brand pairs.
    for pat, repl in _BRAND_RESPACE:
        out = pat.sub(repl, out)
    return out


def _sanitize_instructions(rewrites: "RewriteResult") -> None:
    """Apply _sanitize_text to every instruction's rewritten field in place."""
    for ins in rewrites.instructions:
        if ins.rewritten:
            cleaned = _sanitize_text(ins.rewritten)
            if cleaned != ins.rewritten:
                ins.rewritten = cleaned


def run_pipeline(
    resume_docx_path: Union[str, Path],
    jd_text: str,
    output_path: Union[str, Path],
    want_cover_letter: bool = False,
    cover_letter_pdf_path: Optional[Union[str, Path]] = None,
    llm_override: Optional[LLM] = None,
) -> PipelineResult:
    """Run the full pipeline. Input may be .docx or .pdf; output format
    matches the input.

    The output_path's extension is overridden to match the input format so
    callers that hard-coded .docx still produce a valid file.
    """
    # Request-wide wall-clock budget. Non-critical steps (second rewriter
    # pass, cover letter) skip with a warning instead of 503-ing the entire
    # pipeline if we're running long. The LLM layer also honors this
    # `deadline` via its kwarg so a slow chain of provider fallbacks can't
    # block the whole run.
    settings = get_settings()
    start_time = time.time()
    deadline = start_time + float(settings.max_request_seconds)

    src = Path(resume_docx_path)
    suffix = src.suffix.lower()
    is_pdf = suffix == ".pdf"

    # Normalize output path extension to match input
    out_path = Path(output_path)
    if is_pdf and out_path.suffix.lower() != ".pdf":
        out_path = out_path.with_suffix(".pdf")
    elif not is_pdf and out_path.suffix.lower() != ".docx":
        out_path = out_path.with_suffix(".docx")

    logger.info("Parsing resume (%s): %s", "pdf" if is_pdf else "docx", src)

    cover_letter_error: Optional[str] = None
    warnings: List[str] = []
    length_hints: Optional[Dict[int, int]] = None
    parsed_docx: Optional[ParsedDocx] = None
    parsed_pdf = None

    if is_pdf:
        from backend.parsers.pdf_parser import parse_pdf, parsed_pdf_to_analysis
        parsed_pdf = parse_pdf(src)
        warnings.extend(parsed_pdf.warnings)
        heuristic = parsed_pdf_to_analysis(parsed_pdf)
        paragraphs = parsed_pdf.paragraphs
        full_text = parsed_pdf.full_text
        length_hints = _length_hints_from_pdf(parsed_pdf)
    else:
        parsed_docx = parse_docx(src)
        heuristic = parsed_to_analysis(parsed_docx)
        paragraphs = parsed_docx.paragraphs
        full_text = parsed_docx.full_text

    logger.info("Analyzing JD")
    jd = jd_analyzer.analyze(jd_text, llm=llm_override)

    logger.info("Analyzing resume")
    resume = resume_analyzer.analyze(full_text, heuristic=heuristic, llm=llm_override)
    resume.paragraphs = paragraphs  # LLM doesn't have indices

    logger.info("Running matcher")
    match_report = matcher.match(jd, resume, llm=llm_override)
    # Empty-report guard: if all three buckets are empty AND overall_score
    # is 0, the LLM probably returned malformed JSON. Retry exactly once
    # before continuing — the rewriter is useless without a real report.
    if (
        not match_report.covered
        and not match_report.weakly_covered
        and not match_report.missing
        and match_report.overall_score == 0.0
    ):
        logger.warning("Matcher returned an empty report; retrying once.")
        match_report = matcher.match(jd, resume, llm=llm_override)

    logger.info("Running rewriter (length_hints=%s)", "yes" if length_hints else "no")
    rewrites = rewriter.rewrite(jd, resume, match_report, paragraphs, length_hints=length_hints, llm=llm_override)

    rewritten_text = _rewritten_text_for_ats(paragraphs, rewrites)
    ats = ats_optimizer.score(rewritten_text, jd, llm=llm_override)
    logger.info("Initial ATS coverage: %.2f", ats.keyword_coverage)

    # Time-budget gate on the optional second rewriter pass. If we're
    # already past the deadline, skip pass-2 — the user gets pass-1's
    # rewrites instead of a 503 or a hung request.
    if time.time() >= deadline:
        logger.warning(
            "Pipeline past time budget (%.0fs elapsed); skipping pass-2 rewrite + cover letter.",
            time.time() - start_time,
        )
        skip_pass2 = True
    else:
        skip_pass2 = False

    if not skip_pass2 and ats.keyword_coverage < ATS_THRESHOLD and ats.missing_keywords:
        logger.info("Coverage below threshold, doing a second rewriter pass")
        # Build a minimal "boost" report — just the missing keywords as
        # priorities. The full match_report would blow past Groq's 8K TPM
        # cap, so we keep this very lean. The rewriter will still see the
        # full JD analysis via the `jd` argument.
        boosted = MatchReport(
            covered=[],
            weakly_covered=ats.missing_keywords[:20],
            missing=[],
            overall_score=match_report.overall_score,
            rewrite_priorities=[
                f"Surface these missing JD keywords if truthful: {', '.join(ats.missing_keywords[:20])}",
            ],
            notes="",
        )
        pass2 = rewriter.rewrite(jd, resume, boosted, paragraphs, length_hints=length_hints, llm=llm_override)

        # CRITICAL: pass 2 is ADDITIVE, not a replacement. If pass 2 fails or
        # returns nothing, we MUST keep pass 1's work. A pass 2 instruction
        # only wins over pass 1 if it actually changed the bullet (not a
        # no-op) — otherwise pass 1's earlier change stays.
        merged_by_idx = {ins.paragraph_index: ins for ins in rewrites.instructions}
        for ins in pass2.instructions:
            new_changed = ins.rewritten and ins.rewritten.strip() != ins.original.strip()
            if new_changed:
                merged_by_idx[ins.paragraph_index] = ins
            # else: pass 2 returned no-op; keep pass 1's instruction if any.

        from backend.models import RewriteResult as _RewriteResult
        rewrites = _RewriteResult(instructions=list(merged_by_idx.values()))
        rewritten_text = _rewritten_text_for_ats(paragraphs, rewrites)
        ats = ats_optimizer.score(rewritten_text, jd, llm=llm_override)
        logger.info(
            "Post-rewrite ATS coverage: %.2f (merged: %d pass-1 + %d pass-2 additions = %d total)",
            ats.keyword_coverage,
            len(rewrites.instructions) - sum(1 for i in pass2.instructions if i.rewritten and i.rewritten.strip() != i.original.strip()),
            sum(1 for i in pass2.instructions if i.rewritten and i.rewritten.strip() != i.original.strip()),
            len(rewrites.instructions),
        )

    # Final post-merge sanity pass: mechanical cleanup (double spaces,
    # digit-unit gluing, brand pair gluing) before handing to the writer.
    _sanitize_instructions(rewrites)

    # Dispatch to the right writer
    logger.info("Writing tailored file to %s", out_path)
    if is_pdf:
        from backend.writers.pdf_writer import apply_rewrites as apply_rewrites_pdf
        out = apply_rewrites_pdf(parsed_pdf, rewrites.instructions, out_path)
        # Close the PyMuPDF document after saving (it's now on disk)
        try:
            parsed_pdf.document.close()
        except Exception:
            pass
    else:
        out = apply_rewrites_docx(parsed_docx, rewrites.instructions, out_path)

    cover_text: Optional[str] = None
    cover_pdf: Optional[Path] = None
    if want_cover_letter:
        # Time-budget gate: if we're already past the deadline, skip the
        # cover letter rather than 503-ing. The resume tailoring is the
        # primary product; the letter is a bonus.
        if time.time() >= deadline:
            cover_letter_error = (
                "Cover letter skipped: request exceeded the time budget. "
                "Try again — usually transient, often due to provider rate limits."
            )
            logger.warning(cover_letter_error)
        else:
            logger.info("Generating cover letter")
            try:
                cover_text = cover_letter_agent.generate(jd, resume, llm=llm_override)
            except Exception as e:
                logger.exception("Cover letter generation failed")
                cover_letter_error = f"Cover letter generation failed: {type(e).__name__}"
                cover_text = None
        if cover_text and cover_letter_pdf_path is not None:
            try:
                from backend.writers.cover_letter_pdf import render_cover_letter_pdf
                cover_pdf = render_cover_letter_pdf(
                    cover_text,
                    cover_letter_pdf_path,
                    candidate_name=resume.candidate_name or None,
                    contact=resume.contact or None,
                )
            except Exception:
                logger.exception("Cover letter PDF render failed")
                cover_pdf = None

    logger.info(
        "Pipeline complete in %.1fs (budget=%.0fs)",
        time.time() - start_time, float(settings.max_request_seconds),
    )

    return PipelineResult(
        output_path=out,
        output_format="pdf" if is_pdf else "docx",
        jd=jd,
        resume=resume,
        match_report=match_report,
        ats_score=ats,
        cover_letter_text=cover_text,
        cover_letter_pdf_path=cover_pdf,
        cover_letter_error=cover_letter_error,
        warnings=warnings,
    )
