"""Orchestrator: runs the full multi-agent pipeline end-to-end.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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

    logger.info("Running rewriter (length_hints=%s)", "yes" if length_hints else "no")
    rewrites = rewriter.rewrite(jd, resume, match_report, paragraphs, length_hints=length_hints, llm=llm_override)

    rewritten_text = _rewritten_text_for_ats(paragraphs, rewrites)
    ats = ats_optimizer.score(rewritten_text, jd, llm=llm_override)
    logger.info("Initial ATS coverage: %.2f", ats.keyword_coverage)

    if ats.keyword_coverage < ATS_THRESHOLD and ats.missing_keywords:
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
        logger.info("Generating cover letter")
        try:
            cover_text = cover_letter_agent.generate(jd, resume, llm=llm_override)
        except Exception:
            logger.exception("Cover letter generation failed")
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

    return PipelineResult(
        output_path=out,
        output_format="pdf" if is_pdf else "docx",
        jd=jd,
        resume=resume,
        match_report=match_report,
        ats_score=ats,
        cover_letter_text=cover_text,
        cover_letter_pdf_path=cover_pdf,
        warnings=warnings,
    )
