"""Pydantic models used across the pipeline."""
from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class JDAnalysis(BaseModel):
    """Structured view of a job description."""
    role_title: str = ""
    seniority: str = ""  # junior / mid / senior / staff / lead / etc.
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)  # ATS keywords
    responsibilities: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    culture_signals: List[str] = Field(default_factory=list)
    company_name: Optional[str] = None
    summary: str = ""


class ResumeParagraph(BaseModel):
    """One paragraph extracted from the resume with positional info."""
    index: int
    text: str
    style: Optional[str] = None
    is_heading: bool = False
    section: Optional[str] = None  # e.g. "experience", "education"
    is_bullet: bool = False


class ResumeAnalysis(BaseModel):
    """Logical view of the resume independent of formatting."""
    candidate_name: str = ""
    contact: Dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    experience_bullets: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    sections_found: List[str] = Field(default_factory=list)
    paragraphs: List[ResumeParagraph] = Field(default_factory=list)


class MatchReport(BaseModel):
    """Gap analysis between resume and JD."""
    covered: List[str] = Field(default_factory=list)
    weakly_covered: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    overall_score: float = 0.0  # 0..1
    rewrite_priorities: List[str] = Field(default_factory=list)
    notes: str = ""


class RewriteInstruction(BaseModel):
    """Per-paragraph rewrite directive."""
    paragraph_index: int
    original: str
    rewritten: str
    reason: str = ""


class RewriteResult(BaseModel):
    instructions: List[RewriteInstruction] = Field(default_factory=list)


class ATSScore(BaseModel):
    keyword_coverage: float = 0.0  # 0..1
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class TailorResponse(BaseModel):
    download_url: str
    output_format: str = "docx"  # "docx" or "pdf"
    match_score: float
    jd_analysis: JDAnalysis
    match_report: MatchReport
    ats_score: ATSScore
    cover_letter: Optional[str] = None
    cover_letter_url: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class CoverLetterResponse(BaseModel):
    text: str
    download_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    model: str = ""
