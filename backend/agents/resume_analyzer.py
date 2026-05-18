"""Agent: extract logical structure from a resume's plain text.

We already do a lot of this heuristically in the docx parser. This agent runs
on top of that to fill gaps (e.g. extracting summary text out of a profile
section that has no heading, or detecting skills inline in bullets).
"""
from __future__ import annotations

from backend.agents.llm import LLM
from backend.models import ResumeAnalysis


SYSTEM = """You are a resume parsing specialist. You receive the plain text of a candidate's resume and return a structured view.

Return ONLY valid JSON matching this schema:
{
  "candidate_name": "string",
  "contact": {"email": "string|null", "phone": "string|null", "linkedin": "string|null", "github": "string|null", "location": "string|null"},
  "summary": "the candidate's profile / objective paragraph, or empty string",
  "skills": ["string"],
  "experience_bullets": ["string"],
  "education": ["string"],
  "projects": ["string"],
  "sections_found": ["string"]
}

Rules:
- "experience_bullets" should be the verbatim bullet text (you may strip leading dashes/dots).
- "skills" are individual technologies/tools/methods, not categories.
- Do not invent content the resume doesn't contain.
- Return JSON only, no prose, no code fences.
"""


def analyze(resume_text: str, heuristic: ResumeAnalysis | None = None, *, llm: "LLM | None" = None) -> ResumeAnalysis:
    if not resume_text or not resume_text.strip():
        return heuristic or ResumeAnalysis()

    if llm is None:
        llm = LLM()
    data = llm.complete_json(
        system=SYSTEM,
        user=f"Resume text:\n\n{resume_text}",
        max_tokens=4000,
    )
    if not isinstance(data, dict):
        return heuristic or ResumeAnalysis()

    # Merge with the heuristic view (heuristic carries the paragraphs list)
    merged = (heuristic.model_dump() if heuristic else {})
    for k, v in data.items():
        if v in (None, "", [], {}):
            continue
        merged[k] = v
    return ResumeAnalysis(**merged)
