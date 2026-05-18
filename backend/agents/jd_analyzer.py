"""Agent: extract structured information from a job description."""
from __future__ import annotations

from backend.agents.llm import LLM
from backend.models import JDAnalysis


SYSTEM = """You are an expert technical recruiter and ATS analyst. You receive a raw job description and extract structured information that will be used to tailor a resume.

Return ONLY valid JSON matching this schema:
{
  "role_title": "string",
  "seniority": "junior|mid|senior|staff|lead|principal|other",
  "must_have_skills": ["string"],
  "nice_to_have_skills": ["string"],
  "keywords": ["string"],
  "responsibilities": ["string"],
  "qualifications": ["string"],
  "culture_signals": ["string"],
  "company_name": "string|null",
  "summary": "one-paragraph plain-language summary of the role"
}

Rules:
- "keywords" should be the exact terms an ATS scanner is most likely to match on (technologies, frameworks, methodologies, certifications, domain terms). Keep them short, atomic, lowercase where applicable.
- "must_have_skills" are non-negotiable requirements. "nice_to_have" are bonuses or "preferred".
- Do not invent skills not mentioned in the JD.
- If something is unclear, return an empty array, not a guess.
- Return JSON only, no prose, no code fences.
"""


def analyze(jd_text: str, *, llm: "LLM | None" = None) -> JDAnalysis:
    if not jd_text or not jd_text.strip():
        return JDAnalysis()
    if llm is None:
        llm = LLM()
    data = llm.complete_json(
        system=SYSTEM,
        user=f"Job description:\n\n{jd_text}",
        max_tokens=2500,
    )
    if not isinstance(data, dict):
        return JDAnalysis()
    # Pydantic will coerce/validate
    return JDAnalysis(**{k: v for k, v in data.items() if v is not None})
