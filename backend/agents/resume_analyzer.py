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

    # Merge with the heuristic view (heuristic carries the paragraphs list).
    # The LLM is told contact fields can be null, so we strip nulls before
    # handing to ResumeAnalysis (its contact: Dict[str, str] rejects None
    # values).
    merged = (heuristic.model_dump() if heuristic else {})
    for k, v in data.items():
        if v in (None, "", [], {}):
            continue
        if k == "contact" and isinstance(v, dict):
            v = {sub_k: sub_v for sub_k, sub_v in v.items()
                 if isinstance(sub_v, str) and sub_v.strip()}
            if not v:
                continue
        merged[k] = v

    # Post-process candidate_name: resumes often have a single "header
    # paragraph" that crams the name AND a headline together ("SIDDIQ AHMED |
    # Generative AI Engineer | LLM Applications"). Without this split the
    # cover-letter renderer prints the whole string as the bold name. We
    # extract the headline into its own field.
    raw_name = (merged.get("candidate_name") or "").strip()
    name, headline = _split_name_headline(raw_name)
    if name:
        merged["candidate_name"] = name
    if headline and not merged.get("candidate_headline"):
        merged["candidate_headline"] = headline

    return ResumeAnalysis(**merged)


def _split_name_headline(raw: str) -> tuple[str, str]:
    """Split a header string into (name, headline).

    Heuristics, in order:
      1. If the string contains a pipe `|`, em-dash, or en-dash, split on
         the first occurrence — the part before is the name, after is the
         headline.
      2. If the first segment is longer than 4 words, fall back to just
         the first 2 words as the name and the remainder as the headline.
      3. If neither, return (raw, "").
    """
    if not raw:
        return "", ""
    import re
    # Step 1: split on first pipe / em-dash / en-dash / bullet separator.
    parts = re.split(r"\s*[|—–•]\s*", raw, maxsplit=1)
    name = parts[0].strip()
    headline = parts[1].strip() if len(parts) > 1 else ""

    # Step 2: if "name" is still too long to be a real name, cut it down.
    name_tokens = name.split()
    if len(name_tokens) > 4:
        head_extra = " ".join(name_tokens[2:])
        name = " ".join(name_tokens[:2])
        # Prepend the chopped portion to any existing headline.
        if headline:
            headline = head_extra + " | " + headline
        else:
            headline = head_extra

    # Clean trailing/leading separator junk from headline.
    headline = headline.strip(" |—–•,.")
    return name, headline
