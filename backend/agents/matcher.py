"""Agent: gap analysis between resume and JD."""
from __future__ import annotations

from backend.agents.llm import LLM
from backend.models import JDAnalysis, ResumeAnalysis, MatchReport


SYSTEM = """You are a hiring manager doing first-pass resume screening. Compare a candidate's resume to the job description and produce a gap analysis.

Return ONLY valid JSON matching this schema:
{
  "covered": ["JD requirements that the resume already demonstrates well"],
  "weakly_covered": ["JD requirements the resume touches but should emphasize more"],
  "missing": ["JD requirements the resume does not address at all"],
  "overall_score": 0.0,
  "rewrite_priorities": ["specific guidance for the rewriter, in priority order"],
  "notes": "short paragraph of overall assessment"
}

CLASSIFICATION RULES — be generous, not literal:

A requirement is COVERED if any of the following are true:
  - The resume names the exact technology/skill, OR
  - The resume names a specific instance/synonym that obviously implies it. Examples:
      * "ChromaDB" or "FAISS" or "Pinecone" implies "vector databases" / "embeddings retrieval"
      * "GPT-4o" or "Claude" or "Gemini" implies "LLMs" / "large language models"
      * "LangChain" or "LangGraph" or "CrewAI" implies "agent orchestration" / "multi-agent systems"
      * "Spring Boot + REST APIs" implies "backend development" / "microservices"
      * "Docker + GCP" implies "containerization" / "cloud deployment"
      * "Stripe payment integration" implies "third-party API integration" / "payment systems"
      * "FastAPI" implies "Python web framework" / "API development"
      * A RAG pipeline implies "retrieval", "embeddings", "prompt engineering"
      * Building a multi-agent system implies "system design", "agent architecture"

A requirement is WEAKLY_COVERED if the resume hints at it but doesn't make it prominent (e.g. mentioned once in passing, or implied but the JD asks for explicit experience).

A requirement is MISSING only if there's no truthful angle in the resume at all. Be careful: do not mark something MISSING if the resume contains evidence of it under a different name. When in doubt about whether to label something COVERED or WEAKLY_COVERED, prefer WEAKLY_COVERED — that signals the rewriter to surface it more explicitly.

Other rules:
- overall_score is a float 0.0 to 1.0. Weight COVERED items most, WEAKLY_COVERED partially, MISSING heavily. 0.8+ = strong, 0.5-0.8 = workable with tailoring, <0.5 = likely poor fit.
- rewrite_priorities should be concrete and actionable, e.g. "In NewsXsys bullet 1, swap 'multi-agent workflows' for 'agent orchestration' to match JD vocabulary" or "Surface 'vector databases' alongside the ChromaDB mention".
- For MISSING items, note explicitly: "Cannot be addressed by rewriting — candidate would need to add real experience."
- Return JSON only, no prose, no code fences.
"""


def match(jd: JDAnalysis, resume: ResumeAnalysis) -> MatchReport:
    # Trim the resume payload to the fields the matcher actually needs.
    # Excluding `paragraphs` (positional, can be huge) keeps us under
    # Groq's per-request token cap that was 413-ing on content-rich resumes.
    trimmed_resume = resume.model_dump(
        exclude={"paragraphs", "sections_found"}
    )
    user = f"""Job description analysis:
{jd.model_dump_json(indent=2)}

Candidate resume analysis:
{_compact_json(trimmed_resume)}
"""
    llm = LLM()
    data = llm.complete_json(system=SYSTEM, user=user, max_tokens=2500)
    if not isinstance(data, dict):
        return MatchReport()
    return MatchReport(**{k: v for k, v in data.items() if v is not None})


def _compact_json(data) -> str:
    import json
    return json.dumps(data, indent=2, ensure_ascii=False)
