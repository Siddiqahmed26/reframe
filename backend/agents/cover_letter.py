"""Agent: generate a tailored cover letter."""
from __future__ import annotations

from backend.agents.llm import LLM
from backend.models import JDAnalysis, ResumeAnalysis


SYSTEM = """You write tailored, human-sounding cover letters. Your output is a single cover letter (no JSON, no preamble, no postamble), 220-340 words, three to four short paragraphs:

  - Opening: name the role and the company by name if known, plus one specific hook that ties the candidate's experience to a stated need in the JD.
  - Middle (1-2 paragraphs): two or three concrete achievements from the candidate's resume, mapped to the JD's responsibilities. Use real numbers and names from the resume.
  - Closing: a brief, confident line about wanting to discuss further, and a sign-off.

Rules:
  - Never invent experience. Only reference achievements that appear in the candidate's resume.
  - No clichés ("I am writing to express my interest..."). Open with substance.
  - No em dashes. Use commas, periods, or semicolons.
  - Plain text only. No markdown, no headers, no bullets.
  - Address it "Dear Hiring Manager," if no specific contact is mentioned.
  - End with "Sincerely," followed by the candidate's name from their resume.
"""


def generate(jd: JDAnalysis, resume: ResumeAnalysis) -> str:
    user = f"""Job description analysis:
{jd.model_dump_json(indent=2)}

Candidate resume analysis:
{resume.model_dump_json(indent=2)}

Write the cover letter now.
"""
    # Use the FAST model. Reasons:
    #   1. Cover-letter prose doesn't need the strongest model — it's
    #      single-shot generation with no schema to follow.
    #   2. The fast model on Groq (llama-3.1-8b-instant) has its own
    #      independent daily token quota (500K) separate from the main
    #      model's (gpt-oss-120b @ 200K). So the cover letter still
    #      generates even when the rewriter has exhausted the main quota.
    llm = LLM(fast=True)
    return llm.complete(system=SYSTEM, user=user, max_tokens=1500, temperature=0.4)
