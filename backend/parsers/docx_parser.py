"""Parse .docx into a logical representation while keeping a reference to the
original document so the writer can put rewritten text back in the same place
without disturbing formatting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document
from docx.document import Document as DocxDocument
from docx.text.paragraph import Paragraph

from backend.models import ResumeParagraph, ResumeAnalysis


# Section headers we look for. Case-insensitive substring match.
SECTION_KEYWORDS = {
    "summary": ["summary", "profile", "objective", "about"],
    "experience": ["experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "projects": ["projects", "selected projects", "personal projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "awards": ["awards", "honors", "achievements"],
    "publications": ["publications", "papers"],
}


@dataclass
class ParsedDocx:
    """Holds both the python-docx Document and a logical view of it."""
    document: DocxDocument
    paragraphs: List[ResumeParagraph]
    full_text: str


def _classify_paragraph(p: Paragraph, idx: int, current_section: Optional[str]) -> Tuple[ResumeParagraph, Optional[str]]:
    """Return a ResumeParagraph and the (possibly updated) current section."""
    text = p.text.strip()
    style_name = p.style.name if p.style is not None else None

    # Detect heading
    is_heading = False
    if style_name and "Heading" in style_name:
        is_heading = True
    # Heuristic: short ALL CAPS or Title Case lines are often section headers
    if text and len(text) <= 40:
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, sum(1 for c in text if c.isalpha()))
        if upper_ratio > 0.7 and len(text.split()) <= 5:
            is_heading = True

    # Detect section by keyword match
    section = current_section
    if is_heading or (text and len(text) <= 40):
        lowered = text.lower()
        for key, kws in SECTION_KEYWORDS.items():
            if any(kw in lowered for kw in kws):
                section = key
                is_heading = True
                break

    # Bullet detection: docx bullets usually live in list paragraph styles,
    # but plenty of resumes just use a leading dash/dot character.
    is_bullet = False
    if style_name and ("List" in style_name or "Bullet" in style_name):
        is_bullet = True
    elif text.startswith(("•", "●", "·", "-", "*", "▪", "◦")):
        is_bullet = True

    return (
        ResumeParagraph(
            index=idx,
            text=text,
            style=style_name,
            is_heading=is_heading,
            section=section,
            is_bullet=is_bullet,
        ),
        section,
    )


def parse_docx(path: str | Path) -> ParsedDocx:
    """Parse a .docx file and return both the original Document and a logical view."""
    doc = Document(str(path))
    paragraphs: List[ResumeParagraph] = []
    current_section: Optional[str] = None

    # Walk paragraphs in body
    for idx, para in enumerate(doc.paragraphs):
        rp, current_section = _classify_paragraph(para, idx, current_section)
        paragraphs.append(rp)

    # Also walk tables (many resumes have a left sidebar in a table)
    table_offset = len(doc.paragraphs)
    for t_idx, table in enumerate(doc.tables):
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                for p_idx, para in enumerate(cell.paragraphs):
                    composite_idx = table_offset + 1000 * t_idx + 100 * r_idx + 10 * c_idx + p_idx
                    rp, current_section = _classify_paragraph(para, composite_idx, current_section)
                    paragraphs.append(rp)

    full_text = "\n".join(p.text for p in paragraphs if p.text)

    return ParsedDocx(document=doc, paragraphs=paragraphs, full_text=full_text)


def parsed_to_analysis(parsed: ParsedDocx) -> ResumeAnalysis:
    """Build a ResumeAnalysis from a ParsedDocx using lightweight heuristics."""
    analysis = ResumeAnalysis(paragraphs=parsed.paragraphs)
    sections_seen = set()

    # First non-empty line is usually the candidate name
    for p in parsed.paragraphs:
        if p.text:
            analysis.candidate_name = p.text
            break

    # Simple contact extraction from the first ~5 paragraphs
    import re
    head_text = "\n".join(p.text for p in parsed.paragraphs[:8])
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", head_text)
    phone_match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", head_text)
    linkedin_match = re.search(r"(linkedin\.com/in/[\w-]+)", head_text, re.IGNORECASE)
    github_match = re.search(r"(github\.com/[\w-]+)", head_text, re.IGNORECASE)

    if email_match:
        analysis.contact["email"] = email_match.group(0)
    if phone_match:
        analysis.contact["phone"] = phone_match.group(0).strip()
    if linkedin_match:
        analysis.contact["linkedin"] = linkedin_match.group(0)
    if github_match:
        analysis.contact["github"] = github_match.group(0)

    # Bucket paragraphs by section
    for p in parsed.paragraphs:
        if not p.text or p.is_heading:
            continue
        sec = p.section
        if sec:
            sections_seen.add(sec)
        if sec == "summary":
            if analysis.summary:
                analysis.summary += " " + p.text
            else:
                analysis.summary = p.text
        elif sec == "experience" and p.is_bullet:
            analysis.experience_bullets.append(p.text)
        elif sec == "education":
            analysis.education.append(p.text)
        elif sec == "projects" and p.is_bullet:
            analysis.projects.append(p.text)
        elif sec == "skills":
            # Skills are often comma- or pipe-separated
            for token in re.split(r"[,;|/]", p.text):
                token = token.strip()
                if token and len(token) <= 40:
                    analysis.skills.append(token)

    analysis.sections_found = sorted(sections_seen)
    return analysis
