"""Render a cover letter (plain text from the LLM) as a clean one-page PDF.

Uses reportlab's Platypus so paragraphs flow and wrap naturally. The layout
mirrors a standard business letter:

  [candidate name + contact line, right-aligned or centered header]
  [today's date]
  [blank line]
  Dear Hiring Manager,
  [body paragraphs]
  Sincerely,
  [candidate name]

Unicode hyphen / dash / quote variants get transliterated to their ASCII
equivalents before rendering, because reportlab's built-in Helvetica AFM
metrics don't cover the U+2010 hyphen / U+2013 en-dash family and renders
them as unmapped glyphs (often visible as "I" or boxes on re-extraction).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional


# Same fallbacks as the PDF writer — Helvetica AFM is Latin-1 only.
_GLYPH_FALLBACKS = {
    "•": "·",
    "●": "·",
    "▪": "·",
    "◦": "·",
    "■": "·",
    "□": "·",
    "▶": ">",
    "►": ">",
    "–": "-",
    "—": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",
    "​": "",
}


def _sanitize(text: str) -> str:
    if not text:
        return text
    out = []
    for ch in text:
        if ch in _GLYPH_FALLBACKS:
            out.append(_GLYPH_FALLBACKS[ch])
        elif ord(ch) > 0xFF:
            out.append("")
        else:
            out.append(ch)
    return "".join(out)


def render_cover_letter_pdf(
    text: str,
    out_path: str | Path,
    candidate_name: Optional[str] = None,
    contact: Optional[Dict[str, str]] = None,
) -> Path:
    """Write the cover letter to out_path as a PDF and return the path.

    Args:
        text: the LLM-generated cover letter body. May include the salutation
            and sign-off; we'll dedupe if the renderer also writes them.
        candidate_name: shown in the header. Optional.
        contact: dict with optional keys 'email', 'phone', 'linkedin', 'github',
            'location'. Rendered as a single line under the name.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    name_style = ParagraphStyle(
        name="Name",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        name="Contact",
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor="#444444",
        spaceAfter=12,
    )
    date_style = ParagraphStyle(
        name="Date",
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=11,
        leading=15.5,
        spaceAfter=11,
    )

    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="Cover Letter",
        author=candidate_name or "",
    )

    story = []

    if candidate_name:
        story.append(Paragraph(_html_escape(_sanitize(candidate_name)), name_style))

    contact_markup = _format_contact_line(contact)
    if contact_markup:
        # Already-safe HTML with <a> hyperlinks; do NOT re-escape.
        story.append(Paragraph(contact_markup, contact_style))
    else:
        story.append(Spacer(1, 4))

    story.append(Paragraph(date.today().strftime("%B %d, %Y"), date_style))

    for block in _split_paragraphs(text):
        sanitized = _sanitize(block)
        para = _html_escape(sanitized).replace("\n", "<br/>")
        story.append(Paragraph(para, body_style))

    doc.build(story)
    return out


_LINK_COLOR = "#2A66C8"


def _format_contact_line(contact: Optional[Dict[str, str]]) -> str:
    """Return reportlab markup for the contact line.

    LinkedIn and GitHub values may arrive as bare usernames (because the PDF
    parser strips icon glyphs and the LLM extracts just the handle). We
    normalize to proper URLs so the rendered cover letter shows a real
    clickable link (linkedin.com/in/USER, github.com/USER, mailto:EMAIL)
    instead of an orphan handle.
    """
    if not contact:
        return ""
    order = ("email", "phone", "linkedin", "github", "location")
    parts = []
    for key in order:
        val = (contact.get(key) or "").strip()
        if not val:
            continue
        parts.append(_render_contact_item(key, val))
    return "  |  ".join(parts)


def _render_contact_item(key: str, value: str) -> str:
    sanitized = _sanitize(value)
    if key == "email":
        href = f"mailto:{sanitized}"
        return _link(href, sanitized)
    if key == "linkedin":
        url = _ensure_linkedin_url(sanitized)
        display = url.replace("https://", "").replace("http://", "")
        return _link(url, display)
    if key == "github":
        url = _ensure_github_url(sanitized)
        display = url.replace("https://", "").replace("http://", "")
        return _link(url, display)
    return _html_escape(sanitized)


def _link(href: str, display: str) -> str:
    return (
        f'<a href="{_html_escape(href)}" color="{_LINK_COLOR}">'
        f'<u>{_html_escape(display)}</u>'
        f'</a>'
    )


def _ensure_linkedin_url(value: str) -> str:
    v = value.strip().lstrip("@")
    if v.lower().startswith(("http://", "https://")):
        return v
    if "linkedin.com" in v.lower():
        return "https://" + v.lstrip("/")
    return f"https://linkedin.com/in/{v.lstrip('/')}"


def _ensure_github_url(value: str) -> str:
    v = value.strip().lstrip("@")
    if v.lower().startswith(("http://", "https://")):
        return v
    if "github.com" in v.lower():
        return "https://" + v.lstrip("/")
    return f"https://github.com/{v.lstrip('/')}"


def _split_paragraphs(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if not line:
            if buf:
                parts.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        parts.append("\n".join(buf))
    return parts


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
