"""Editorial cover-letter PDF renderer (reportlab Platypus).

Visually mirrors the polished .docx style in backend/agents/cover_letter.py:
  - Helvetica-Bold name header, navy, slightly tracked.
  - Helvetica contact strip, slate, middle-dot separators.
  - Thin emerald-cyan accent rule under the header.
  - Times-Roman body at 11pt / 15pt leading (~1.36 line spacing).
  - Greeting paragraph, body paragraphs with first-line indent on
    continuations, then "Sincerely," + signature gap + typed name in
    Helvetica-Bold for visual contrast.

We deliberately stick to reportlab's built-in base 14 fonts (Helvetica,
Times-Roman) so the renderer has no system-font dependency and ships clean
on Hugging Face Spaces. Calibri and Georgia are not registered.
"""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from datetime import date
from typing import Optional
from xml.sax.saxutils import escape as xml_escape


logger = logging.getLogger(__name__)


# Style palette — keep in sync with backend/agents/cover_letter.py
_NAVY = "#1A2339"
_SLATE = "#6E7689"
_LINK = "#475C7A"   # hyperlinks: slate-blue, restrained
_BODY_INK = "#262C3A"
_BRAND_ACCENT = "#5BE8C5"

# Type scale matched to the docx renderer (the user reported earlier output
# felt cramped; 12pt body at 17pt leading fills a typical 280-340 word
# cover letter cleanly without looking dense).
_NAME_SIZE = 24
_CONTACT_SIZE = 10
_DATE_SIZE = 11.5
_BODY_SIZE = 12
_BODY_LEADING = 17
_TYPED_NAME_SIZE = 12.5


# Same fallbacks as the docx renderer — base 14 PDF fonts (Helvetica,
# Times-Roman) cover Latin-1 only. Substitute typographic glyphs the LLM
# might still emit despite the system prompt.
_TEXT_FALLBACKS = {
    "—": ", ",   # em dash → comma + space (we forbid em dashes)
    "–": "-",    # en dash → hyphen
    "‐": "-", "‑": "-", "‒": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...",
}


def _sanitize_text(text: str) -> str:
    """Strip control chars and substitute typographic glyphs we don't want
    in a print document. Identical semantics to the docx-side helper."""
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        if ch in _TEXT_FALLBACKS:
            out.append(_TEXT_FALLBACKS[ch])
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\t"):
            continue
        out.append(ch)
    return "".join(out)


def _sanitize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", " ", _sanitize_text(name).strip())


def _strip_handle(value: str, *, host_fragment: str, handle_prefix: str) -> str:
    """Normalize a LinkedIn/GitHub field into '<prefix>/<handle>'."""
    v = (value or "").strip().lstrip("@").lstrip("/")
    if not v:
        return ""
    v = re.sub(r"^https?://(www\.)?", "", v, flags=re.I)
    low = v.lower()
    idx = low.find(host_fragment)
    if idx >= 0:
        v = v[idx + len(host_fragment):].lstrip("/")
    if v.lower().startswith("in/"):
        v = v[3:]
    return f"{handle_prefix}/{v}" if v else ""


def _full_url(value: str, *, host: str) -> str:
    """Return a canonical https URL for a LinkedIn/GitHub field."""
    v = (value or "").strip().lstrip("@").lstrip("/")
    if not v:
        return ""
    if v.lower().startswith(("http://", "https://")):
        return v
    if host.lower() in v.lower():
        return "https://" + v
    if host == "linkedin.com":
        path = v[3:] if v.lower().startswith("in/") else v
        return f"https://linkedin.com/in/{path.lstrip('/')}"
    return f"https://{host}/{v}"


def _format_contact_items(contact: Optional[dict]) -> list:
    """Return ordered (display_text, optional_url) tuples for the contact strip."""
    if not contact:
        return []
    items = []

    email = (contact.get("email") or "").strip()
    if email:
        items.append((_sanitize_text(email), f"mailto:{email}"))

    phone = (contact.get("phone") or "").strip()
    if phone:
        items.append((_sanitize_text(phone), None))

    location = (contact.get("location") or "").strip()
    if location:
        items.append((_sanitize_text(location), None))

    linkedin_display = _strip_handle(
        contact.get("linkedin") or "",
        host_fragment="linkedin.com",
        handle_prefix="in",
    )
    if linkedin_display:
        url = _full_url(contact.get("linkedin") or "", host="linkedin.com")
        items.append((_sanitize_text(linkedin_display), url or None))

    github_display = _strip_handle(
        contact.get("github") or "",
        host_fragment="github.com",
        handle_prefix="gh",
    )
    if github_display:
        url = _full_url(contact.get("github") or "", host="github.com")
        items.append((_sanitize_text(github_display), url or None))

    return items


def _format_contact_markup(contact: Optional[dict]) -> str:
    """Return reportlab Paragraph-compatible markup for the contact strip.
    Items with a URL render as <link> tags (underlined, slate-blue);
    items without URL render as plain text. Middle-dot separator."""
    items = _format_contact_items(contact)
    if not items:
        return ""
    rendered = []
    for text, url in items:
        safe = xml_escape(text)
        if url:
            rendered.append(
                f'<link href="{xml_escape(url)}" color="{_LINK}"><u>{safe}</u></link>'
            )
        else:
            rendered.append(safe)
    return " · ".join(rendered)


def _format_contact(contact: Optional[dict]) -> str:
    """Build the contact line. Middle-dot (·) separated, surrounded by
    thin spaces ( ) for a touch more breathing room on the page."""
    if not contact:
        return ""
    parts: list[str] = []

    email = (contact.get("email") or "").strip()
    if email:
        parts.append(_sanitize_text(email))

    phone = (contact.get("phone") or "").strip()
    if phone:
        parts.append(_sanitize_text(phone))

    location = (contact.get("location") or "").strip()
    if location:
        parts.append(_sanitize_text(location))

    linkedin = _strip_handle(
        contact.get("linkedin") or "",
        host_fragment="linkedin.com",
        handle_prefix="in",
    )
    if linkedin:
        parts.append(_sanitize_text(linkedin))

    github = _strip_handle(
        contact.get("github") or "",
        host_fragment="github.com",
        handle_prefix="gh",
    )
    if github:
        parts.append(_sanitize_text(github))

    # Thin-space + middle-dot + thin-space
    sep = " · "
    return sep.join(parts)


_GREETING_RE = re.compile(r"^\s*(dear\s+|hello\s+|hi\s+|to\s+whom)", re.IGNORECASE)
_SIGNOFF_RE = re.compile(
    r"^\s*(sincerely|regards|best|yours|thank\s+you)[,\s]",
    re.IGNORECASE,
)


def _split_body_paragraphs(text: str) -> list[str]:
    """Mirror the docx-side split: prefer blank lines, fall back to single
    newlines if the LLM produced no blank-line breaks."""
    cleaned = _sanitize_text(text or "").strip()
    if not cleaned:
        return []
    if "\n\n" in cleaned:
        chunks = [c.strip() for c in cleaned.split("\n\n")]
        return [c for c in chunks if c]
    return [ln.strip() for ln in cleaned.splitlines() if ln.strip()]


def _strip_trailing_name(paragraphs: list[str], name_clean: str) -> list[str]:
    """If the LLM tucked the typed name onto the last paragraph as a trailing
    `\n{name}`, peel it off so we don't print it twice (the typed-name slot
    at the bottom of the letter renders it in Helvetica-Bold)."""
    if not paragraphs or not name_clean:
        return paragraphs
    last = paragraphs[-1]
    trail_re = re.compile(
        r"\n\s*" + re.escape(name_clean) + r"\s*$",
        re.IGNORECASE,
    )
    stripped = trail_re.sub("", last).rstrip()
    if stripped != last:
        paragraphs = paragraphs[:-1] + [stripped]
    return paragraphs


def _escape_for_paragraph(text: str) -> str:
    """reportlab Paragraph parses pseudo-XML markup; escape user content
    before handing it in so '<' or '&' in the body don't blow up parsing."""
    return xml_escape(text or "", entities={"\n": "<br/>"})


def build_cover_letter_pdf(
    text: str,
    candidate_name: Optional[str] = None,
    contact: Optional[dict] = None,
    date_str: Optional[str] = None,
    *,
    candidate_headline: Optional[str] = None,
    page_size: str = "letter",
) -> bytes:
    """Render a styled, single-page cover-letter PDF and return the bytes.

    Args:
        text: the LLM-generated body, with embedded greeting and (optional)
            sign-off lines. Whitespace-tolerant; blank-line paragraph
            separators are preferred but not required.
        candidate_name: rendered in the header. If empty the header is
            skipped entirely (we do NOT emit "None" or "Your Name").
        contact: dict of optional keys email, phone, linkedin, github,
            location. Missing keys are dropped from the contact strip
            cleanly.
        date_str: override the date label (e.g. "May 18, 2026"). Defaults
            to today's date in long form.
        page_size: "letter" (default) or "a4".
    """
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import LETTER, A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    name_clean = _sanitize_name(candidate_name)
    body_paragraphs = _split_body_paragraphs(text)
    if len(body_paragraphs) < 2:
        logger.warning(
            "Cover letter pdf: only %d paragraph(s) detected after split; "
            "raw text len=%d.",
            len(body_paragraphs), len(text or ""),
        )
    body_paragraphs = _strip_trailing_name(body_paragraphs, name_clean)

    # Pull greeting and sign-off out of the body so we can style each
    # block independently. Greeting stays atomic. Sign-off lines route to
    # the bottom block (rendered with a 30pt signature gap).
    greeting: Optional[str] = None
    signoff: Optional[str] = None
    if body_paragraphs and _GREETING_RE.match(body_paragraphs[0]):
        greeting = body_paragraphs[0]
        body_paragraphs = body_paragraphs[1:]
    if body_paragraphs:
        # Look at the last 2 paragraphs for a short sign-off line.
        for tail_idx in (-1, -2):
            try:
                cand = body_paragraphs[tail_idx]
            except IndexError:
                continue
            if _SIGNOFF_RE.match(cand) and len(cand) < 40:
                signoff = cand
                body_paragraphs.pop(tail_idx)
                break
    if signoff is None:
        signoff = "Sincerely,"

    page = LETTER if (page_size or "letter").lower() != "a4" else A4

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        title="Cover Letter",
        author=name_clean or "",
    )

    # Styles
    # Note: we used to insert U+2009 (thin space) between every letter of
    # the name to approximate letter-spacing. The visual result was
    # dispersion ("S I D D I Q  A H M E D") not subtle tracking. Removed —
    # we now trust Helvetica-Bold at 24pt to anchor the page.
    name_style = ParagraphStyle(
        name="Name",
        fontName="Helvetica-Bold",
        fontSize=_NAME_SIZE,
        leading=_NAME_SIZE + 4,
        textColor=HexColor(_NAVY),
        alignment=TA_LEFT,
        spaceAfter=2,
    )
    headline_style = ParagraphStyle(
        name="Headline",
        fontName="Helvetica",
        fontSize=10.5,
        leading=13,
        textColor=HexColor(_SLATE),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    contact_style = ParagraphStyle(
        name="Contact",
        fontName="Helvetica",
        fontSize=_CONTACT_SIZE,
        leading=_CONTACT_SIZE + 3,
        textColor=HexColor(_SLATE),
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    date_style = ParagraphStyle(
        name="Date",
        fontName="Times-Roman",
        fontSize=_DATE_SIZE,
        leading=_DATE_SIZE + 4,
        textColor=HexColor(_BODY_INK),
        alignment=TA_LEFT,
        spaceAfter=14,
    )
    greeting_style = ParagraphStyle(
        name="Greeting",
        fontName="Times-Roman",
        fontSize=_BODY_SIZE,
        leading=_BODY_LEADING,
        textColor=HexColor(_BODY_INK),
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    body_first_style = ParagraphStyle(
        name="BodyFirst",
        fontName="Times-Roman",
        fontSize=_BODY_SIZE,
        leading=_BODY_LEADING,
        textColor=HexColor(_BODY_INK),
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        name="Body",
        fontName="Times-Roman",
        fontSize=_BODY_SIZE,
        leading=_BODY_LEADING,
        textColor=HexColor(_BODY_INK),
        alignment=TA_LEFT,
        firstLineIndent=18,  # 0.25 inch
        spaceAfter=10,
    )
    signoff_style = ParagraphStyle(
        name="SignOff",
        fontName="Times-Roman",
        fontSize=_BODY_SIZE,
        leading=_BODY_LEADING,
        textColor=HexColor(_BODY_INK),
        alignment=TA_LEFT,
        spaceBefore=14,
        spaceAfter=0,
    )
    typed_name_style = ParagraphStyle(
        name="TypedName",
        fontName="Helvetica-Bold",
        fontSize=_TYPED_NAME_SIZE,
        leading=_TYPED_NAME_SIZE + 2,
        textColor=HexColor(_NAVY),
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
    )

    story = []

    # 1. Name header.
    if name_clean:
        # 24pt Helvetica-Bold is the visual anchor; manual letter-spacing
        # dispersed the letters instead of subtly tracking them.
        story.append(Paragraph(_escape_for_paragraph(name_clean.upper()), name_style))

    # 1b. Optional headline directly under the name (smaller muted slate),
    #     before the contact strip and accent rule. Sanitizes typographic
    #     glyphs the LLM may have emitted in the headline string.
    headline_clean = _sanitize_text(candidate_headline or "").strip()
    if headline_clean:
        story.append(Paragraph(_escape_for_paragraph(headline_clean), headline_style))

    # 2. Contact strip. Items with a URL render as clickable <link> tags
    #    (underlined slate-blue) so the PDF carries real hyperlinks for
    #    email, LinkedIn, and GitHub.
    contact_markup = _format_contact_markup(contact)
    if contact_markup:
        # _format_contact_markup already escapes plain text and emits valid
        # reportlab Paragraph markup; do NOT pass it through _escape_for_paragraph.
        story.append(Paragraph(contact_markup, contact_style))

    # 3. Accent rule under the header.
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.75,
            color=HexColor(_BRAND_ACCENT),
            spaceBefore=4,
            spaceAfter=18,
            lineCap="butt",
        )
    )

    # 4. Date (override if provided, else long-form today).
    date_label = (date_str or "").strip() or date.today().strftime("%B %d, %Y")
    story.append(Paragraph(_escape_for_paragraph(date_label), date_style))

    # 5. Greeting (if the LLM included one).
    if greeting:
        story.append(Paragraph(_escape_for_paragraph(greeting), greeting_style))

    # 6. Body paragraphs.
    for i, para in enumerate(body_paragraphs):
        # First body paragraph gets no first-line indent; subsequent
        # continuations get 0.25" indent for editorial polish.
        style = body_first_style if i == 0 else body_style
        story.append(Paragraph(_escape_for_paragraph(para), style))

    # 7. Sign-off: "Sincerely," (or LLM-emitted), 30pt signature gap, typed
    #    name in Helvetica-Bold navy.
    story.append(Paragraph(_escape_for_paragraph(signoff), signoff_style))
    story.append(Spacer(1, 30))
    if name_clean:
        story.append(Paragraph(_escape_for_paragraph(name_clean), typed_name_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
