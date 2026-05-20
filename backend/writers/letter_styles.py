"""Low-level python-docx styling helpers for the cover-letter renderer.

Centralizes the OXML poking that python-docx doesn't expose ergonomically:
font color/weight on a Run, paragraph spacing in points, and a paragraph
bottom border (the "accent rule" beneath the header).
"""
from __future__ import annotations

from typing import Tuple

from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor


def set_run_font(
    run,
    *,
    name: str,
    size_pt: float,
    bold: bool = False,
    color_rgb: Tuple[int, int, int] | None = None,
    expanded_twips: int = 0,
) -> None:
    """Apply font name, size, weight, color, and optional letter-spacing
    (expanded by `expanded_twips`, 20 twips = 1 pt) to a Run.

    python-docx does font name + size + bold on `run.font`, but color via
    `run.font.color.rgb`, and letter spacing only via raw OXML. This wraps
    all of that so callers don't have to think about it.
    """
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    # Ensure east-asian font isn't auto-substituted (Word will sometimes pick
    # an East Asian fallback that looks wrong if rPr.eastAsia is unset).
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)

    if color_rgb is not None:
        run.font.color.rgb = RGBColor(*color_rgb)

    if expanded_twips:
        # w:spacing inside rPr controls character spacing in twips. Positive
        # = expanded, negative = condensed.
        existing = rPr.find(qn("w:spacing"))
        if existing is None:
            existing = OxmlElement("w:spacing")
            rPr.append(existing)
        existing.set(qn("w:val"), str(expanded_twips))


def set_paragraph_spacing(
    paragraph,
    *,
    before_pt: float | None = None,
    after_pt: float | None = None,
    line: float | None = None,
    first_line_indent_in: float | None = None,
) -> None:
    """Set before/after spacing in points and line spacing (multiplier)."""
    pf = paragraph.paragraph_format
    if before_pt is not None:
        pf.space_before = Pt(before_pt)
    if after_pt is not None:
        pf.space_after = Pt(after_pt)
    if line is not None:
        pf.line_spacing = line
    if first_line_indent_in is not None:
        from docx.shared import Inches
        pf.first_line_indent = Inches(first_line_indent_in)


def add_hyperlink_run(
    paragraph,
    url: str,
    text: str,
    *,
    name: str,
    size_pt: float,
    color_rgb: Tuple[int, int, int],
    underline: bool = True,
) -> None:
    """Append a hyperlinked run to `paragraph`.

    python-docx has no native hyperlink API, so we register an external
    relationship on the parent part and emit raw OXML: a `<w:hyperlink>`
    element wrapping a `<w:r>` styled with the given font/size/color (and
    optionally underlined). The hyperlink renders blue-and-underlined in
    Word when opened, and opens the URL on click.
    """
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:cs"), name)
    rPr.append(rFonts)

    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(round(size_pt * 2))))  # half-points
    rPr.append(sz)

    rR, rG, rB = color_rgb
    color = OxmlElement("w:color")
    color.set(qn("w:val"), f"{rR:02X}{rG:02X}{rB:02X}")
    rPr.append(color)

    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rPr.append(u)

    r.append(rPr)

    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)

    hyperlink.append(r)
    paragraph._element.append(hyperlink)


def add_horizontal_rule(
    paragraph,
    *,
    color_rgb: Tuple[int, int, int],
    width_eighth_pt: int = 6,
    space_pt: int = 4,
) -> None:
    """Add a bottom border to `paragraph`. The border renders as a thin
    horizontal rule beneath the paragraph's content (typically used on an
    empty paragraph to create a divider line).

    width_eighth_pt is in 1/8 of a point per OOXML — 6 = 0.75 pt, 8 = 1 pt.
    space_pt is points of whitespace between the text and the rule.
    """
    pPr = paragraph._element.get_or_add_pPr()
    # Reuse existing pBdr if present, else create one.
    pBdr = pPr.find(qn("w:pBdr"))
    if pBdr is None:
        pBdr = OxmlElement("w:pBdr")
        pPr.append(pBdr)
    bottom = pBdr.find(qn("w:bottom"))
    if bottom is None:
        bottom = OxmlElement("w:bottom")
        pBdr.append(bottom)
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(width_eighth_pt))
    bottom.set(qn("w:space"), str(space_pt))
    r, g, b = color_rgb
    bottom.set(qn("w:color"), f"{r:02X}{g:02X}{b:02X}")
