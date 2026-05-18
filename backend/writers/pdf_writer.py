"""In-place PDF rewriter.

For each rewrite instruction we:
  1. Find the original PdfBlock by its packed (page, block_idx) id.
  2. Redact every span in that block so the original glyphs disappear from
     the page (links and images outside the block are untouched).
  3. Insert the new text into the same bounding box, using the first span's
     font/size/color as the template.

If the new text doesn't fit at the original font size we shrink in 5% steps
(down to 70%) before giving up. Combined with the rewriter's per-paragraph
character budget this keeps overflow rare in practice.

Fonts: PyMuPDF can use a font's *name* if it maps to one of the 14 base PDF
fonts. We map by inspecting the original font name for bold/italic/serif
hints and pick the closest base font. For embedded custom fonts this is a
visible compromise, but base-font fallback is robust across machines.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from backend.models import RewriteInstruction
from backend.parsers.pdf_parser import ParsedPdf, PdfBlock, PdfSpan


# PyMuPDF base 14 font aliases. Two-letter codes are PyMuPDF's shorthand.
_BASE_FONT = {
    ("sans", "regular"): "helv",
    ("sans", "bold"): "hebo",
    ("sans", "italic"): "heit",
    ("sans", "bolditalic"): "hebi",
    ("serif", "regular"): "tiro",
    ("serif", "bold"): "tibo",
    ("serif", "italic"): "tiit",
    ("serif", "bolditalic"): "tibi",
    ("mono", "regular"): "cour",
    ("mono", "bold"): "cobo",
    ("mono", "italic"): "coit",
    ("mono", "bolditalic"): "cobi",
}


def _font_family(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ("mono", "courier", "consol", "menlo", "cour")):
        return "mono"
    if any(k in n for k in ("times", "roman", "serif", "georgia", "garamond", "cambria")):
        return "serif"
    return "sans"


def _font_weight(name: str, flags: int) -> str:
    n = (name or "").lower()
    # PyMuPDF span flags: bit 4 (16) = bold, bit 1 (2) = italic
    is_bold = ("bold" in n) or ("black" in n) or ("heavy" in n) or bool(flags & 16)
    is_italic = ("italic" in n) or ("oblique" in n) or bool(flags & 2)
    if is_bold and is_italic:
        return "bolditalic"
    if is_bold:
        return "bold"
    if is_italic:
        return "italic"
    return "regular"


def _pick_base_font(span: PdfSpan) -> str:
    family = _font_family(span.font)
    weight = _font_weight(span.font, span.flags)
    return _BASE_FONT[(family, weight)]


def _color_tuple(color_int: int) -> Tuple[float, float, float]:
    r = ((color_int >> 16) & 0xFF) / 255.0
    g = ((color_int >> 8) & 0xFF) / 255.0
    b = (color_int & 0xFF) / 255.0
    return (r, g, b)


# Glyph fallbacks: the PDF base-14 fonts (helv/tiro/cour) only cover
# Latin-1. Any character outside that range renders as a missing-glyph box
# (often shown as "?" on extraction). We swap problem chars for the closest
# Latin-1 equivalent that the base font can actually draw.
_GLYPH_FALLBACKS = {
    "•": "·",  # • bullet -> · middle dot
    "●": "·",  # ● black circle
    "▪": "·",  # ▪
    "◦": "·",  # ◦
    "■": "·",  # ■
    "□": "·",  # □
    "▶": ">",       # ▶
    "►": ">",       # ►
    "–": "-",       # – en dash
    "—": "-",       # — em dash
    "‐": "-",       # ‐ hyphen
    "‑": "-",       # ‑ non-breaking hyphen
    "‒": "-",       # ‒ figure dash
    "‘": "'",       # ' left single quote
    "’": "'",       # ' right single quote
    "“": '"',       # " left double quote
    "”": '"',       # " right double quote
    "…": "...",     # … ellipsis
    " ": " ",       # non-breaking space (safer as plain)
    "​": "",        # zero-width space
}


def _sanitize_for_base_font(text: str) -> str:
    """Replace characters that PyMuPDF's base-14 fonts can't render."""
    if not text:
        return text
    out = []
    for ch in text:
        if ch in _GLYPH_FALLBACKS:
            out.append(_GLYPH_FALLBACKS[ch])
        elif ord(ch) > 0xFF:
            # Anything beyond Latin-1 isn't in base-14 fonts; drop to ASCII '?'
            # is worse than dropping the char outright. Drop it.
            out.append("")
        else:
            out.append(ch)
    return "".join(out)


# Patterns that resumes conventionally bold — quantitative achievements.
# We auto-bold these in rewritten bullets so the output matches the
# original's emphasis style without having to round-trip per-span formatting.
_METRIC_PATTERNS = [
    r"\b\d+(?:\.\d+)?\+?\s*(?:years?|yrs?|months?|weeks?|days?|hours?|hrs?|mins?|minutes?|seconds?)\b",
    r"\b\d+\.\d+\+?(?=\s|$|[,.;])",     # 1.5+, 4.2
    r"~?\d+(?:\.\d+)?%",                # 70%, ~60%, 95%
    r"\b\d+[Kk]\+?(?=\s|$|[,.;])",      # 10K, 10K+
    r"\b\d+\+(?=\s|$|[,.;])",           # 5+, 15+, 40+
    r"\b\d{2,}\b(?=\s+(?:bullet|module|workflow|integration|review|customer|user|record|API|endpoint|webhook|task|deployment|project|sprint|feature|defect|file|test|line|day|week|month|year))",
    r"\b\d+-week\b",                    # 2-week
    r"\b\d+-day\b",
]
_METRIC_RE = re.compile("(" + "|".join(_METRIC_PATTERNS) + ")")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _to_metric_bold_html(text: str) -> str:
    """Wrap metric substrings in <b>. Escapes HTML first so the rewrite
    text is safe to inject."""
    if not text:
        return ""
    escaped = _html_escape(text)
    return _METRIC_RE.sub(r"<b>\1</b>", escaped)


def _try_insert_html(page, rect, new_text: str, base_size: float,
                     color: Tuple[float, float, float]) -> bool:
    """Render new_text into rect via insert_htmlbox so metric patterns get
    bolded inline. Returns True if it fit, False if PyMuPDF reported overflow
    (caller should fall back to insert_textbox)."""
    import pymupdf

    html_body = _to_metric_bold_html(new_text)
    if "<b>" not in html_body:
        # No metrics to bold — no point routing through the HTML pipeline;
        # let the textbox path handle it (more reliable font matching).
        return False

    r, g, b = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    css = (
        "* { margin: 0; padding: 0; }\n"
        "body {\n"
        f"  font-family: sans-serif;\n"
        f"  font-size: {base_size:.1f}pt;\n"
        f"  color: rgb({r}, {g}, {b});\n"
        "  line-height: 1.15;\n"
        "}\n"
        "b { font-weight: bold; }\n"
    )
    html = f"<html><body>{html_body}</body></html>"
    try:
        # insert_htmlbox returns (spare_height, scale). spare_height < 0 means
        # the content overflowed the rect. We accept slight downward scaling
        # (down to 0.94) to match the textbox path's behavior.
        spare, scale = page.insert_htmlbox(
            rect, html, css=css, scale_low=0.94, rotate=0,
        )
    except Exception:
        return False
    return spare >= 0




def _block_bbox(block: PdfBlock) -> Tuple[float, float, float, float]:
    """Union of all span bboxes. We pad the right edge slightly so a
    same-length rewrite has a tiny bit of slack."""
    x0 = min(s.bbox[0] for s in block.spans)
    y0 = min(s.bbox[1] for s in block.spans)
    x1 = max(s.bbox[2] for s in block.spans)
    y1 = max(s.bbox[3] for s in block.spans)
    return (x0, y0, x1, y1)


def _is_bulleted(text: str) -> bool:
    if not text:
        return False
    return text.lstrip()[:1] in {"•", "●", "·", "▪", "◦", "■", "□", "▶", "►", "–", "—", "-", "*"}


def _preserve_bullet_prefix(original: str, new_text: str) -> str:
    """If the original line starts with a bullet glyph + space, keep that
    prefix on the rewrite so the bullet marker isn't lost."""
    stripped = original.lstrip()
    if not stripped:
        return new_text
    prefix_char = stripped[:1]
    bullets = {"•", "●", "·", "▪", "◦", "■", "□", "▶", "►", "–", "—", "-", "*"}
    if prefix_char in bullets and not new_text.lstrip().startswith(prefix_char):
        # Keep the leading whitespace too (indentation)
        leading_ws_len = len(original) - len(stripped)
        return original[:leading_ws_len] + prefix_char + " " + new_text.lstrip()
    return new_text


def apply_rewrites(
    parsed: ParsedPdf,
    instructions: List[RewriteInstruction],
    out_path: str | Path,
) -> Path:
    """Apply rewrites in-place on the PyMuPDF document and save."""
    import pymupdf
    from backend.parsers.pdf_parser import _pack_index

    by_idx: Dict[int, RewriteInstruction] = {ins.paragraph_index: ins for ins in instructions}
    blocks_by_idx: Dict[int, PdfBlock] = {
        _pack_index(blk.page_idx, blk.block_idx, blk.sub_idx): blk
        for blk in parsed.blocks
    }

    doc = parsed.document

    # Group edits by page so we can redact + insert in one pass per page.
    edits_by_page: Dict[int, List[Tuple[PdfBlock, str]]] = {}
    for packed_idx, ins in by_idx.items():
        if not ins.rewritten or ins.rewritten == ins.original:
            continue
        blk = blocks_by_idx.get(packed_idx)
        if blk is None:
            continue
        new_text = _preserve_bullet_prefix(ins.original, ins.rewritten.strip())
        new_text = _sanitize_for_base_font(new_text)
        edits_by_page.setdefault(blk.page_idx, []).append((blk, new_text))

    for page_idx, edits in edits_by_page.items():
        page = doc[page_idx]

        # Step 1: redact original glyphs. We redact each span's bbox
        # individually so we don't whiteout neighbouring content that
        # happens to fall inside the block's outer bbox.
        for blk, _ in edits:
            for span in blk.spans:
                rect = pymupdf.Rect(*span.bbox)
                page.add_redact_annot(rect, fill=(1, 1, 1))
        # images=PDF_REDACT_IMAGE_NONE preserves images that touch the redact
        # rectangle but don't actually overlap glyphs.
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

        # Step 2: insert the new text into each block's original bbox.
        for blk, new_text in edits:
            _insert_into_block(page, blk, new_text)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # garbage=4 + deflate=True keeps the file small after redactions.
    doc.save(str(out), garbage=4, deflate=True)
    return out


def _insert_into_block(page, blk: PdfBlock, new_text: str) -> None:
    """Place new_text inside the block's bbox using the first span's style.

    Two-phase strategy:
      Phase A: try insert_htmlbox so resume-convention metrics (numbers,
        percentages, '5+', '~70%' etc.) render in bold — matching how the
        original PDF emphasises quantitative wins.
      Phase B: fall back to insert_textbox with shrink-to-fit if the HTML
        path fails or overflows.

    Why fall back: insert_htmlbox uses PyMuPDF's Story renderer which has
    slightly different layout semantics than insert_textbox. If it can't
    fit the text in the rect, it returns negative spare_height and we
    drop down to the more reliable textbox path.
    """
    import pymupdf

    template = blk.spans[0]
    base_size = template.size
    color = _color_tuple(template.color)
    fontname = _pick_base_font(template)

    x0, y0, x1, y1 = _block_bbox(blk)
    # Expand the box vertically a bit; the original block bbox is sometimes
    # measured tight to the glyph caps and a tiny font size mismatch can push
    # us into the next line.
    rect = pymupdf.Rect(x0, y0 - 1, x1 + 1, y1 + max(2, base_size * 0.3))

    # Phase A: HTML with auto-bolded metrics.
    if _try_insert_html(page, rect, new_text, base_size, color):
        return

    # Narrow shrink-to-fit range (max 6% reduction) so neighbouring bullets
    # don't end up at visibly different sizes. The rewriter already enforces
    # a per-bullet char budget, so text that lands here should almost always
    # fit at 1.0×; the 0.97/0.94 fallbacks are for occasional overflows.
    for scale in (1.0, 0.97, 0.94):
        size = base_size * scale
        rc = page.insert_textbox(
            rect,
            new_text,
            fontname=fontname,
            fontsize=size,
            color=color,
            align=pymupdf.TEXT_ALIGN_LEFT,
        )
        if rc >= 0:
            return

    # Doesn't fit even at 0.94× — truncate the text rather than shrink further,
    # so all bullets stay at the same visual size. Truncate aggressively in
    # 10% steps from the tail until it fits at the slightly-smaller size.
    truncated = new_text
    while len(truncated) > 20:
        truncated = truncated[: int(len(truncated) * 0.9)].rstrip(" ,;.") + "."
        rc = page.insert_textbox(
            rect,
            truncated,
            fontname=fontname,
            fontsize=base_size * 0.94,
            color=color,
            align=pymupdf.TEXT_ALIGN_LEFT,
        )
        if rc >= 0:
            return

    # Last resort fallback — write whatever fits as plain text at the position.
    page.insert_text(
        (x0, y0 + base_size),
        new_text[:120],
        fontname=fontname,
        fontsize=base_size * 0.94,
        color=color,
    )
