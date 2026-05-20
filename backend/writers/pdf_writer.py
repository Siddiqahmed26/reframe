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
    # Latin ligatures (U+FB00..U+FB06): expand so ATS keyword extraction sees
    # real letters. Without this, "fine-tuning" renders as the FB01 glyph and
    # extracts as the ligature char, missing the keyword.
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}


def _normalize_sentence_endings(text: str) -> str:
    """Replace a trailing semicolon with a period.

    The rewriter sometimes ends a bullet with ';' because the source clause
    was joined in a longer compound sentence and the LLM stopped mid-list.
    A trailing ';' reads as 'unfinished'. Internal semicolons are kept
    (often valid in long-form bullets).
    """
    if not text:
        return text
    stripped = text.rstrip()
    if stripped.endswith(";"):
        return stripped[:-1].rstrip() + "."
    return text


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


def _tokenize_runs(text: str) -> List[Tuple[str, bool]]:
    """Split text into alternating (chunk, is_bold) runs based on metric
    patterns. Bold runs are quantitative tokens like '5+', '~70%', '10K+',
    '2-week'. Everything else is regular."""
    runs: List[Tuple[str, bool]] = []
    last = 0
    for m in _METRIC_RE.finditer(text):
        if m.start() > last:
            runs.append((text[last:m.start()], False))
        runs.append((m.group(), True))
        last = m.end()
    if last < len(text):
        runs.append((text[last:], False))
    return runs


_ATTACHING_PUNCT = ",.;:!?)]}"


def _layout_lines(
    text: str,
    fontname_reg: str,
    fontname_bold: str,
    size: float,
    max_width: float,
) -> List[List[Tuple[str, bool, float]]]:
    """Wrap text into lines manually. Returns a list of lines, where each
    line is a list of (word, is_bold, width_in_points). Word widths use the
    appropriate font (regular or bold) so wrap accounts for the wider bold
    glyphs. Words that start with attaching punctuation (",;:.!?)]}") render
    flush against the previous word (no leading space)."""
    import pymupdf

    runs = _tokenize_runs(text)
    words: List[Tuple[str, bool, float]] = []
    for chunk, is_bold in runs:
        font = fontname_bold if is_bold else fontname_reg
        for part in chunk.split(" "):
            if not part:
                continue
            w = pymupdf.get_text_length(part, fontname=font, fontsize=size)
            words.append((part, is_bold, w))

    space_w = pymupdf.get_text_length(" ", fontname=fontname_reg, fontsize=size)
    lines: List[List[Tuple[str, bool, float]]] = []
    cur: List[Tuple[str, bool, float]] = []
    cur_w = 0.0
    for word, is_bold, w in words:
        wants_space = bool(cur) and word[:1] not in _ATTACHING_PUNCT
        added = (space_w if wants_space else 0) + w
        if cur and cur_w + added > max_width:
            lines.append(cur)
            cur = [(word, is_bold, w)]
            cur_w = w
        else:
            cur.append((word, is_bold, w))
            cur_w += added
    if cur:
        lines.append(cur)
    return lines


def _render_lines(
    page,
    lines: List[List[Tuple[str, bool, float]]],
    x0: float,
    baseline_y: float,
    line_height: float,
    size: float,
    color: Tuple[float, float, float],
    fontname_reg: str,
    fontname_bold: str,
    space_w: float,
) -> None:
    """Place each word at the computed (x, baseline_y) via insert_text.
    Regular and bold runs use the same base-14 family so visual weight is
    consistent and Latin-1 glyphs render cleanly (no ligature substitution).
    Attaching punctuation gets no leading space — keeps `~70%,` rendered
    tight, not as `~70% ,`."""
    for line_idx, line in enumerate(lines):
        y = baseline_y + line_idx * line_height
        x = x0
        for i, (word, is_bold, w) in enumerate(line):
            font = fontname_bold if is_bold else fontname_reg
            page.insert_text(
                (x, y), word,
                fontname=font, fontsize=size, color=color,
            )
            x += w
            if i < len(line) - 1:
                next_word = line[i + 1][0]
                if next_word[:1] not in _ATTACHING_PUNCT:
                    x += space_w


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
        new_text = _normalize_sentence_endings(new_text)
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
    """Render new_text into the block's bbox via manual per-word layout.

    Strategy: tokenise text into regular/bold runs (bold = quantitative
    tokens like "5+", "~70%", "10K+"), wrap into lines using
    `pymupdf.get_text_length`, then place each word individually with
    `page.insert_text`. This keeps the rewritten text in the same base-14
    family (helv/hebo) as the rest of the document — consistent visual
    weight, no NimbusSans/Helvetica mixing, and no font-driven ligature
    substitution (helv doesn't have fi/fl ligatures, so "fine-tuning"
    extracts cleanly for ATS keyword matching).

    If the text needs more vertical lines than the original block had, we
    truncate from the tail at word boundaries rather than shrinking the
    font — so all bullets stay at the same visual size.
    """
    import pymupdf

    template = blk.spans[0]
    base_size = template.size
    color = _color_tuple(template.color)
    family = _font_family(template.font)
    fontname_reg = _BASE_FONT[(family, "regular")]
    fontname_bold = _BASE_FONT[(family, "bold")]

    x0, y0, x1, y1 = _block_bbox(blk)
    # Available width: the original block plus a tiny right-edge slack so a
    # similarly-sized rewrite doesn't get pushed onto an extra line by a
    # single-pixel measurement difference.
    max_width = (x1 - x0) + 4

    line_height = base_size * 1.18
    # First line baseline sits at roughly cap-top + 0.8 * size. Subsequent
    # lines step down by line_height.
    baseline_y = y0 + base_size * 0.82

    # Vertical room: the original block height plus a small descender margin.
    available_height = (y1 - y0) + base_size * 0.5
    max_lines = max(1, int(available_height / line_height) + 1)

    lines = _layout_lines(new_text, fontname_reg, fontname_bold, base_size, max_width)

    # Truncate by trailing words if we'd overflow the original block height.
    if len(lines) > max_lines:
        words = new_text.split()
        while len(words) > 6:
            words.pop()
            candidate = " ".join(words).rstrip(" ,;:.") + "."
            lines = _layout_lines(candidate, fontname_reg, fontname_bold, base_size, max_width)
            if len(lines) <= max_lines:
                break
        lines = lines[:max_lines]

    space_w = pymupdf.get_text_length(" ", fontname=fontname_reg, fontsize=base_size)
    _render_lines(
        page, lines,
        x0=x0,
        baseline_y=baseline_y,
        line_height=line_height,
        size=base_size,
        color=color,
        fontname_reg=fontname_reg,
        fontname_bold=fontname_bold,
        space_w=space_w,
    )
