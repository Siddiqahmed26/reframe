"""Format-preserving DOCX writer.

Strategy: we never create new paragraphs and never touch paragraph-level
properties. For each paragraph that has rewritten text, we:

  1. Take the new text string from the rewrite instructions.
  2. Place it inside the FIRST run of the paragraph, which carries the
     run-level formatting (font, size, bold/italic/underline, color).
  3. Clear the text of every other run in that paragraph, keeping their
     run/style elements intact so the XML structure does not change shape.

This means paragraph alignment, indentation, spacing, bullet markers, section
breaks, and page margins are entirely untouched. The visible difference
between the original and the rewritten document is the *text content* of
bullets and summary lines only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from docx.document import Document as DocxDocument
from docx.text.paragraph import Paragraph

from backend.models import RewriteInstruction
from backend.parsers.docx_parser import ParsedDocx


def _iter_all_paragraphs(doc: DocxDocument):
    """Yield every paragraph in body and in tables, in the same order the
    parser visited them. Returns tuples of (composite_index, paragraph)."""
    body_count = len(doc.paragraphs)
    for idx, p in enumerate(doc.paragraphs):
        yield idx, p
    for t_idx, table in enumerate(doc.tables):
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                for p_idx, p in enumerate(cell.paragraphs):
                    composite_idx = body_count + 1000 * t_idx + 100 * r_idx + 10 * c_idx + p_idx
                    yield composite_idx, p


def _replace_text_preserving_format(paragraph: Paragraph, new_text: str) -> None:
    """Put new_text into the paragraph while preserving every formatting
    attribute of the original first run."""
    runs = paragraph.runs
    if not runs:
        # An empty paragraph (rare in resumes for content lines). Just add a run.
        paragraph.add_run(new_text)
        return

    # First run gets the new text
    first = runs[0]
    first.text = new_text

    # Subsequent runs are emptied but their elements remain so style references
    # downstream (e.g. an italic run that styled a job title) don't disappear.
    for run in runs[1:]:
        run.text = ""


def apply_rewrites(parsed: ParsedDocx, instructions: List[RewriteInstruction], out_path: str | Path) -> Path:
    """Apply all rewrite instructions to the original parsed document and save."""
    # Index by paragraph index for O(1) lookup
    by_idx: Dict[int, RewriteInstruction] = {ins.paragraph_index: ins for ins in instructions}

    if not by_idx:
        # Nothing to rewrite; still save a copy so the API can return the file
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        parsed.document.save(str(out))
        return out

    for composite_idx, p in _iter_all_paragraphs(parsed.document):
        ins = by_idx.get(composite_idx)
        if ins is None:
            continue
        if not ins.rewritten or ins.rewritten == ins.original:
            continue
        _replace_text_preserving_format(p, ins.rewritten)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    parsed.document.save(str(out))
    return out
