"""Best-effort original-format annotations for DOCX and text-based PDF files."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re


def _highlighted_phrases(ai_result):
    if not ai_result:
        return []
    return [item["text"].strip() for item in ai_result.get("segments", [])
            if item.get("ai") and item.get("text", "").strip()]


def _paragraphs_in_container(container):
    """Yield paragraphs from document, tables, and nested tables."""
    for paragraph in getattr(container, "paragraphs", []):
        yield paragraph
    for table in getattr(container, "tables", []):
        for row in table.rows:
            for cell in row.cells:
                yield from _paragraphs_in_container(cell)


def annotate_docx(original_bytes: bytes, ai_result) -> tuple[bytes, int]:
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX

    doc = Document(BytesIO(original_bytes))
    phrases = _highlighted_phrases(ai_result)
    count = 0

    containers = [doc]
    for section in doc.sections:
        containers.extend([section.header, section.footer])

    seen_paragraphs = set()
    for container in containers:
        for paragraph in _paragraphs_in_container(container):
            marker = id(paragraph._p)
            if marker in seen_paragraphs:
                continue
            seen_paragraphs.add(marker)

            full_text = paragraph.text
            if not full_text:
                continue

            ranges = []
            lowered = full_text.casefold()
            for phrase in phrases:
                start = 0
                target = phrase.casefold()
                while target and (pos := lowered.find(target, start)) >= 0:
                    ranges.append((pos, pos + len(phrase)))
                    start = pos + max(1, len(target))

            if not ranges:
                continue

            offset = 0
            for run in paragraph.runs:
                run_start, run_end = offset, offset + len(run.text)
                if any(a < run_end and b > run_start for a, b in ranges):
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                offset = run_end
            count += len(ranges)

    output = BytesIO()
    doc.save(output)
    return output.getvalue(), count


def annotate_pdf(original_bytes: bytes, ai_result) -> tuple[bytes, int]:
    import fitz

    phrases = _highlighted_phrases(ai_result)
    pdf = fitz.open(stream=original_bytes, filetype="pdf")
    count = 0
    try:
        for page in pdf:
            for phrase in phrases:
                # PyMuPDF can find exact phrase matches in text-based PDFs.
                # Line-wrapped or OCR-only text may not map exactly.
                try:
                    rects = page.search_for(phrase, quads=False)
                except Exception:
                    rects = []
                for rect in rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_info(content="TextTrace: segment classified as likely AI-written; review required.")
                    annot.update()
                    count += 1
        output = pdf.tobytes(garbage=4, deflate=True)
        return output, count
    finally:
        pdf.close()


def create_annotated_copy(filename: str, original_bytes: bytes, ai_result):
    """Return (annotated_bytes, filename, annotation_count, warning)."""
    ext = Path(filename).suffix.lower()
    if not ai_result:
        return original_bytes, filename, 0, "AI-writing analysis was not run."
    if ext == ".docx":
        data, count = annotate_docx(original_bytes, ai_result)
        warning = None if count else (
            "No exact AI-likely phrases could be mapped to DOCX runs. "
            "The file structure was preserved, but no highlights were added."
        )
        return data, f"{Path(filename).stem}_TextTrace_highlighted.docx", count, warning
    if ext == ".pdf":
        data, count = annotate_pdf(original_bytes, ai_result)
        warning = None if count else (
            "No exact phrases could be mapped to PDF page coordinates. "
            "Scanned or line-wrapped text may require OCR/coordinate mapping."
        )
        return data, f"{Path(filename).stem}_TextTrace_highlighted.pdf", count, warning
    return original_bytes, filename, 0, (
        "Original-format highlighting is supported for DOCX and text-based PDF. "
        "For this file type, use the in-app highlighted review."
    )
