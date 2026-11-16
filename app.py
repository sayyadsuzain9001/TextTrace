from __future__ import annotations

import io
import os
from pathlib import Path
from datetime import datetime
import re
import shutil
import subprocess
import tempfile
import streamlit as st

import fitz  # PyMuPDF: prepend cover page to the annotated PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from texttrace.engine import (
    analyze_document, analyze_similarity, load_uploaded_document,
    annotate_docx, annotate_pdf, make_pdf_report, MODEL_CONFIG_PATH, make_html_review
)


def _cover_wrap(text, font, size, max_width):
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        candidate = word if not current else current + " " + word
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _cover_text(c, text, x, y, max_width, font="Helvetica", size=8,
                color=None, leading=None, max_lines=None):
    color = color or colors.HexColor("#0B1E49")
    lines = _cover_wrap(text, font, size, max_width)
    if max_lines is not None:
        lines = lines[:max_lines]
    c.setFont(font, size)
    c.setFillColor(color)
    step = leading or size * 1.25
    for line in lines:
        c.drawString(x, y, line)
        y -= step
    return y


def _cover_card(c, x, y, width, height, fill, stroke="#DFE8F5"):
    c.setFillColor(colors.HexColor(fill) if isinstance(fill, str) else fill)
    c.setStrokeColor(colors.HexColor(stroke) if isinstance(stroke, str) else stroke)
    c.setLineWidth(0.7)
    c.roundRect(x, y, width, height, 9, fill=1, stroke=1)


def _make_ai_highlight_cover(filename, doc_text, ai_result, similarity_result):
    """Create a one-page branded cover using actual TextTrace analysis results."""
    from io import BytesIO

    page_w, page_h = A4
    margin = 13 * mm
    usable = page_w - 2 * margin
    navy = colors.HexColor("#0B1E49")
    blue = colors.HexColor("#1976F3")
    purple = colors.HexColor("#7547F5")
    pink = colors.HexColor("#F2385A")
    green = colors.HexColor("#08A783")
    mid = colors.HexColor("#536784")
    pale_pink, pale_blue = "#FFF1F4", "#EFF6FF"
    pale_purple, pale_green, pale_gray = "#F5F1FF", "#EEFBF6", "#F4F8FD"

    def pct(value):
        try:
            value = float(value)
            if value < 0:
                return None
            if value <= 1:
                value *= 100
            return min(100.0, value)
        except (TypeError, ValueError):
            return None

    ai_result = ai_result or {}
    similarity_result = similarity_result or {}
    ai_percent = pct(ai_result.get("ai_percent", ai_result.get("percent")))
    sim_percent = pct(similarity_result.get("percent"))
    segments = ai_result.get("segments") or []
    threshold = ai_result.get("threshold")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = None

    word_count = len(re.findall(r"\b[\w'-]+\b", doc_text or ""))
    char_count = len(doc_text or "")
    above = sum(
        1 for s in segments
        if threshold is not None and
        float(s.get("probability", s.get("score", -1))) >= threshold
    )
    below = max(0, len(segments) - above) if threshold is not None else None

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"TextTrace cover - {Path(filename).name}")
    c.setFillColor(colors.white)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
    c.setFillColor(blue)
    c.rect(0, page_h - 4, page_w * .62, 4, fill=1, stroke=0)
    c.setFillColor(purple)
    c.rect(page_w * .62, page_h - 4, page_w * .38, 4, fill=1, stroke=0)

    # Brand and report title
    c.setFont("Helvetica-Bold", 30)
    c.setFillColor(navy)
    c.drawString(margin, page_h - 31 * mm, "Text")
    x_trace = margin + stringWidth("Text", "Helvetica-Bold", 30)
    c.setFillColor(blue)
    c.drawString(x_trace, page_h - 31 * mm, "Trace")
    c.setFillColor(purple)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(x_trace + stringWidth("Trace", "Helvetica-Bold", 30) + 2, page_h - 25 * mm, "✦")
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(margin + 1, page_h - 37 * mm, "A I   A N A L Y S I S   S T U D I O")
    c.setFillColor(blue)
    c.roundRect(margin, page_h - 43 * mm, 31 * mm, 2, 1, fill=1, stroke=0)
    c.setFillColor(purple)
    c.roundRect(margin + 31 * mm, page_h - 43 * mm, 20 * mm, 2, 1, fill=1, stroke=0)
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(margin, page_h - 54 * mm, "AI WRITING REVIEW REPORT")
    _cover_text(c, "A smart review of your document using machine learning and text analysis. "
                   "Highlighted passages are model-selected for review, not proof of authorship.",
                margin, page_h - 61 * mm, usable * .86, size=8, color=mid, leading=10, max_lines=3)

    # Three honest metric cards: this build does not run similarity automatically.
    gap = 8
    card_y, card_h = page_h - 100 * mm, 30 * mm
    card_w = (usable - 2 * gap) / 3
    metrics = [
        ("AI-WRITING ESTIMATE", f"{ai_percent:.1f}%" if ai_percent is not None else "Not run",
         "Model estimate; not proof", pale_pink, pink, "AI"),
        ("TOTAL WORDS", f"{word_count:,}", "Extracted document text", pale_purple, purple, "W"),
        ("SEGMENTS ANALYSED", str(len(segments)) if ai_result else "Not run",
         (f"Above: {above} | Below: {below}" if below is not None else "Threshold unavailable"),
         pale_green, green, "≡"),
    ]
    for i, (title, value, sub, fill, accent, icon) in enumerate(metrics):
        x = margin + i * (card_w + gap)
        _cover_card(c, x, card_y, card_w, card_h, fill)
        c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=.13))
        c.circle(x + 19, card_y + card_h - 21, 11, fill=1, stroke=0)
        c.setFillColor(accent)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(x + 19, card_y + card_h - 24, icon)
        _cover_text(c, title, x + 35, card_y + card_h - 14, card_w - 40,
                    "Helvetica-Bold", 6.5, navy, 7, 2)
        c.setFillColor(accent)
        c.setFont("Helvetica-Bold", 15 if len(value) > 10 else 17)
        c.drawString(x + 10, card_y + 15, value)
        _cover_text(c, sub, x + 10, card_y + 7, card_w - 18, size=5.7, color=mid, leading=6, max_lines=1)

    # Document info + ring
    info_y, info_h = page_h - 151 * mm, 44 * mm
    left_w = usable * .62
    _cover_card(c, margin, info_y, left_w, info_h, pale_gray)
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin + 10, info_y + info_h - 15, "Document Information")
    details = [
        ("File name", Path(filename).name),
        ("File format", Path(filename).suffix.upper().lstrip(".") or "Unknown"),
        ("Word count", f"{word_count:,}"),
        ("Character count", f"{char_count:,}"),
        ("Analysis date", datetime.now().strftime("%d %b %Y, %I:%M %p")),
    ]
    yy = info_y + info_h - 27
    for label, value in details:
        c.setFillColor(mid)
        c.setFont("Helvetica", 6.8)
        c.drawString(margin + 10, yy, label)
        _cover_text(c, value, margin + 77, yy, left_w - 88, "Helvetica-Bold", 6.6, navy, 7, 1)
        yy -= 8.1

    rx, rw = margin + left_w + gap, usable - left_w - gap
    _cover_card(c, rx, info_y, rw, info_h, pale_blue)
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 8.3)
    c.drawString(rx + 8, info_y + info_h - 15, "Overall AI-Writing Result")
    # Keep the ring, percentage, and label on separate baselines. The former
    # layout put the label and the disclaimer on top of each other in exported PDFs.
    cx, cy, radius = rx + rw / 2, info_y + 27, 12
    c.setLineWidth(5)
    c.setStrokeColor(colors.HexColor("#DCE4EF"))
    c.circle(cx, cy, radius, fill=0, stroke=1)
    if ai_percent is None:
        c.setFillColor(mid)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(cx, cy - 2.5, "NOT RUN")
    else:
        c.setStrokeColor(pink)
        c.arc(cx-radius, cy-radius, cx+radius, cy+radius, 90, -3.6 * ai_percent)
        c.setFillColor(pink)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(cx, cy - 3, f"{ai_percent:.1f}%")
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 5.5)
    c.drawCentredString(cx, cy + radius + 4, "AI-LIKELIHOOD")
    _cover_text(c, "Estimate only; not a verified authorship verdict.", rx + 8, info_y + 7,
                rw - 16, size=5.5, color=mid, leading=6, max_lines=1)

    # Full-width model panel. Similarity is intentionally omitted because it is
    # not automatically run during this workflow and must not be shown as a result.
    low_y, low_h = page_h - 198 * mm, 38 * mm
    model_w = usable
    _cover_card(c, margin, low_y, model_w, low_h, pale_gray)
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(margin + 9, low_y + low_h - 14, "AI Model Results")
    models = ai_result.get("models") or []
    model_scores = ai_result.get("model_scores") or {}
    if isinstance(models, dict):
        models = list(models.keys())
    if not models and isinstance(model_scores, dict):
        models = list(model_scores.keys())
    if models:
        n = min(4, len(models))
        mini_w = (model_w - 24 - 8 * (n - 1)) / n
        for i, model in enumerate(models[:4]):
            mx = margin + 12 + i * (mini_w + 8)
            _cover_text(c, str(model), mx, low_y + low_h - 25, mini_w,
                        "Helvetica-Bold", 6.2, blue, 7, 2)
            val = pct(model_scores.get(model)) if isinstance(model_scores, dict) else None
            c.setFillColor(navy)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(mx, low_y + 10, f"{val:.1f}%" if val is not None else "Score unavailable")
            c.setFillColor(colors.HexColor("#DDE5F0"))
            c.roundRect(mx, low_y + 4, mini_w, 3, 1.5, fill=1, stroke=0)
            if val is not None:
                c.setFillColor(blue)
                c.roundRect(mx, low_y + 4, mini_w * val / 100, 3, 1.5, fill=1, stroke=0)
    else:
        _cover_text(c, "Per-model scores were not returned by the analysis engine. The overall estimate above is based on the available analysis output.",
                    margin + 9, low_y + 12, model_w - 18, size=6.5, color=mid, max_lines=2)

    # Notice and footer
    notice_y, notice_h = 24 * mm, 22 * mm
    _cover_card(c, margin, notice_y, usable, notice_h, "#FFF8E9", "#F6D99C")
    c.setFillColor(colors.HexColor("#F5A400"))
    c.circle(margin + 16, notice_y + notice_h - 14, 7, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(margin + 16, notice_y + notice_h - 17, "!")
    c.setFillColor(colors.HexColor("#D88700"))
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(margin + 29, notice_y + notice_h - 11, "Important Notice")
    _cover_text(c, "AI-writing estimates can produce false positives and false negatives. "
                   "Review highlighted passages in context; this estimate is not proof of authorship.",
                margin + 29, notice_y + notice_h - 20, usable - 39, size=6.3, color=navy, leading=7.2, max_lines=3)

    c.setStrokeColor(colors.HexColor("#DFE8F5"))
    c.line(margin, 10 * mm, page_w - margin, 10 * mm)
    c.setFillColor(navy)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(margin, 5.5 * mm, "TextTrace")
    c.setFillColor(mid)
    c.setFont("Helvetica", 6.2)
    c.drawString(margin + 35, 5.5 * mm, "AI Analysis Studio | Research prototype — review required")
    c.drawRightString(page_w - margin, 5.5 * mm, "Page 1")
    c.showPage()
    c.save()
    return buf.getvalue()


def _prepend_cover_to_highlighted_pdf(highlighted_pdf_bytes, filename, doc_text, ai_result, similarity_result):
    """Return a PDF with TextTrace cover first, followed by the AI-highlighted original PDF."""
    cover_bytes = _make_ai_highlight_cover(filename, doc_text, ai_result, similarity_result)
    cover_doc = fitz.open(stream=cover_bytes, filetype="pdf")
    highlighted_doc = fitz.open(stream=highlighted_pdf_bytes, filetype="pdf")
    cover_doc.insert_pdf(highlighted_doc)
    result = cover_doc.tobytes(garbage=4, deflate=True)
    cover_doc.close()
    highlighted_doc.close()
    return result


def _convert_docx_bytes_to_pdf(docx_bytes):
    """Convert an annotated DOCX to PDF with LibreOffice on Windows.

    Uses soffice.com (the console launcher) and a unique temporary profile.
    The document is converted as a DOCX, not rebuilt from extracted text.
    """
    with tempfile.TemporaryDirectory(prefix="texttrace_docx_pdf_") as temp_dir:
        temp = Path(temp_dir).resolve()
        docx_path = temp / "annotated_original.docx"
        pdf_path = temp / "annotated_original.pdf"
        profile_path = temp / "lo_profile"
        profile_path.mkdir(parents=True, exist_ok=True)
        docx_path.write_bytes(docx_bytes)

        # Check PATH first, then common Windows installation locations.
        candidates = [
            shutil.which("soffice.com"),
            shutil.which("soffice"),
            shutil.which("libreoffice"),
            r"C:\Program Files\LibreOffice\program\soffice.com",
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.com",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
        soffice = next(
            (str(Path(candidate)) for candidate in candidates
             if candidate and Path(candidate).is_file()),
            None,
        )

        detail = "LibreOffice was not found in PATH or its standard Windows locations."
        if soffice:
            profile_uri = profile_path.as_uri()
            command = [
                soffice,
                f"-env:UserInstallation={profile_uri}",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(temp),
                str(docx_path),
            ]
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW
                        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
                        else 0
                    ),
                )
                if proc.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
                    return pdf_path.read_bytes()

                detail = (
                    f"LibreOffice exit code: {proc.returncode}. "
                    f"Output: {(proc.stdout or '').strip() or '(none)'}. "
                    f"Error: {(proc.stderr or '').strip() or '(none)'}. "
                    f"PDF created: {pdf_path.exists()}."
                )
            except Exception as exc:
                detail = f"LibreOffice conversion error: {exc}"

        # Optional fallback: requires Microsoft Word installed on Windows.
        try:
            from docx2pdf import convert
            convert(str(docx_path), str(pdf_path))
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                return pdf_path.read_bytes()
            detail += "; docx2pdf did not create a PDF."
        except Exception as exc:
            detail += f"; Word/docx2pdf fallback unavailable ({exc})."

        raise RuntimeError(
            "TextTrace created the highlighted DOCX but could not convert it to PDF. "
            "The combined PDF (cover + highlighted document) was not generated. "
            + detail
        )


def _make_ai_highlighted_pdf(original_bytes, filename, doc_text, ai_result):
    """Create the requested output: cover page first, then annotated original pages."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        annotated_pdf = annotate_pdf(original_bytes, ai_result)
    elif suffix == ".docx":
        annotated_docx = annotate_docx(original_bytes, ai_result)
        annotated_pdf = _convert_docx_bytes_to_pdf(annotated_docx)
    else:
        raise ValueError("A combined AI-highlighted PDF is available for PDF and DOCX uploads only.")
    return _prepend_cover_to_highlighted_pdf(
        annotated_pdf, filename, doc_text, ai_result, None
    )


APP_ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="TextTrace | AI Analysis Studio", page_icon="✦", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 75% 0%, #182443 0%, #0a1020 48%, #080d18 100%); color:#e7edff; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#080d18,#0d1525); border-right:1px solid #202b43; }
[data-testid="stSidebar"] * { color:#d2dcf2; }
.brand {display:flex;align-items:center;gap:12px;padding:8px 4px 18px 4px}
.brand-icon {background:linear-gradient(135deg,#4d7cff,#9357ff);border-radius:12px;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:22px;color:white}
.brand-title {font-size:20px;font-weight:800;color:#f5f7ff;letter-spacing:-.5px}
.brand-sub {font-size:9px;letter-spacing:2px;color:#8295b8;font-weight:800}
.eyebrow {color:#63779d;font-size:10px;letter-spacing:2px;font-weight:800;text-transform:uppercase;margin-bottom:8px}
h1,h2,h3 {letter-spacing:-.7px}
.metric-card {background:linear-gradient(135deg,#1b2742,#141d31);border:1px solid #2a3856;border-radius:16px;padding:18px 20px;min-height:100px}
.metric-label {font-size:13px;color:#aab8d3;margin-bottom:8px}
.metric-value {font-size:32px;font-weight:800;color:#dbe5ff;line-height:1.1;overflow-wrap:anywhere;word-break:break-word}
.notice {background:#151f35;border:1px solid #293a5c;border-left:4px solid #6ba8ff;border-radius:12px;padding:14px 16px;margin:14px 0;color:#c9d8f4}
.warn {background:#251d2d;border:1px solid #493044;border-left:4px solid #ff7b9a;border-radius:12px;padding:14px 16px;margin:14px 0;color:#f2cbd7}
.file-chip {background:#17233a;border:1px solid #2c3d5c;border-radius:12px;padding:14px 16px;overflow-wrap:anywhere;word-break:break-word}
div.stButton > button {border-radius:10px;border:1px solid #2c3d5c;background:#192744;color:#dce6ff;font-weight:650}
div.stButton > button[kind="primary"] {background:linear-gradient(100deg,#557cff,#8757f7);border:0;color:white}
[data-testid="stFileUploader"] section {border-color:#354766;background:#111a2c}
small, .muted {color:#8b9bb8}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "document_result" not in st.session_state:
    st.session_state.document_result = None
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = ""
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "similarity_result" not in st.session_state:
    st.session_state.similarity_result = None

pages = {
    "Dashboard": ("◈", "WORKSPACE"),
    "Analyse document": ("⌕", "WORKSPACE"),
    "AI Writing": ("✳", "ANALYSIS TOOLS"),
    "Similarity": ("↝", "ANALYSIS TOOLS"),
    "Reports": ("▤", "OUTPUTS"),
    "Settings": ("⚙", "PREFERENCES"),
}

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">T</div><div><div class="brand-title">TextTrace</div><div class="brand-sub">AI ANALYSIS STUDIO</div></div></div>', unsafe_allow_html=True)
    last_group = None
    for name, (glyph, group) in pages.items():
        if group != last_group:
            st.markdown(f'<div class="eyebrow" style="margin:14px 0 8px 4px">{group}</div>', unsafe_allow_html=True)
            last_group = group
        if st.button(f"{glyph}  {name}", key=f"nav_{name}", use_container_width=True, type="primary" if st.session_state.page == name else "secondary"):
            st.session_state.page = name
            st.rerun()
    st.markdown("---")
    st.markdown('<div class="eyebrow">SYSTEM STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#6de6bc;font-weight:650">● Workspace ready</div>', unsafe_allow_html=True)
    if st.session_state.uploaded_name:
        st.markdown('<div class="muted" style="margin-top:14px">Latest document</div>', unsafe_allow_html=True)
        # Show full name, allow wrapping instead of truncating it.
        st.markdown(f'<div class="file-chip" style="margin-top:8px">{st.session_state.uploaded_name.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</div>', unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.doc_text.split()):,} words/tokens")

def metric(label, value):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

def require_document():
    if not st.session_state.doc_text:
        st.info("Upload a document in **Analyse document** first.")
        if st.button("Open document upload", type="primary"):
            st.session_state.page = "Analyse document"
            st.rerun()
        return False
    return True

def show_upload():
    st.markdown('<div class="eyebrow">WORKSPACE / 02</div>', unsafe_allow_html=True)
    st.title("Analyse document")
    st.write("Upload a document or source-code file. TextTrace keeps the original upload available and creates a separate highlighted review copy.")
    uploaded = st.file_uploader(
        "Choose a document",
        type=["txt","docx","pdf","py","js","java","cpp","c","h","cs","html","css","md","ts","jsx","sql"],
        key="main_uploader",
        help="Supported: TXT, DOCX, PDF and common source-code files."
    )
    if uploaded:
        raw = uploaded.getvalue()
        # Responsive filename treatment: wrapping in the card avoids truncation for long names.
        st.markdown('<div class="file-chip">', unsafe_allow_html=True)
        st.markdown(f"**Selected file**  \n\n`{uploaded.name}`")
        st.markdown(f"{len(raw):,} bytes · {uploaded.type or 'file'}")
        st.markdown('</div>', unsafe_allow_html=True)
        # Restore the earlier upload summary: filename, word count, character count and preview.
        try:
            preview_text, preview_meta = load_uploaded_document(uploaded.name, raw)
            preview_words = len(preview_text.split())
            st.markdown("#### Document preview")
            st.markdown(
                f'<div class="file-chip"><b>File name</b><br>{uploaded.name.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}<br><br>'
                f'<b>Words</b>: {preview_words:,} &nbsp; · &nbsp; <b>Characters</b>: {len(preview_text):,} &nbsp; · &nbsp; <b>Size</b>: {len(raw):,} bytes</div>',
                unsafe_allow_html=True
            )
            with st.expander("View extracted document text", expanded=False):
                st.text_area("Extracted text preview", value=preview_text[:12000], height=260, disabled=True, label_visibility="collapsed")
                if len(preview_text) > 12000:
                    st.caption(f"Preview shows the first 12,000 characters of {len(preview_text):,}. Full text is used for analysis.")
        except Exception as preview_error:
            st.warning(f"Could not preview this file: {preview_error}")
        if st.button("Analyse this document", type="primary", use_container_width=True):
            with st.spinner("Extracting text and analysing segments..."):
                try:
                    text, meta = load_uploaded_document(uploaded.name, raw)
                    result = analyze_document(text)
                    result["metadata"] = meta
                    st.session_state.uploaded_bytes = raw
                    st.session_state.uploaded_name = uploaded.name
                    st.session_state.doc_text = text
                    st.session_state.document_result = result
                    st.session_state.similarity_result = None
                    st.session_state.page = "AI Writing"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not analyse this file: {e}")
    if st.session_state.uploaded_name:
        st.markdown("#### Current document")
        st.markdown(f'<div class="file-chip">{st.session_state.uploaded_name.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.download_button("Download original file", data=st.session_state.uploaded_bytes, file_name=st.session_state.uploaded_name, mime="application/octet-stream", use_container_width=True)
        with b:
            if st.button("Open AI Writing review", use_container_width=True):
                st.session_state.page = "AI Writing"
                st.rerun()

if st.session_state.page == "Dashboard":
    st.markdown('<div class="eyebrow">TEXTTRACE / ANALYSIS WORKSPACE</div>', unsafe_allow_html=True)
    st.title("AI analysis workspace")
    st.write("Review AI-writing signals, compare text with a local reference library, and export a report.")
    cols = st.columns(3)
    with cols[0]:
        metric("Current document", "Ready" if st.session_state.doc_text else "No file")
    with cols[1]:
        metric("AI analysis", "Complete" if st.session_state.document_result else "Pending")
    with cols[2]:
        metric("Similarity review", "Complete" if st.session_state.similarity_result else "Pending")
    st.markdown("")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Workspace tools")
        if st.button("＋  Analyse a document", type="primary", use_container_width=True):
            st.session_state.page = "Analyse document"; st.rerun()
        if st.button("✳  Review AI writing", use_container_width=True):
            st.session_state.page = "AI Writing"; st.rerun()
        if st.button("↝  Check similarity", use_container_width=True):
            st.session_state.page = "Similarity"; st.rerun()
        if st.button("▤  Open reports", use_container_width=True):
            st.session_state.page = "Reports"; st.rerun()
    with right:
        st.subheader("Recent document")
        if st.session_state.uploaded_name:
            st.markdown(f'<div class="file-chip">{st.session_state.uploaded_name.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</div>', unsafe_allow_html=True)
            st.caption(f"{len(st.session_state.doc_text.split()):,} words/tokens")
        else:
            st.info("Your latest document will appear here after upload.")
        st.markdown('<div class="notice">AI scores are uncertain signals, not proof of authorship. Review the highlighted passages in context.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Analyse document":
    show_upload()

elif st.session_state.page == "AI Writing":
    st.markdown('<div class="eyebrow">ANALYSIS TOOL / 03</div>', unsafe_allow_html=True)
    st.title("AI-writing review")
    st.write("Inspect model scores by segment. Use these as uncertain signals, not authorship verdicts.")
    if require_document():
        result = st.session_state.document_result
        st.markdown(
            f'<div class="file-chip"><b>Document</b><br>{st.session_state.uploaded_name.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}<br>'
            f'<span style="color:#aab8d3">{len(st.session_state.doc_text.split()):,} words · {len(st.session_state.doc_text):,} characters</span></div>',
            unsafe_allow_html=True
        )
        if not result:
            if st.button("Run AI analysis", type="primary"):
                st.session_state.document_result = analyze_document(st.session_state.doc_text); st.rerun()
            st.stop()
        c1, c2, c3 = st.columns(3)
        with c1: metric("Segments above threshold", f"{result['ai_percent']:.1f}%" if result.get("ai_percent") is not None else "Unavailable")
        with c2: metric("Segments analysed", f"{len(result['segments']):,}")
        with c3: metric("Decision threshold", f"{result['threshold']:.2f}")
        st.caption(f"Models: {', '.join(result['models'])} · Method: {result['method']}")
        st.caption(f"Model status: {result.get('model_status', 'Not available')}")
        if not result.get("trained_model_available", False):
            st.error("No trained model produced a score. Install the configured model artifacts before using this analysis; no demo heuristic score is substituted.")
        st.markdown('<div class="warn"><b>Important:</b> A high model score is not proof that a passage was written by AI. False positives and false negatives are possible.</div>', unsafe_allow_html=True)
        for seg in result["segments"]:
            label = "Above threshold" if seg["probability"] >= result["threshold"] else "Below threshold"
            with st.expander(f"SEGMENT {seg['index']:03d} · {label} · {seg['probability']*100:.1f}%"):
                st.write(seg["text"])
                st.progress(min(1.0, max(0.0, seg["probability"])))
                st.caption("AI-likelihood estimate for this segment; not a verified authorship label.")
        st.markdown("### Download reviewed copy")
        st.write("Only above-threshold AI-likely text spans are highlighted in yellow where they can be matched. Below-threshold text is left unchanged; this is an uncertain model estimate, not proof of authorship. The original file is unchanged.")
        original = st.session_state.uploaded_bytes
        name = st.session_state.uploaded_name
        try:
            if name.lower().endswith(".docx"):
                highlighted_docx = annotate_docx(original, result)
                st.download_button(
                    "Download AI-highlighted DOCX",
                    data=highlighted_docx,
                    file_name=f"{Path(name).stem}_AI_highlighted.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="secondary"
                )
                try:
                    highlighted_pdf = _make_ai_highlighted_pdf(
                        original, name, st.session_state.doc_text, result
                    )
                    st.download_button(
                        "Download AI-highlighted PDF (cover + highlighted document)",
                        data=highlighted_pdf,
                        file_name=f"{Path(name).stem}_AI_highlighted_with_cover.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                except Exception as pdf_error:
                    st.warning(str(pdf_error))
            elif name.lower().endswith(".pdf"):
                highlighted_pdf = _make_ai_highlighted_pdf(
                    original, name, st.session_state.doc_text, result
                )
                st.download_button(
                    "Download AI-highlighted PDF (cover + highlighted document)",
                    data=highlighted_pdf,
                    file_name=f"{Path(name).stem}_AI_highlighted_with_cover.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            elif name.lower().endswith(".txt"):
                st.download_button(
                    "Download AI-highlighted review (HTML)",
                    data=make_html_review(result),
                    file_name=f"{Path(name).stem}_AI_highlighted.html",
                    mime="text/html",
                    type="primary"
                )
            else:
                st.info("For source-code files, download the AI review report. TextTrace does not validate prose AI detection on source code.")
        except Exception as e:
            st.warning(f"Could not annotate the original format: {e}. You can still download the report.")
        try:
            report_body = make_pdf_report(name, result, st.session_state.similarity_result)
            pdf = _prepend_cover_to_highlighted_pdf(
                report_body, name, st.session_state.doc_text, result, st.session_state.similarity_result
            )
            st.download_button("Download TextTrace PDF report (with cover)", data=pdf, file_name=f"{Path(name).stem}_TextTrace_report_with_cover.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Could not generate PDF report: {e}")

elif st.session_state.page == "Similarity":
    st.markdown('<div class="eyebrow">ANALYSIS TOOL / 04</div>', unsafe_allow_html=True)
    st.title("Similarity review")
    st.write("Review possible matches in your local reference library.")
    if require_document():
        if st.button("Run similarity check", type="primary", use_container_width=True):
            with st.spinner("Comparing against readable files in repository/ ..."):
                st.session_state.similarity_result = analyze_similarity(st.session_state.doc_text, APP_ROOT / "repository")
        result = st.session_state.similarity_result
        if result:
            c1, c2 = st.columns(2)
            with c1: metric("Local overlap estimate", f"{result['percent']:.1f}%")
            with c2: metric("Source matches returned", f"{len(result['sources'])}")
            st.caption(result["method"])
            if result["sources"]:
                for source in result["sources"]:
                    with st.expander(f"{source['name']} · {source['score']:.1f}% match"):
                        st.write(source["excerpt"])
                        st.caption(source["path"])
            else:
                st.info("No strong local matches were found. This does not mean the document is unique on the web.")
            st.markdown('<div class="notice">This feature checks only files in <code>repository/</code>. It does not perform a public-web search or access private Turnitin data.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Reports":
    st.markdown('<div class="eyebrow">OUTPUTS / 05</div>', unsafe_allow_html=True)
    st.title("Reports")
    st.write("Download the AI-highlighted copy and a separate TextTrace report.")
    if require_document():
        result = st.session_state.document_result
        if result:
            st.markdown(f'<div class="file-chip"><b>{st.session_state.uploaded_name.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</b><br>{len(st.session_state.doc_text.split()):,} words/tokens · {len(result["segments"]):,} segments</div>', unsafe_allow_html=True)
            try:
                report_body = make_pdf_report(st.session_state.uploaded_name, result, st.session_state.similarity_result)
                pdf = _prepend_cover_to_highlighted_pdf(
                    report_body, st.session_state.uploaded_name,
                    st.session_state.doc_text, result, st.session_state.similarity_result
                )
                st.download_button("Download PDF report (with cover)", pdf, file_name=f"{Path(st.session_state.uploaded_name).stem}_TextTrace_report_with_cover.pdf", mime="application/pdf", type="primary")
            except Exception as e: st.error(f"Report generation failed: {e}")
            raw = st.session_state.uploaded_bytes
            name = st.session_state.uploaded_name
            try:
                if name.lower().endswith(".docx"):
                    out_docx = annotate_docx(raw, result)
                    st.download_button(
                        "Download AI-highlighted original DOCX",
                        out_docx,
                        file_name=f"{Path(name).stem}_AI_highlighted.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                if name.lower().endswith((".docx", ".pdf")):
                    try:
                        out_pdf = _make_ai_highlighted_pdf(
                            raw, name, st.session_state.doc_text, result
                        )
                        st.download_button(
                            "Download AI-highlighted PDF (cover + highlighted document)",
                            out_pdf,
                            file_name=f"{Path(name).stem}_AI_highlighted_with_cover.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                    except Exception as pdf_error:
                        st.warning(str(pdf_error))
            except Exception as e:
                st.warning(f"Annotation was not available: {e}")
        else:
            st.info("Run AI Writing review to create report files.")

elif st.session_state.page == "Settings":
    st.markdown('<div class="eyebrow">PREFERENCES / 06</div>', unsafe_allow_html=True)
    st.title("Settings")
    st.write("Workspace configuration and model status.")
    config_path = MODEL_CONFIG_PATH
    if config_path.exists():
        try:
            import json
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            st.success("Model configuration file found.")
            st.json({"model_names": cfg.get("model_names"), "threshold": cfg.get("threshold"), "dataset": cfg.get("dataset")})
        except Exception as e: st.warning(f"Could not read model config: {e}")
    else:
        st.warning("No models/ensemble_config.json found. The app will use a clearly labelled demo fallback until model files are installed.")
    st.markdown('<div class="notice">No external API is connected in this package. Local similarity only checks your repository/ folder.</div>', unsafe_allow_html=True)
