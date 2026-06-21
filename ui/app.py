# ui\app.py

from __future__ import annotations

import io
import json
import traceback
from typing import Any, Iterable, Optional

import streamlit as st
from docx import Document as DocxDocument
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when running `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import run_main_pipeline
from main_pipeline import MainPipelineResult
from processor.pipeline import PipelineContext



# -----------------------------
# Compatibility helpers
# -----------------------------
def _question_text(q: Any) -> str:
    # Supports QuestionItem(question=...) and Question(prompt=...)
    return (getattr(q, "question", None) or getattr(q, "prompt", None) or "").strip()


def _question_answer(q: Any) -> str:
    return (getattr(q, "answer", None) or "").strip()


def _question_metadata(q: Any) -> dict:
    md = getattr(q, "metadata", None)
    return md if isinstance(md, dict) else {}


def _iter_questions(ctx: PipelineContext) -> Iterable[Any]:
    if not ctx or not ctx.questions:
        return []
    items = getattr(ctx.questions, "questions", None)
    return items if items else []


def _safe_title(ctx: PipelineContext) -> str:
    title = getattr(ctx.document, "title", None) if ctx and ctx.document else None
    if title and str(title).strip():
        return str(title).strip()
    return f"Document {getattr(ctx.document, 'id', 'unknown')}"


# -----------------------------
# Export helpers
# -----------------------------
def question_set_to_docx(contexts: list[PipelineContext]) -> bytes:
    doc = DocxDocument()
    doc.add_heading("Generated Questions", level=1)

    total = 0
    for i, ctx in enumerate(contexts, start=1):
        qs = list(_iter_questions(ctx))
        if not qs:
            continue

        doc.add_heading(f"{i}. {_safe_title(ctx)}", level=2)
        for j, q in enumerate(qs, start=1):
            text = _question_text(q)
            ans = _question_answer(q)
            md = _question_metadata(q)

            doc.add_paragraph(f"Q{j}. {text}")
            if ans:
                doc.add_paragraph(f"Answer: {ans}")
            if md:
                doc.add_paragraph(f"Metadata: {json.dumps(md, ensure_ascii=False)}")
            total += 1

    if total == 0:
        doc.add_paragraph("No questions generated.")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def question_set_to_pdf(contexts: list[PipelineContext]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x_margin = 40
    y = height - 40
    line_h = 16

    # Optional Unicode font registration fallback
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
        c.setFont("DejaVu", 11)
    except Exception:
        c.setFont("Helvetica", 11)

    def write_line(text: str = ""):
        nonlocal y
        if y < 40:
            c.showPage()
            try:
                c.setFont("DejaVu", 11)
            except Exception:
                c.setFont("Helvetica", 11)
            y = height - 40
        c.drawString(x_margin, y, text[:1300])  # hard cap
        y -= line_h

    write_line("Generated Questions")
    write_line("-" * 40)

    total = 0
    for i, ctx in enumerate(contexts, start=1):
        qs = list(_iter_questions(ctx))
        if not qs:
            continue

        write_line(f"{i}. {_safe_title(ctx)}")
        for j, q in enumerate(qs, start=1):
            text = _question_text(q)
            ans = _question_answer(q)
            md = _question_metadata(q)

            write_line(f"  Q{j}. {text}")
            if ans:
                write_line(f"     Answer: {ans}")
            if md:
                write_line(f"     Metadata: {json.dumps(md, ensure_ascii=False)}")
            total += 1
        write_line("")

    if total == 0:
        write_line("No questions generated.")

    c.save()
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# UI Rendering
# -----------------------------
def render_context(ctx: PipelineContext, idx: int) -> None:
    title = _safe_title(ctx)
    with st.expander(f"{idx}. {title}", expanded=False):
        if ctx.errors:
            st.error("Errors:")
            for err in ctx.errors:
                st.code(err)

        extraction = ctx.extraction
        if extraction:
            clean_len = len((getattr(extraction, "clean_text", "") or "").strip())
            kws = getattr(extraction, "keywords", None) or []
            top_sents = getattr(extraction, "top_sentences", None) or []
            st.caption(
                f"clean_text_len={clean_len} | keywords={len(kws)} | top_sentences={len(top_sents)}"
            )

        qs = list(_iter_questions(ctx))
        if not qs:
            st.info("No questions generated for this document.")
            return

        for q_idx, q in enumerate(qs, start=1):
            q_text = _question_text(q)
            q_ans = _question_answer(q)
            q_md = _question_metadata(q)

            st.markdown(f"**Q{q_idx}. {q_text}**")
            if q_ans:
                st.write(f"Answer: {q_ans}")
            if q_md:
                with st.expander("Metadata", expanded=False):
                    st.json(q_md)


def _stats_block(result: MainPipelineResult) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total documents", result.stats.total_documents)
    col2.metric("Processed", result.stats.processed_documents)
    col3.metric("Failed", result.stats.failed_documents)
    col4.metric("Total questions", result.stats.total_questions)


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Question Generator", layout="wide")
    st.title("Document Question Generator")

    with st.sidebar:
        st.header("Run settings")
        query = st.text_input("Query", value="machine learning basics")
        max_results = st.number_input("Max search results", min_value=1, max_value=50, value=5, step=1)
        limit = st.number_input("Process limit (0 = no limit)", min_value=0, max_value=100, value=0, step=1)
        run_btn = st.button("Run pipeline", type="primary")

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None

    if run_btn:
        st.session_state["last_error"] = None
        with st.spinner("Running pipeline..."):
            try:
                limit_arg: Optional[int] = None if int(limit) == 0 else int(limit)
                result = run_main_pipeline(
                    query=query.strip(),
                    max_results=int(max_results),
                    limit=limit_arg,
                )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.session_state["last_result"] = None
                st.session_state["last_error"] = f"{exc}\n\n{traceback.format_exc()}"

    if st.session_state["last_error"]:
        st.error("Pipeline failed")
        st.code(st.session_state["last_error"])
        return

    result: MainPipelineResult | None = st.session_state["last_result"]
    if not result:
        st.info("Set your query and click **Run pipeline**.")
        return

    _stats_block(result)

    st.divider()
    st.subheader("Per-document results")
    if not result.contexts:
        st.warning("No documents returned.")
    else:
        for i, ctx in enumerate(result.contexts, start=1):
            render_context(ctx, i)

    st.divider()
    st.subheader("Export")

    contexts_with_questions = [
        c for c in result.contexts if c.questions and getattr(c.questions, "questions", None)
    ]

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        docx_bytes = question_set_to_docx(contexts_with_questions)
        st.download_button(
            "Download DOCX",
            data=docx_bytes,
            file_name="generated_questions.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            disabled=len(contexts_with_questions) == 0,
        )

    with c2:
        pdf_bytes = question_set_to_pdf(contexts_with_questions)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="generated_questions.pdf",
            mime="application/pdf",
            disabled=len(contexts_with_questions) == 0,
        )

    with c3:
        raw = []
        for ctx in contexts_with_questions:
            raw.append(
                {
                    "document_id": getattr(ctx.document, "id", None),
                    "title": _safe_title(ctx),
                    "questions": [
                        {
                            "question": _question_text(q),
                            "answer": _question_answer(q),
                            "metadata": _question_metadata(q),
                        }
                        for q in _iter_questions(ctx)
                    ],
                }
            )
        st.download_button(
            "Download JSON",
            data=json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="generated_questions.json",
            mime="application/json",
            disabled=len(contexts_with_questions) == 0,
        )


if __name__ == "__main__":
    main()
