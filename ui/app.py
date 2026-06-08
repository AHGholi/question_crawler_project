# ui/app.py

import os
import sys
import yaml
from pathlib import Path

# Add project root (parent of /ui) to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Load Google credentials from config.yaml (project root)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    google_cfg = cfg.get("google", {})
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = google_cfg.get("api_key", "")
    if "GOOGLE_CSE_ID" not in os.environ and "GOOGLE_CSE_CX" not in os.environ:
        os.environ["GOOGLE_CSE_ID"] = google_cfg.get("cse_id", "") or google_cfg.get("cse_cx", "")



import io
import logging
from contextlib import contextmanager
from typing import Optional

import streamlit as st
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from main import run_main_pipeline
from main_pipeline import MainPipelineResult
from processor.pipeline import PipelineContext
from utils.models import QuestionSet


@contextmanager
def capture_logs(level: int = logging.INFO):
    """
    Capture all root logger output into a buffer.
    """
    logger = logging.getLogger()
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)

    try:
        yield buffer
    finally:
        logger.removeHandler(handler)


def question_set_to_docx(qset: QuestionSet) -> bytes:
    document = Document()
    document.add_heading("Generated Questions", level=1)

    document.add_paragraph(f"Document ID: {qset.document.id}")
    if qset.strategy:
        document.add_paragraph(f"Strategy: {qset.strategy}")

    document.add_heading("Questions", level=2)

    for idx, question in enumerate(qset.questions, start=1):
        p = document.add_paragraph(style="List Number")
        p.add_run(f"{idx}. {question.prompt}")

        if question.answer:
            document.add_paragraph(f"Answer: {question.answer}")

        if question.metadata:
            metadata_paragraph = document.add_paragraph("Metadata:")
            for key, value in question.metadata.items():
                metadata_paragraph.add_run(f"\n - {key}: {value}")

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def question_set_to_pdf(qset: QuestionSet) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Generated Questions")
    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y, f"Document ID: {qset.document.id}")
    y -= 20

    if qset.strategy:
        pdf.drawString(50, y, f"Strategy: {qset.strategy}")
        y -= 20

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Questions")
    y -= 30

    pdf.setFont("Helvetica", 11)
    line_height = 18

    for idx, question in enumerate(qset.questions, start=1):
        pdf.drawString(60, y, f"{idx}. {question.prompt}")
        y -= line_height

        if question.answer:
            pdf.drawString(60, y, f"Answer: {question.answer}")
            y -= line_height

        if question.metadata:
            pdf.drawString(60, y, "Metadata:")
            y -= line_height
            for key, value in question.metadata.items():
                pdf.drawString(80, y, f"{key}: {value}")
                y -= line_height

        y -= line_height

        if y < 80:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_context(ctx: PipelineContext):
    st.subheader("Document details")
    st.write(f"Document ID: {ctx.document.id}")
    if ctx.document.title:
        st.write(f"Title: {ctx.document.title}")

    if ctx.errors:
        st.error("This document failed:")
        for err in ctx.errors:
            st.write(f"- {err}")
        return

    if getattr(ctx, "summary", None):
        st.subheader("Summary")
        st.write(ctx.summary)
    else:
        st.info("No summary (summarizer not implemented yet).")

    if ctx.questions and ctx.questions.questions:
        qset = ctx.questions
        st.subheader("Generated Questions")

        for idx, q in enumerate(qset.questions, start=1):
            st.markdown(f"**{idx}. {q.prompt}**")
            if q.answer:
                st.write(f"Answer: {q.answer}")
            if q.metadata:
                with st.expander("Metadata", expanded=False):
                    st.json(q.metadata)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download Word (.docx)",
                data=question_set_to_docx(qset),
                file_name=f"questions_{ctx.document.id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        with col2:
            st.download_button(
                label="Download PDF",
                data=question_set_to_pdf(qset),
                file_name=f"questions_{ctx.document.id}.pdf",
                mime="application/pdf",
            )
    else:
        st.warning("No questions generated for this document.")

def main():
    st.set_page_config(page_title="Question Generator", page_icon="❓", layout="wide")
    st.title("Question Generation Pipeline")

    st.markdown(
        """
        Enter a query to crawl documents, extract text, rank sentences, and generate questions.
        Results are displayed per document.
        """
    )

    # --- Session state defaults ---
    if "result" not in st.session_state:
        st.session_state.result = None
    if "logs" not in st.session_state:
        st.session_state.logs = ""
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0

    # --- Input form (prevents rerun on each widget change) ---
    with st.form("run_form"):
        query = st.text_input(
            "Search query",
            placeholder="e.g. climate change impacts on coral reefs",
            key="query",
        )
        max_results = st.number_input(
            "Max search results",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="max_results",
        )
        limit = st.number_input(
            "Process limit (after crawling)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="limit",
        )

        submitted = st.form_submit_button("Run pipeline", type="primary")

    # --- Run pipeline only on submit ---
    if submitted:
        if not query.strip():
            st.warning("Please enter a query.")
            st.stop()

        limit_arg = None if limit == 0 else int(limit)

        with st.spinner("Running pipeline..."):
            with capture_logs() as log_buffer:
                try:
                    pipeline_result: MainPipelineResult = run_main_pipeline(
                    query=query,
                    max_results=int(max_results),
                    limit=limit_arg,
                )

                    logs = log_buffer.getvalue()
                except Exception as exc:
                    logs = log_buffer.getvalue()
                    st.error(f"Pipeline failed: {exc}")
                    if logs:
                        st.subheader("Logs")
                        st.code(logs, language="text")
                    st.stop()

        # Persist outputs across reruns
        st.session_state.result = pipeline_result
        st.session_state.logs = logs
        st.session_state.selected_idx = 0

        st.success("Pipeline completed.")

    # --- Display persisted results ---
    result = st.session_state.result
    logs = st.session_state.logs

    if result:
        st.subheader("Summary")
        st.write(f"Total documents seen: {result.stats.total_documents}")
        st.write(f"Processed: {result.stats.processed_documents}")
        st.write(f"Failed: {result.stats.failed_documents}")
        st.write(f"Total questions: {result.stats.total_questions}")

        result: Optional[MainPipelineResult] = st.session_state.result
        if result is None:
            st.info("Run the pipeline to see results.")
            return


        if not result.contexts:
            st.warning("No documents were processed.")
        else:
            st.subheader("Select a document result")

            labels = [
                f"{i+1}. {ctx.document.title or '(untitled)'} — {ctx.document.id}"
                for i, ctx in enumerate(result.contexts)
            ]

            # Clamp index in case list size changed
            if st.session_state.selected_idx >= len(labels):
                st.session_state.selected_idx = 0

            selected_idx = st.selectbox(
                "Document",
                list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=st.session_state.selected_idx,
                key="selected_idx_selectbox",
            )

            st.session_state.selected_idx = selected_idx
            render_context(result.contexts[selected_idx])

        st.subheader("Execution logs")
        if logs:
            st.code(logs, language="text")
        else:
            st.write("No logs emitted.")


if __name__ == "__main__":
    main()
