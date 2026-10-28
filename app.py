import os
import json
import inspect
import importlib
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Reload local modules dynamically to prevent stale module caching across hot-reloads
import config
import database
import pdf_processor
import rag_engine
import generator
import evaluator
import diagram_engine
import blueprint_engine
import structured_output

importlib.reload(config)
importlib.reload(database)
importlib.reload(pdf_processor)
importlib.reload(rag_engine)
importlib.reload(generator)
importlib.reload(evaluator)
importlib.reload(diagram_engine)
importlib.reload(blueprint_engine)
importlib.reload(structured_output)

from config import PROJECT_INFO, MARK_SCHEMES, UPLOADS_DIR, GEMINI_MODELS
from database import (
    init_db, save_document_record, list_documents,
    save_generated_answer, list_recent_answers,
    save_evaluation, list_recent_evaluations
)
from pdf_processor import AcademicPDFProcessor
from rag_engine import AcademicRAGEngine
from generator import AcademicAnswerGenerator, NOT_ENOUGH_INFO_MSG
from evaluator import StudentAnswerEvaluator
from diagram_engine import DiagramEngine
from blueprint_engine import AcademicBlueprintEngine, compute_doc_fingerprint

# Initialize database
init_db()

# Streamlit Page Configuration - Clean Academic Layout
st.set_page_config(
    page_title="StudyGenie – AI Academic Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, Clean Academic Styling (No gradients, neon, or excessive decoration)
st.markdown("""
<style>
    /* Base typography and clean light background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #fbfbfd;
        color: #1f2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    /* Minimal Header */
    .app-header {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 14px;
        margin-bottom: 22px;
    }
    .app-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin: 0;
        letter-spacing: -0.4px;
    }
    .app-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-top: 3px;
        margin-bottom: 0;
    }

    /* Main Section Headings */
    .main-heading {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
        margin: 0 0 4px 0;
    }
    .main-description {
        font-size: 14px;
        color: #4b5563;
        margin: 0 0 18px 0;
    }

    /* Clean Card Containers */
    .clean-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }
    .section-label {
        font-size: 13px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    /* Status Boxes */
    .status-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #2563eb;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 13px;
        color: #374151;
        margin-top: 8px;
    }
    .status-box-empty {
        background-color: #f9fafb;
        border: 1px dashed #d1d5db;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 13px;
        color: #6b7280;
        margin-top: 8px;
    }
    .info-callout {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 13px;
        color: #4b5563;
    }

    /* Scorecard for Evaluation */
    .score-card-clean {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #2563eb;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }
    .score-title {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
        margin: 0;
    }
    .score-meta {
        font-size: 14px;
        color: #4b5563;
        margin-top: 4px;
    }

    /* Source Citation */
    .source-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 13px;
    }

    /* Blueprint Question Card */
    .blueprint-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
    .stButton > button {
        border-radius: 4px !important;
        font-weight: 500 !important;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 500;
        color: #4b5563;
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
def _is_valid_rag_engine(engine):
    if not engine or not hasattr(engine, "search") or not hasattr(engine, "clear"):
        return False
    try:
        sig = inspect.signature(engine.search)
        return "active_doc_ids" in sig.parameters
    except Exception:
        return False

if "rag_engine" not in st.session_state or not _is_valid_rag_engine(st.session_state.rag_engine):
    st.session_state.rag_engine = AcademicRAGEngine()
    if st.session_state.get("all_chunks"):
        st.session_state.rag_engine.build_index(st.session_state.all_chunks)
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None
if "selected_question_for_eval" not in st.session_state:
    st.session_state.selected_question_for_eval = ""
if "blueprint_data" not in st.session_state:
    st.session_state.blueprint_data = None
if "blueprint_fingerprint" not in st.session_state:
    st.session_state.blueprint_fingerprint = None
if "current_paper_evaluation" not in st.session_state:
    st.session_state.current_paper_evaluation = None
if "is_demo_mode" not in st.session_state:
    st.session_state.is_demo_mode = False
if "user_question_input" not in st.session_state:
    st.session_state.user_question_input = ""
if "target_marks_input" not in st.session_state:
    st.session_state.target_marks_input = 10
if "active_doc_ids" not in st.session_state:
    st.session_state.active_doc_ids = []

sample_pdf_path = UPLOADS_DIR / "Sample_Operating_Systems_Notes.pdf"

# Document Ingestion Helper
def process_uploaded_files(files, replace=True, api_key=None, model="gemini-2.5-flash"):
    processor = AcademicPDFProcessor()
    new_chunks = []
    new_metadata = []
    
    if replace:
        st.session_state.all_chunks = []
        st.session_state.doc_metadata = []
        st.session_state.is_demo_mode = False
        st.session_state.blueprint_data = None
        st.session_state.blueprint_fingerprint = None
        st.session_state.current_answer = None
        st.session_state.current_evaluation = None
        st.session_state.user_question_input = ""
        st.session_state.active_doc_ids = []
        if hasattr(st.session_state.rag_engine, "clear"):
            st.session_state.rag_engine.clear()
        else:
            st.session_state.rag_engine = AcademicRAGEngine()

    for f in files:
        save_path = UPLOADS_DIR / f.name
        with open(save_path, "wb") as out_f:
            out_f.write(f.getbuffer())
        
        info = processor.process_file(str(save_path), f.name)
        save_document_record(
            filename=f.name,
            file_path=str(save_path),
            total_pages=info["total_pages"],
            chunk_count=info["total_chunks"],
            file_size=len(f.getbuffer())
        )
        new_chunks.extend(info["chunks"])
        new_metadata.append(info)
    
    st.session_state.all_chunks.extend(new_chunks)
    st.session_state.doc_metadata.extend(new_metadata)
    st.session_state.rag_engine.build_index(st.session_state.all_chunks)
    
    # Update active document IDs
    st.session_state.active_doc_ids = [d.get("doc_id") for d in st.session_state.doc_metadata]

    # Generate blueprint
    bp_engine = AcademicBlueprintEngine(
        api_key=api_key or os.getenv("GEMINI_API_KEY"),
        model_name=model
    )
    st.session_state.blueprint_data = bp_engine.generate_blueprint(
        st.session_state.doc_metadata,
        st.session_state.all_chunks
    )
    st.session_state.blueprint_fingerprint = compute_doc_fingerprint(st.session_state.doc_metadata)

def load_sample_demo_mode():
    if sample_pdf_path.exists():
        processor = AcademicPDFProcessor()
        doc_info = processor.process_file(str(sample_pdf_path), "Sample_Operating_Systems_Notes.pdf")
        st.session_state.all_chunks = doc_info["chunks"]
        st.session_state.doc_metadata = [doc_info]
        st.session_state.is_demo_mode = True
        st.session_state.active_doc_ids = [doc_info.get("doc_id")]
        st.session_state.rag_engine.build_index(st.session_state.all_chunks)
        
        bp_engine = AcademicBlueprintEngine(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name="gemini-2.5-flash"
        )
        st.session_state.blueprint_data = bp_engine.generate_blueprint(
            st.session_state.doc_metadata,
            st.session_state.all_chunks
        )
        st.session_state.blueprint_fingerprint = compute_doc_fingerprint(st.session_state.doc_metadata)

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">StudyGenie</div>
    <div class="app-subtitle">AI Academic Assistant</div>
</div>
""", unsafe_allow_html=True)

# Sidebar (Narrow, Simple & Functional)
with st.sidebar:
    st.markdown("### Settings & Options")
    
    # Model Configuration
    st.markdown("##### AI Model")
    user_api_key = st.text_input(
        "Gemini API Key (Optional)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Leave blank to use autonomous local synthesis or the server key."
    )
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        st.session_state.rag_engine.api_key = user_api_key

    selected_model = st.selectbox(
        "Model Tier",
        options=GEMINI_MODELS,
        index=0
    )

    st.divider()

    # Document Selection Scope
    st.markdown("##### Active Documents")
    if st.session_state.doc_metadata:
        doc_names = [d.get("source_doc", "Document") for d in st.session_state.doc_metadata]
        if len(doc_names) > 1:
            scope_options = ["All Selected Documents"] + doc_names
            selected_scope = st.selectbox(
                "Filter retrieval to:",
                options=scope_options,
                index=0
            )
            if selected_scope == "All Selected Documents":
                st.session_state.active_doc_ids = [d.get("doc_id") for d in st.session_state.doc_metadata]
            else:
                st.session_state.active_doc_ids = [
                    d.get("doc_id") for d in st.session_state.doc_metadata 
                    if d.get("source_doc") == selected_scope
                ]
        else:
            st.session_state.active_doc_ids = [st.session_state.doc_metadata[0].get("doc_id")]
            st.caption(f"Active file: **{doc_names[0]}**")
    else:
        st.session_state.active_doc_ids = []
        st.caption("No document loaded.")

    st.write("")
    if st.button("Load Sample Notes (Demo)", use_container_width=True):
        load_sample_demo_mode()
        st.rerun()

    if st.button("Clear Documents", use_container_width=True):
        st.session_state.all_chunks = []
        st.session_state.doc_metadata = []
        st.session_state.is_demo_mode = False
        st.session_state.blueprint_data = None
        st.session_state.blueprint_fingerprint = None
        st.session_state.current_answer = None
        st.session_state.current_evaluation = None
        st.session_state.user_question_input = ""
        st.session_state.active_doc_ids = []
        if hasattr(st.session_state.rag_engine, "clear"):
            st.session_state.rag_engine.clear()
        else:
            st.session_state.rag_engine = AcademicRAGEngine()
        st.rerun()

# Top Navigation Tabs
tab_study, tab_eval, tab_blueprint, tab_explorer, tab_history = st.tabs([
    "Ask Questions",
    "Evaluate Answer",
    "Exam Blueprint",
    "Document Explorer",
    "History"
])

# =========================================================================
# TAB 1: Main Study Assistant & Question Answering
# =========================================================================
with tab_study:
    st.markdown('<div class="main-heading">Your Academic Study Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">Upload your study material and ask questions based on your notes.</div>', unsafe_allow_html=True)

    # 1. Straightforward PDF Upload Area
    with st.container():
        st.markdown('<div class="clean-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Study Material (PDF)</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload course notes or syllabus (PDF)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            replace_existing = st.checkbox(
                "Replace previously uploaded documents",
                value=True,
                help="When checked, previous notes are cleared when uploading new files."
            )
        with col_up2:
            if uploaded_files:
                if st.button("Process Document", type="primary", use_container_width=True):
                    with st.spinner("Processing document, extracting text, and building search index..."):
                        process_uploaded_files(
                            uploaded_files,
                            replace=replace_existing,
                            api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                            model=selected_model
                        )
                    st.success(f"Indexed {len(uploaded_files)} file(s) into {len(st.session_state.all_chunks)} chunks.")
                    st.rerun()

        # Uploaded status display
        if st.session_state.doc_metadata:
            doc_names = ", ".join([d.get("source_doc", "Document") for d in st.session_state.doc_metadata])
            total_p = sum([d.get("total_pages", 1) for d in st.session_state.doc_metadata])
            total_c = len(st.session_state.all_chunks)
            mode_note = " (Demo Mode)" if st.session_state.get("is_demo_mode") else ""
            st.markdown(f"""
            <div class="status-box">
                <strong>Uploaded:</strong> {doc_names}{mode_note} &nbsp;•&nbsp; 
                {total_p} page(s), {total_c} chunk(s) &nbsp;•&nbsp; Ready
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-box-empty">
                No document currently uploaded. Upload a PDF above, or click "Load Sample Notes (Demo)" in the sidebar.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Question Input Area
    with st.container():
        st.markdown('<div class="clean-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

        # Quick Pick from Active Question Bank if available
        active_presets = []
        if st.session_state.blueprint_data and st.session_state.blueprint_data.get("questions_by_marks"):
            for m in [2, 5, 10, 15]:
                for q_item in st.session_state.blueprint_data["questions_by_marks"].get(m, []):
                    active_presets.append(q_item["question"])

        if active_presets:
            selected_preset = st.selectbox(
                "Quick pick from active question bank:",
                ["-- Select a question or enter your own below --"] + active_presets,
                index=0
            )
            if selected_preset != "-- Select a question or enter your own below --":
                default_q = selected_preset
            elif st.session_state.get("user_question_input"):
                default_q = st.session_state.user_question_input
            else:
                default_q = ""
        else:
            default_q = st.session_state.get("user_question_input", "")

        user_question = st.text_area(
            "Question:",
            value=default_q,
            height=75,
            placeholder="Enter your academic question here...",
            label_visibility="collapsed"
        )

        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            default_marks_idx = 2
            preset_target = st.session_state.get("target_marks_input", 10)
            if preset_target in [2, 5, 10, 15]:
                default_marks_idx = [2, 5, 10, 15].index(preset_target)

            target_marks = st.selectbox(
                "Marks Allotted:",
                options=[2, 5, 10, 15],
                index=default_marks_idx,
                format_func=lambda m: f"{m} Marks ({MARK_SCHEMES[m]['label'].split('(')[1].replace(')', '')})"
            )
        with col_m2:
            scheme_info = MARK_SCHEMES[target_marks]
            expected_pts = target_marks * 2
            st.markdown(f"""
            <div class="info-callout" style="margin-top: 26px;">
                Expected Points: <strong>{expected_pts} points</strong> ({target_marks} marks × 2) &nbsp;|&nbsp; 
                Length: <strong>{scheme_info['expected_length']}</strong>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        if st.button("Get Answer", type="primary", use_container_width=True):
            if not user_question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.all_chunks:
                st.warning("Please upload a PDF document first.")
            else:
                with st.spinner(f"Retrieving context from notes and generating {target_marks}-mark answer..."):
                    top_k = 3 if target_marks == 2 else (5 if target_marks == 5 else 7)
                    active_ids = st.session_state.get("active_doc_ids")
                    try:
                        retrieved_chunks = st.session_state.rag_engine.search(
                            user_question, 
                            top_k=top_k,
                            active_doc_ids=active_ids
                        )
                    except TypeError:
                        st.session_state.rag_engine = AcademicRAGEngine()
                        if st.session_state.all_chunks:
                            st.session_state.rag_engine.build_index(st.session_state.all_chunks)
                        retrieved_chunks = st.session_state.rag_engine.search(
                            user_question, 
                            top_k=top_k,
                            active_doc_ids=active_ids
                        )
                    
                    generator_inst = AcademicAnswerGenerator(
                        api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                        model_name=selected_model
                    )
                    
                    result = generator_inst.generate_scheme_and_answer(
                        question=user_question,
                        marks=target_marks,
                        context_chunks=retrieved_chunks
                    )
                    
                    st.session_state.current_answer = result
                    st.session_state.selected_question_for_eval = user_question
                    
                    # Persist to local database
                    save_generated_answer(
                        doc_id=1 if st.session_state.doc_metadata else None,
                        question=user_question,
                        marks=target_marks,
                        answer_text=result["answer_markdown"],
                        scheme_dict=result["scheme"],
                        diagram_code=result.get("diagram_code", ""),
                        sources=result.get("sources", [])
                    )

        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Generated Answer Output Area
    if st.session_state.current_answer:
        ans_data = st.session_state.current_answer
        st.markdown('<div class="clean-box">', unsafe_allow_html=True)
        
        if ans_data.get("answer_markdown") == NOT_ENOUGH_INFO_MSG:
            st.warning("The uploaded PDF does not contain enough relevant information to answer this question.")
            st.info("The semantic retrieval engine found no content directly addressing this question in the active document. Please check your question or select the relevant document.")
        else:
            st.markdown(f"### Answer ({ans_data['marks']} Marks)")
            
            # Answer content
            structured_data = ans_data.get("structured_output")
            if structured_data and ans_data.get("is_structured", True):
                out_tab1, out_tab2, out_tab3 = st.tabs([
                    "Formatted Answer",
                    "Structured Sections",
                    "JSON Schema"
                ])
                with out_tab1:
                    st.markdown(ans_data["answer_markdown"])
                with out_tab2:
                    summary = structured_data.get("executive_summary")
                    if summary:
                        st.info(f"**Summary:** {summary}")
                    for sec in structured_data.get("sections", []):
                        st.markdown(f"#### {sec.get('heading', 'Section')}")
                        if sec.get("summary"):
                            st.caption(sec["summary"])
                        sec_type = sec.get("type")
                        if sec_type == "table":
                            tdata = sec.get("table_data", {})
                            headers = tdata.get("headers", [])
                            rows = tdata.get("rows", [])
                            if headers and rows:
                                table_md = "| " + " | ".join(headers) + " |\n"
                                table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                                for r in rows:
                                    table_md += "| " + " | ".join([str(cell).replace("|", "/") for cell in r]) + " |\n"
                                st.markdown(table_md)
                        elif sec_type == "key_value":
                            items = sec.get("items", [])
                            for it in items:
                                st.markdown(f"- **{it.get('key', 'Property')}:** {it.get('value', '')}")
                        elif sec_type == "ordered_steps":
                            for s in sec.get("steps", []):
                                st.markdown(f"**Step {s.get('step_number', 1)}: {s.get('phase_name', 'Phase')}** — {s.get('action', '')}")
                        elif sec_type == "bullet_points":
                            for idx_pt, pt in enumerate(sec.get("points", []), 1):
                                st.markdown(f"**{idx_pt}.** {pt}")
                with out_tab3:
                    st.json(structured_data)
            else:
                st.markdown(ans_data["answer_markdown"])

            # Diagram Display
            if ans_data.get("diagram_needed") and ans_data.get("diagram_code"):
                st.divider()
                diag_title = ans_data.get("diagram_title", "Architecture / Workflow Diagram")
                st.markdown(f"#### Diagram: {diag_title}")
                diagram_html = DiagramEngine.generate_html_embed(ans_data["diagram_code"], diag_title)
                components.html(diagram_html, height=360, scrolling=True)
                with st.expander("View Mermaid Code"):
                    st.code(ans_data["diagram_code"], language="mermaid")
            elif ans_data.get("diagram_message"):
                st.divider()
                st.caption(f"Diagram Note: {ans_data['diagram_message']}")

            # Marking Scheme
            st.divider()
            st.markdown(f"#### Marking Scheme ({ans_data['marks']} Marks)")
            scheme_obj = ans_data.get("scheme", {})
            components_list = scheme_obj.get("components", [])
            if components_list:
                scheme_rows = [{"Criterion / Component": c.get("component"), "Weightage": f"{c.get('marks')} Marks"} for c in components_list]
                st.table(scheme_rows)

            # Examiner Tips
            if ans_data.get("tips_for_full_marks"):
                with st.expander("Examiner Tips for Scoring Full Marks"):
                    for tip in ans_data["tips_for_full_marks"]:
                        st.markdown(f"- {tip}")

            # Verified Citations
            if ans_data.get("sources"):
                with st.expander(f"Verified Citations from PDF ({len(ans_data['sources'])} chunks)"):
                    for src in ans_data["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{src.get('source_doc', 'Document')}</strong> — Page {src.get('page', 1)} &nbsp;|&nbsp; Relevance: <code>{src.get('score', 'High')}</code><br/>
                            <span style="color: #4b5563;">"{src.get('text', '')[:220]}..."</span>
                        </div>
                        """, unsafe_allow_html=True)

            # Download Action
            download_content = f"""# {PROJECT_INFO['title']}
## Question: {ans_data['question']} ({ans_data['marks']} Marks)

---
### EVALUATION SCHEME
{json.dumps(ans_data.get('scheme', {}), indent=2)}

---
### MODEL ANSWER
{ans_data['answer_markdown']}

---
### DIAGRAM (MERMAID)
```mermaid
{ans_data.get('diagram_code', '')}
```
"""
            st.write("")
            st.download_button(
                label="Download Answer (Markdown)",
                data=download_content,
                file_name=f"Answer_{ans_data['marks']}M.md",
                mime="text/markdown"
            )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# TAB 2: Student Answer Evaluator
# =========================================================================
with tab_eval:
    st.markdown('<div class="main-heading">Student Answer Evaluator</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">Evaluate student answers against the official marking scheme and syllabus standards.</div>', unsafe_allow_html=True)

    ev_mode_tab1, ev_mode_tab2 = st.tabs([
        "Single Question Grading",
        "Full Paper Evaluation"
    ])

    # -------------------------------------------------------------
    # SUBTAB 1: Single Question Grading
    # -------------------------------------------------------------
    with ev_mode_tab1:
        st.markdown('<div class="clean-box">', unsafe_allow_html=True)
        eval_q = st.text_input(
            "Examination Question:",
            value=st.session_state.selected_question_for_eval or (default_q or "Enter question to evaluate"),
            placeholder="Enter question"
        )
        
        col_ev_m1, col_ev_m2 = st.columns([1, 1])
        with col_ev_m1:
            eval_marks = st.selectbox(
                "Target Marks:",
                options=[2, 5, 10, 15],
                index=2,
                key="eval_marks_select"
            )
            expected_pts_guide = eval_marks * 2
            st.caption(f"Expected points guide: **{expected_pts_guide} points** ({eval_marks} Marks × 2)")
        with col_ev_m2:
            st.write("")
            if st.button("Load Sample Answer for Testing", use_container_width=True):
                st.session_state.sample_student_input = (
                    f"Regarding {eval_q}:\n"
                    "This concept provides core functional capability and data management. "
                    "It consists of distinct modular components operating across sequential stages. "
                    "The primary advantages are modular design and predictable latency."
                )

        student_text = st.text_area(
            "Student Submitted Answer:",
            value=st.session_state.get("sample_student_input", ""),
            height=150,
            placeholder="Write or paste the student's answer here..."
        )

        col_run1, col_run2 = st.columns([2, 1])
        with col_run1:
            eval_button = st.button("Evaluate Answer", type="primary", use_container_width=True)
        with col_run2:
            if st.button("Reset Evaluation", use_container_width=True):
                st.session_state.current_evaluation = None
                st.session_state.sample_student_input = ""
                st.rerun()

        if eval_button:
            if not student_text.strip():
                st.warning("Please provide a student answer to evaluate.")
            else:
                with st.spinner("Evaluating response against marking scheme..."):
                    if (st.session_state.current_answer 
                        and st.session_state.current_answer.get("question") == eval_q 
                        and st.session_state.current_answer.get("marks") == eval_marks):
                        active_scheme = st.session_state.current_answer["scheme"]
                        model_ans = st.session_state.current_answer["answer_markdown"]
                    else:
                        active_scheme = None
                        model_ans = ""
                        # Ground dynamically from uploaded PDF via RAG
                        if st.session_state.get("all_chunks") and hasattr(st.session_state.rag_engine, "search"):
                            try:
                                top_chunks = st.session_state.rag_engine.search(
                                    eval_q,
                                    top_k=4,
                                    active_doc_ids=st.session_state.get("active_doc_ids")
                                )
                                if top_chunks:
                                    gen = AcademicAnswerGenerator(
                                        api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                                        model_name=selected_model
                                    )
                                    ref_obj = gen.generate_scheme_and_answer(eval_q, eval_marks, top_chunks)
                                    if ref_obj.get("has_sufficient_context"):
                                        model_ans = ref_obj.get("answer_markdown", "")
                                        active_scheme = ref_obj.get("scheme")
                            except Exception as rag_err:
                                print(f"Auto-reference retrieval notice: {rag_err}")

                    evaluator = StudentAnswerEvaluator(
                        api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                        model_name=selected_model
                    )
                    
                    eval_res = evaluator.evaluate_answer(
                        question=eval_q,
                        target_marks=eval_marks,
                        student_answer=student_text,
                        scheme=active_scheme,
                        model_answer=model_ans
                    )
                    
                    st.session_state.current_evaluation = eval_res
                    
                    # Persist to database
                    save_evaluation(
                        answer_id=None,
                        question=eval_q,
                        target_marks=eval_marks,
                        marks_awarded=eval_res["marks_awarded"],
                        student_answer=student_text,
                        feedback=eval_res
                    )

        # Render Evaluation Results
        if st.session_state.current_evaluation:
            ev = st.session_state.current_evaluation
            st.divider()

            # Clean Score Box
            status_text = "Pass" if ev.get("status") == "Pass" else "Re-attempt Recommended"
            st.markdown(f"""
            <div class="score-card-clean">
                <div class="score-title">Score: {ev['marks_awarded']} / {ev['target_marks']} Marks ({ev['percentage']}%)</div>
                <div class="score-meta">
                    University Grade: <strong>{ev['grade']}</strong> &nbsp;|&nbsp; Status: <strong>{status_text}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Marks Awarded", f"{ev['marks_awarded']} / {ev['target_marks']} M")
            col_m2.metric("Score Percentage", f"{ev['percentage']}%")
            col_m3.metric("University Grade", ev['grade'].split("(")[0].strip())
            col_m4.metric("Expected Points Guide", f"{ev.get('expected_points', eval_marks*2)} pts")

            # Rubric Criteria Classification
            st.markdown("##### Criteria Classification")
            col_crit1, col_crit2 = st.columns(2)
            with col_crit1:
                st.markdown("**Criteria Satisfied:**")
                if ev.get("criteria_satisfied"):
                    for s in ev["criteria_satisfied"]:
                        st.markdown(f"- {s}")
                else:
                    st.caption("No criteria fully satisfied.")

                st.markdown("**Missing Criteria:**")
                if ev.get("criteria_missing"):
                    for m in ev["criteria_missing"]:
                        st.markdown(f"- {m}")
                else:
                    st.caption("No mandatory criteria omitted.")

            with col_crit2:
                st.markdown("**Partially Satisfied Criteria:**")
                if ev.get("criteria_partially_satisfied"):
                    for p in ev["criteria_partially_satisfied"]:
                        st.markdown(f"- {p}")
                else:
                    st.caption("None.")

                st.markdown("**Incorrect Statements / Misconceptions:**")
                if ev.get("incorrect_statements"):
                    for inc in ev["incorrect_statements"]:
                        if "None detected" in inc:
                            st.caption(inc)
                        else:
                            st.markdown(f"<div style='background:#fef2f2;border:1px solid #fecaca;padding:6px 10px;margin-bottom:6px;border-radius:4px;color:#991b1b;font-size:13px;'>{inc}</div>", unsafe_allow_html=True)
                else:
                    st.caption("None detected.")

            st.divider()

            # Rubric Breakdown Table
            st.markdown(f"##### Rubric Breakdown ({ev.get('target_marks', eval_marks)} Marks)")
            comp_evals = ev.get("component_evaluations", [])
            if comp_evals:
                breakdown_table = []
                for item in comp_evals:
                    breakdown_table.append({
                        "Criterion": item.get("criterion"),
                        "Allocated": f"{item.get('allocated_marks')} M",
                        "Awarded": f"{item.get('marks_awarded')} M",
                        "Status": item.get("status"),
                        "Remarks": item.get("comments")
                    })
                st.table(breakdown_table)

            # Feedback
            st.markdown("##### Examiner Feedback")
            st.info(ev.get("examiner_feedback", "Ensure core technical definitions and diagrams are included."))

            # Human Review & Score Adjustment
            with st.expander("Score Override (Examiner Review)"):
                st.caption("Adjust score with recorded rationale for university records.")
                col_ov1, col_ov2 = st.columns(2)
                with col_ov1:
                    adj_score = st.number_input(
                        f"Adjusted Marks (Max: {ev['target_marks']}):",
                        min_value=0.0,
                        max_value=float(ev['target_marks']),
                        value=float(ev['marks_awarded']),
                        step=0.5,
                        key="adj_score_input"
                    )
                    reviewer = st.text_input("Reviewer Name:", value="Examiner", key="reviewer_name_input")
                with col_ov2:
                    override_reason = st.text_area(
                        "Override Reason:",
                        placeholder="e.g., Awarded discretionary credit for valid steps.",
                        key="override_reason_input"
                    )
                    if st.button("Apply Score Override"):
                        if not override_reason.strip():
                            st.warning("Please enter a reason for the override.")
                        else:
                            evaluator = StudentAnswerEvaluator()
                            updated_ev = evaluator.apply_human_override(
                                ev,
                                adjusted_marks=adj_score,
                                override_reason=override_reason,
                                reviewer_name=reviewer
                            )
                            st.session_state.current_evaluation = updated_ev
                            st.success(f"Score updated to {adj_score}/{ev['target_marks']} Marks.")
                            st.rerun()

            st.write("")
            st.download_button(
                label="Download Grading Report (Markdown)",
                data=ev.get("report_markdown", ""),
                file_name=f"Grading_Report_{ev.get('target_marks')}M.md",
                mime="text/markdown",
                key="dl_single_eval_md"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # SUBTAB 2: Full Paper Evaluation
    # -------------------------------------------------------------
    with ev_mode_tab2:
        st.markdown('<div class="clean-box">', unsafe_allow_html=True)
        st.markdown("##### Multi-Question Paper Evaluation")
        st.caption("Grade multiple questions together to compute total marks, percentage, and overall grade.")

        if st.session_state.blueprint_data and st.button("Pre-fill Questions from Active Blueprint"):
            bp_q_map = st.session_state.blueprint_data.get("questions_by_marks", {})
            st.session_state.paper_q1_text = bp_q_map.get(2, [{"question": "Define paging in operating systems."}])[0]["question"]
            st.session_state.paper_q1_marks = 2
            st.session_state.paper_q2_text = bp_q_map.get(5, [{"question": "Explain OSI Transport Layer functions."}])[0]["question"]
            st.session_state.paper_q2_marks = 5
            st.session_state.paper_q3_text = bp_q_map.get(10, [{"question": "Discuss Banker's Algorithm with neat diagram."}])[0]["question"]
            st.session_state.paper_q3_marks = 10
            st.success("Loaded questions into paper evaluator.")
            st.rerun()

        col_q1_a, col_q1_b = st.columns([3, 1])
        with col_q1_a:
            paper_q1 = st.text_input("Question 1:", value=st.session_state.get("paper_q1_text", "Define paging in operating systems."), key="pq1_text")
        with col_q1_b:
            paper_q1_m = st.selectbox("Q1 Marks:", [2, 5, 10, 15], index=0, key="pq1_marks")
        paper_q1_ans = st.text_area("Q1 Student Answer:", value="Paging is a memory management scheme that divides logical address space into pages and physical memory into frames.", height=65, key="pq1_ans")

        col_q2_a, col_q2_b = st.columns([3, 1])
        with col_q2_a:
            paper_q2 = st.text_input("Question 2:", value=st.session_state.get("paper_q2_text", "Explain the working of hardware address translation."), key="pq2_text")
        with col_q2_b:
            paper_q2_m = st.selectbox("Q2 Marks:", [2, 5, 10, 15], index=1, key="pq2_marks")
        paper_q2_ans = st.text_area("Q2 Student Answer:", value="CPU generates logical address consisting of page number and offset. Page table translates page number to frame number.", height=65, key="pq2_ans")

        col_q3_a, col_q3_b = st.columns([3, 1])
        with col_q3_a:
            paper_q3 = st.text_input("Question 3:", value=st.session_state.get("paper_q3_text", "Discuss Banker's Algorithm safety state verification."), key="pq3_text")
        with col_q3_b:
            paper_q3_m = st.selectbox("Q3 Marks:", [2, 5, 10, 15], index=2, key="pq3_marks")
        paper_q3_ans = st.text_area("Q3 Student Answer:", value="", placeholder="Leave blank to test an unanswered question...", height=65, key="pq3_ans")

        if st.button("Evaluate Complete Paper", type="primary", use_container_width=True):
            with st.spinner("Evaluating all questions..."):
                paper_data = [
                    {"question": paper_q1, "target_marks": paper_q1_m, "student_answer": paper_q1_ans},
                    {"question": paper_q2, "target_marks": paper_q2_m, "student_answer": paper_q2_ans},
                    {"question": paper_q3, "target_marks": paper_q3_m, "student_answer": paper_q3_ans}
                ]
                evaluator = StudentAnswerEvaluator(
                    api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                    model_name=selected_model
                )
                paper_result = evaluator.evaluate_paper(paper_data)
                st.session_state.current_paper_evaluation = paper_result

        # Display Paper Evaluation Results
        if st.session_state.get("current_paper_evaluation"):
            pres = st.session_state.current_paper_evaluation
            st.divider()

            st.markdown(f"""
            <div class="score-card-clean">
                <div class="score-title">Total Score: {pres['total_marks_obtained']} / {pres['total_maximum_marks']} Marks ({pres['percentage']}%)</div>
                <div class="score-meta">
                    Overall Grade: <strong>{pres['grade']}</strong> &nbsp;|&nbsp; Status: <strong>{pres['classification']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Examination Summary")
            summary_table_rows = []
            for eq in pres["question_evaluations"]:
                summary_table_rows.append({
                    "Question #": f"Q{eq['question_number']}",
                    "Question": eq["question"][:55] + ("..." if len(eq["question"]) > 55 else ""),
                    "Max Marks": f"{eq['target_marks']} M",
                    "Expected Points": f"{eq['expected_points']} pts",
                    "Awarded": f"{eq['marks_awarded']} M",
                    "Percentage": f"{eq['percentage']}%",
                    "Status": eq["status"]
                })
            st.table(summary_table_rows)

            with st.expander("Detailed Question-Wise Breakdown", expanded=False):
                for eq in pres["question_evaluations"]:
                    st.markdown(eq["report_markdown"])
                    st.divider()

            st.download_button(
                label="Download Examination Grade Sheet (Markdown)",
                data=pres["summary_report_markdown"],
                file_name=f"Examination_Grade_Sheet_{pres['total_marks_obtained']}_of_{pres['total_maximum_marks']}M.md",
                mime="text/markdown",
                key="dl_paper_eval_md"
            )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# TAB 3: Exam Blueprint & Important Questions
# =========================================================================
with tab_blueprint:
    st.markdown('<div class="main-heading">Exam Question Blueprint</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">Questions generated strictly from your uploaded course notes.</div>', unsafe_allow_html=True)

    if not st.session_state.doc_metadata or not st.session_state.all_chunks:
        st.info("No document currently loaded. Please upload course material in the 'Ask Questions' tab or sidebar.")
    else:
        active_ids = st.session_state.get("active_doc_ids")
        if active_ids:
            active_metadata = [d for d in st.session_state.doc_metadata if d.get("doc_id") in active_ids or d.get("source_doc") in active_ids]
            active_chunks = [c for c in st.session_state.all_chunks if c.get("doc_id") in active_ids or c.get("source_doc") in active_ids]
        else:
            active_metadata = st.session_state.doc_metadata
            active_chunks = st.session_state.all_chunks

        current_fp = compute_doc_fingerprint(active_metadata)
        source_names = [d.get("source_doc", "Document") for d in active_metadata]

        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.caption(f"Active Material: **{', '.join(source_names)}**")
        with col_hdr2:
            refresh_clicked = st.button("Refresh Blueprint", use_container_width=True)

        needs_regen = (
            st.session_state.blueprint_data is None
            or st.session_state.blueprint_fingerprint != current_fp
            or refresh_clicked
        )

        if needs_regen:
            with st.spinner("Analyzing document structure and extracting exam questions..."):
                bp_engine = AcademicBlueprintEngine(
                    api_key=user_api_key or os.getenv("GEMINI_API_KEY"),
                    model_name=selected_model
                )
                blueprint = bp_engine.generate_blueprint(
                    active_metadata,
                    active_chunks
                )
                st.session_state.blueprint_data = blueprint
                st.session_state.blueprint_fingerprint = current_fp
        else:
            blueprint = st.session_state.blueprint_data

        if blueprint.get("status") == "insufficient_content":
            st.warning(blueprint.get("message", "Insufficient academic content to formulate questions."))
        elif blueprint.get("total_questions", 0) == 0:
            st.warning("No examination questions could be extracted from the uploaded document text.")
        else:
            st.caption(f"Generated {blueprint['total_questions']} examination questions from active document(s).")
            q_by_marks = blueprint.get("questions_by_marks", {})
            
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                st.markdown("##### 2-Mark Short Answer Questions")
                q2_list = q_by_marks.get(2, [])
                if not q2_list:
                    st.caption("No 2-mark questions found.")
                else:
                    for i, q in enumerate(q2_list):
                        st.markdown(f"""
                        <div class="blueprint-card">
                            <strong>{i+1}. {q['question']}</strong><br/>
                            <span style="color: #6b7280; font-size: 12px;">{q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Use Question #{i+1} (2M)", key=f"btn_2m_{q['source_doc']}_{i}"):
                            st.session_state.user_question_input = q["question"]
                            st.session_state.target_marks_input = 2
                            st.rerun()

                st.markdown("##### 5-Mark Conceptual Questions")
                q5_list = q_by_marks.get(5, [])
                if not q5_list:
                    st.caption("No 5-mark questions found.")
                else:
                    for i, q in enumerate(q5_list):
                        st.markdown(f"""
                        <div class="blueprint-card">
                            <strong>{i+1}. {q['question']}</strong><br/>
                            <span style="color: #6b7280; font-size: 12px;">{q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Use Question #{i+1} (5M)", key=f"btn_5m_{q['source_doc']}_{i}"):
                            st.session_state.user_question_input = q["question"]
                            st.session_state.target_marks_input = 5
                            st.rerun()

            with col_bp2:
                st.markdown("##### 10-Mark Detailed Questions")
                q10_list = q_by_marks.get(10, [])
                if not q10_list:
                    st.caption("No 10-mark questions found.")
                else:
                    for i, q in enumerate(q10_list):
                        st.markdown(f"""
                        <div class="blueprint-card">
                            <strong>{i+1}. {q['question']}</strong><br/>
                            <span style="color: #6b7280; font-size: 12px;">{q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Use Question #{i+1} (10M)", key=f"btn_10m_{q['source_doc']}_{i}"):
                            st.session_state.user_question_input = q["question"]
                            st.session_state.target_marks_input = 10
                            st.rerun()

                st.markdown("##### 15-Mark Comprehensive Questions")
                q15_list = q_by_marks.get(15, [])
                if not q15_list:
                    st.caption("No 15-mark questions found.")
                else:
                    for i, q in enumerate(q15_list):
                        st.markdown(f"""
                        <div class="blueprint-card">
                            <strong>{i+1}. {q['question']}</strong><br/>
                            <span style="color: #6b7280; font-size: 12px;">{q['source_doc']} • Page {q.get('page_reference', 1)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Use Question #{i+1} (15M)", key=f"btn_15m_{q['source_doc']}_{i}"):
                            st.session_state.user_question_input = q["question"]
                            st.session_state.target_marks_input = 15
                            st.rerun()


# =========================================================================
# TAB 4: Document Explorer & Chunk Search
# =========================================================================
with tab_explorer:
    st.markdown('<div class="main-heading">Document Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">Inspect extracted pages, raw text, and segment chunks.</div>', unsafe_allow_html=True)

    if not st.session_state.doc_metadata:
        st.info("No documents indexed yet.")
    else:
        for doc in st.session_state.doc_metadata:
            with st.expander(f"{doc['source_doc']} ({doc['total_pages']} Pages, {doc['total_chunks']} Chunks)", expanded=True):
                st.caption(f"Location: `{doc['file_path']}`")
                
                page_sel = st.selectbox(
                    f"Select Page ({doc['source_doc']}):",
                    options=[p["page_number"] for p in doc["pages"]],
                    key=f"page_sel_{doc['source_doc']}"
                )
                
                selected_page_text = next((p["text"] for p in doc["pages"] if p["page_number"] == page_sel), "")
                
                p_words = len(selected_page_text.split())
                p_chars = len(selected_page_text)
                
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Words", p_words)
                col_p2.metric("Characters", p_chars)
                
                st.text_area("Page Text:", value=selected_page_text, height=140, disabled=True, key=f"raw_{doc['source_doc']}_{page_sel}")

        st.divider()
        st.markdown("##### Semantic Search Tester")
        search_query = st.text_input("Enter search query to test similarity retrieval:")
        if search_query:
            try:
                results = st.session_state.rag_engine.search(
                    search_query, 
                    top_k=4, 
                    active_doc_ids=st.session_state.get("active_doc_ids")
                )
            except TypeError:
                st.session_state.rag_engine = AcademicRAGEngine()
                if st.session_state.all_chunks:
                    st.session_state.rag_engine.build_index(st.session_state.all_chunks)
                results = st.session_state.rag_engine.search(
                    search_query, 
                    top_k=4, 
                    active_doc_ids=st.session_state.get("active_doc_ids")
                )
            for r in results:
                st.markdown(f"""
                <div class="source-box">
                    <strong>Rank #{r['rank']}</strong> &nbsp;|&nbsp; 
                    Score: <code>{r['score']}</code> &nbsp;|&nbsp; Page {r['page']} ({r['source_doc']})<br/>
                    <div style="margin-top: 4px; color: #1f2937;">{r['text']}</div>
                </div>
                """, unsafe_allow_html=True)


# =========================================================================
# TAB 5: History & Records
# =========================================================================
with tab_history:
    st.markdown('<div class="main-heading">History & Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">Saved answers and student grading records.</div>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("##### Recent Generated Answers")
        recent_ans = list_recent_answers(limit=8)
        if not recent_ans:
            st.caption("No answers generated yet.")
        else:
            for item in recent_ans:
                with st.expander(f"[{item['marks']}M] {item['question'][:45]}..."):
                    st.caption(f"Date: {item['created_at']}")
                    st.markdown(item["answer_text"][:250] + "...")
    
    with col_h2:
        st.markdown("##### Recent Student Evaluations")
        recent_evs = list_recent_evaluations(limit=8)
        if not recent_evs:
            st.caption("No student evaluations recorded yet.")
        else:
            for ev_item in recent_evs:
                with st.expander(f"Score: {ev_item['marks_awarded']}/{ev_item['target_marks']} — {ev_item['question'][:35]}..."):
                    st.caption(f"Excerpt: {ev_item['student_answer'][:100]}...")
                    feedback_obj = ev_item.get("feedback", {})
                    st.caption(f"Grade: {feedback_obj.get('grade', 'N/A')}")
