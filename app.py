# ============================================================
# AI Study Companion — Streamlit Web App
# Powered by Groq (Llama 3.3) + ChromaDB + LangChain
# ============================================================

import streamlit as st
import sys
import os
sys.path.append('src')

from pdf_processor import process_pdf
from embedder import create_vectorstore, load_vectorstore, vectorstore_exists
from retriever import retrieve_relevant_chunks, format_context
from llm_handler import answer_question, summarize_text, generate_quiz, explain_topic

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — AI Study Companion",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #4F8BF9;
        padding: 20px 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .feature-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #4F8BF9;
    }
    .answer-box {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #2ECC71;
    }
    .quiz-box {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #E74C3C;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 DocMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your AI-powered study companion — Upload any PDF and instantly get answers, summaries, quizzes and explanations</div>', unsafe_allow_html=True)
st.divider()

# ── Session State ─────────────────────────────────────────
# Session state persists data across Streamlit reruns
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload PDF")
    st.markdown("Upload any PDF — textbook, notes, research paper, or article.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to start studying with AI"
    )

    if uploaded_file is not None:
        # ── Save uploaded file to disk ─────────────────────
        os.makedirs("data/uploads", exist_ok=True)
        save_path = f"data/uploads/{uploaded_file.name}"

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # ── Process PDF ────────────────────────────────────
        if not vectorstore_exists(uploaded_file.name) or st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("⏳ Processing PDF... This may take a minute on first upload."):
                try:
                    chunks, full_text = process_pdf(save_path)
                    if len(chunks) == 0:
                        st.error("❌ Could not extract text from this PDF. Please use a PDF with selectable text.")
                    else:
                        create_vectorstore(chunks, uploaded_file.name)
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name      = uploaded_file.name
                        st.session_state.full_text     = full_text
                        st.session_state.chat_history  = []
                        st.success(f"✅ PDF processed! {len(chunks)} chunks created.")
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
        else:
            st.session_state.pdf_processed = True
            st.session_state.pdf_name      = uploaded_file.name
            st.success(f"✅ '{uploaded_file.name}' ready!")

    st.divider()

    if st.session_state.pdf_processed:
        st.markdown(f"**📖 Active PDF:**")
        st.markdown(f"`{st.session_state.pdf_name}`")
        st.divider()

    st.markdown("### 🤖 Powered By")
    st.markdown("- **LLM:** Groq (Llama 3.3 70B)")
    st.markdown("- **Embeddings:** sentence-transformers")
    st.markdown("- **Vector DB:** ChromaDB")
    st.markdown("- **Framework:** LangChain")

# ── Main Content ──────────────────────────────────────────
if not st.session_state.pdf_processed:
    # ── Empty State ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>💬 Q&A</h3>
        <p>Ask any question about your PDF and get instant accurate answers</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>📝 Summarize</h3>
        <p>Get a structured summary of your entire document instantly</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>🧪 Quiz</h3>
        <p>Generate multiple choice questions to test your knowledge</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
        <h3>🔍 Explain</h3>
        <p>Get difficult topics explained in simple, easy terms</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("👈 Upload a PDF from the sidebar to get started!")

else:
    # ── Feature Tabs ──────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A", "📝 Summarize", "🧪 Quiz Generator", "🔍 Explain Topic"])

    # ── Tab 1: Q&A ────────────────────────────────────────
    with tab1:
        st.header("💬 Ask Questions About Your PDF")
        st.markdown("Ask anything about the content of your uploaded PDF.")

        # Show chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        # Question input
        question = st.chat_input("Ask a question about your PDF...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        chunks  = retrieve_relevant_chunks(question, st.session_state.pdf_name, k=4)
                        context = format_context(chunks)
                        answer  = answer_question(question, context)
                        st.write(answer)

                        # Save to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer
                        })
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    # ── Tab 2: Summarize ──────────────────────────────────
    with tab2:
        st.header("📝 Document Summary")
        st.markdown("Get a structured summary of your entire PDF document.")

        col1, col2 = st.columns([2, 1])
        with col2:
            num_chunks = st.slider("Coverage (more = more detailed)", 3, 10, 6)

        if st.button("📝 Generate Summary", type="primary", use_container_width=True):
            with st.spinner("⏳ Summarizing your document..."):
                try:
                    chunks  = retrieve_relevant_chunks("summarize main topics key points", st.session_state.pdf_name, k=num_chunks)
                    context = format_context(chunks)
                    summary = summarize_text(context)

                    st.markdown("### 📋 Summary")
                    st.markdown(summary)

                    # Download button
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name=f"summary_{st.session_state.pdf_name}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ── Tab 3: Quiz Generator ─────────────────────────────
    with tab3:
        st.header("🧪 Quiz Generator")
        st.markdown("Generate multiple choice questions to test your knowledge of the PDF content.")

        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.slider("Number of questions", 3, 10, 5)
        with col2:
            quiz_topic = st.text_input("Focus on specific topic (optional)", placeholder="e.g. machine learning, chapter 2")

        if st.button("🧪 Generate Quiz", type="primary", use_container_width=True):
            with st.spinner("⏳ Generating quiz questions..."):
                try:
                    search_query = quiz_topic if quiz_topic else "main concepts key topics important facts"
                    chunks       = retrieve_relevant_chunks(search_query, st.session_state.pdf_name, k=6)
                    context      = format_context(chunks)
                    quiz         = generate_quiz(context, num_questions)

                    st.markdown("### 📋 Quiz")
                    st.markdown(quiz)

                    # Download button
                    st.download_button(
                        label="⬇️ Download Quiz",
                        data=quiz,
                        file_name=f"quiz_{st.session_state.pdf_name}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ── Tab 4: Explain Topic ──────────────────────────────
    with tab4:
        st.header("🔍 Explain Difficult Topics")
        st.markdown("Get any difficult topic from your PDF explained in simple, easy-to-understand terms.")

        topic = st.text_input(
            "Enter a topic to explain:",
            placeholder="e.g. neural networks, photosynthesis, supply and demand..."
        )

        difficulty = st.select_slider(
            "Explanation level:",
            options=["Elementary", "High School", "Undergraduate", "Expert"],
            value="High School"
        )

        if st.button("🔍 Explain This Topic", type="primary", use_container_width=True):
            if not topic:
                st.warning("⚠️ Please enter a topic to explain!")
            else:
                with st.spinner(f"⏳ Explaining '{topic}'..."):
                    try:
                        chunks  = retrieve_relevant_chunks(topic, st.session_state.pdf_name, k=4)
                        context = format_context(chunks)
                        explanation = explain_topic(
                            f"{topic} (explain at {difficulty} level)",
                            context
                        )

                        st.markdown(f"### 💡 Explanation: {topic}")
                        st.markdown(explanation)

                        # Download button
                        st.download_button(
                            label="⬇️ Download Explanation",
                            data=explanation,
                            file_name=f"explanation_{topic}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")