import streamlit as st
import tempfile
import os
from rag_engine import (
    load_embeddings_model,
    load_llm,
    load_vectorstore,
    process_pdfs,
    answer_with_memory,
    get_loaded_documents
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #1D9E75;
        padding: 0.5rem 0.75rem;
        border-radius: 0 6px 6px 0;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1D9E75;
    }
</style>
""", unsafe_allow_html=True)

# ── Cache heavy models ────────────────────────────────────────────────────────
@st.cache_resource
def get_cached_embeddings():
    return load_embeddings_model()

@st.cache_resource
def get_cached_llm():
    return load_llm()

embeddings_model = get_cached_embeddings()
llm              = get_cached_llm()

# ── Session state ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore(embeddings_model)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 PDF Q&A Bot")
    st.markdown("*Powered by RAG + Llama 3*")
    st.markdown("---")

    # Upload section
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDFs then click Process"
    )

    if uploaded_files:
        if st.button("⚙️ Process PDFs",
                     type="primary",
                     use_container_width=True):
            with st.spinner("Processing PDFs..."):
                final_paths = []
                for uf in uploaded_files:
                    tmp_path = os.path.join(tempfile.gettempdir(), uf.name)
                    with open(tmp_path, "wb") as f:
                        f.write(uf.read())
                    final_paths.append(tmp_path)

                summary, new_vs = process_pdfs(final_paths, embeddings_model)
                st.session_state.vectorstore = new_vs
                st.session_state.doc_stats   = summary
                st.session_state.messages    = []  # clear chat on new docs

                for path in final_paths:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

            st.success("✅ Ready to answer questions!")

    st.markdown("---")

    # Loaded documents section
    loaded = get_loaded_documents(st.session_state.vectorstore)
    if loaded:
        st.markdown("### 📄 Loaded Documents")
        for doc in loaded:
            stats = st.session_state.doc_stats.get(doc, {})
            st.markdown(f"**{doc}**")
            if stats:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<div class='stat-number'>{stats['pages']}</div>"
                        f"<div style='font-size:0.75rem;color:#666'>pages</div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"<div class='stat-number'>{stats['chunks']}</div>"
                        f"<div style='font-size:0.75rem;color:#666'>chunks</div>",
                        unsafe_allow_html=True
                    )
        st.markdown("---")

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#999'>"
        "Built with LangChain · ChromaDB<br>"
        "BGE Embeddings · Groq · Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-header'>PDF Q&A Research Assistant</div>"
    "<div class='sub-header'>"
    "Upload PDFs, ask questions, get answers with citations"
    "</div>",
    unsafe_allow_html=True
)

# Welcome / status message
if not loaded:
    st.info(
        "👈 **Get started:** Upload one or more PDF files "
        "in the sidebar and click **Process PDFs**"
    )
else:
    st.caption(
        f"💬 Chatting across: "
        f"{', '.join(loaded)} · "
        f"{len(st.session_state.messages)//2} questions asked"
    )

st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📎 View sources"):
                for s in message["sources"]:
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<strong>{s['file']}</strong> · {s['section']}<br>"
                        f"<span style='color:#666'>{s['preview']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask anything about your documents...",
    disabled=not loaded
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer with memory
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            result = answer_with_memory(
                question    = prompt,
                chat_history= st.session_state.messages[:-1],
                vectorstore = st.session_state.vectorstore,
                llm         = llm
            )
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📎 View sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<strong>{s['file']}</strong> · {s['section']}<br>"
                        f"<span style='color:#666'>{s['preview']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })