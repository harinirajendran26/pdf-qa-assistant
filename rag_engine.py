import os
import re
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

load_dotenv()

CHROMA_BASE_PATH = "./chroma_db_multi"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"


# ── Versioned path helper ─────────────────────────────────────────────────────
def get_next_chroma_path() -> str:
    """
    Returns a fresh folder path that does not exist yet.
    e.g. ./chroma_db_multi_v1, _v2, _v3 ...
    Avoids Windows file-lock errors entirely — never deletes existing folders.
    """
    i = 1
    while True:
        path = f"{CHROMA_BASE_PATH}_v{i}"
        if not os.path.exists(path):
            return path
        i += 1


# ── Model loaders — called once and cached by Streamlit ──────────────────────
def load_embeddings_model():
    """Load BGE embedding model. Heavy — 500MB. Called ONCE."""
    print("Loading BGE embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_llm():
    """Load Groq LLM client. Called ONCE."""
    print("Loading LLM...")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )


def load_vectorstore(embeddings_model):
    """
    Load the most recently created ChromaDB version from disk.
    Returns None if no DB exists yet.
    """
    i = 1
    latest_path = None
    while True:
        path = f"{CHROMA_BASE_PATH}_v{i}"
        if os.path.exists(path):
            latest_path = path
            i += 1
        else:
            break

    if latest_path:
        print(f"Loading vectorstore from: {latest_path}")
        return Chroma(
            persist_directory=latest_path,
            embedding_function=embeddings_model
        )
    return None


# ── PDF Processing ────────────────────────────────────────────────────────────
def process_pdfs(pdf_paths: list, embeddings_model) -> tuple:
    """
    Process a list of PDF file paths.
    Chunks each PDF, tags every chunk with its source filename.
    Saves to a new versioned ChromaDB folder — never touches existing folders.
    Returns (summary dict, new vectorstore).
    """
    new_chroma_path = get_next_chroma_path()
    print(f"Using new ChromaDB path: {new_chroma_path}")

    all_chunks = []
    summary    = {}

    for pdf_path in pdf_paths:
        filename  = os.path.basename(pdf_path)
        print(f"\nProcessing: {filename}")

        # Load PDF pages
        loader    = PyPDFLoader(pdf_path)
        pages     = loader.load()
        full_text = "\n\n".join([p.page_content for p in pages])

        # Extract abstract and introduction as dedicated chunks
        special_chunks = extract_special_sections(full_text, filename)

        # Find where to start regular chunking (after intro)
        background_start = full_text.find("2 Background\n")
        intro_start      = full_text.find("1 Introduction\n")

        if background_start != -1:
            remaining_start = background_start
        elif intro_start != -1:
            remaining_start = intro_start
        else:
            remaining_start = 0

        remaining_text  = full_text[remaining_start:]
        remaining_clean = clean_text(remaining_text)

        # Chunk the remaining text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        remaining_doc = Document(
            page_content=remaining_clean,
            metadata={"source": filename, "section": "main", "page": 0}
        )

        regular_chunks = splitter.split_documents([remaining_doc])

        # Tag source on special chunks
        for chunk in special_chunks:
            chunk.metadata["source"] = filename

        file_chunks = special_chunks + regular_chunks
        all_chunks.extend(file_chunks)

        summary[filename] = {
            "pages":            len(pages),
            "chunks":           len(file_chunks),
            "special_sections": len(special_chunks)
        }

        print(f"  {filename}: {len(pages)} pages → {len(file_chunks)} chunks "
              f"({len(special_chunks)} special)")

    # Embed all chunks and save to new versioned folder
    print(f"\nEmbedding {len(all_chunks)} total chunks...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings_model,
        persist_directory=new_chroma_path
    )

    print("Embedding complete!")
    return summary, vectorstore


# ── Section extraction ────────────────────────────────────────────────────────
def extract_special_sections(full_text: str, filename: str) -> list:
    """
    Extracts abstract and introduction as complete dedicated chunks.
    Must be called on RAW text before any cleaning.
    """
    special = []

    abstract_start   = full_text.find("Abstract\n")
    intro_start      = full_text.find("1 Introduction\n")
    background_start = full_text.find("2 Background\n")

    # Extract abstract
    if abstract_start != -1 and intro_start != -1:
        abstract_text = full_text[abstract_start:intro_start].strip()
        # Remove footnote lines that pollute the abstract
        lines = [
            l for l in abstract_text.split('\n')
            if not l.strip().startswith(('∗', '†', '‡', '31st', 'arXiv'))
        ]
        abstract_text = '\n'.join(lines).strip()
        special.append(Document(
            page_content=abstract_text,
            metadata={"source": filename, "section": "abstract", "page": 0}
        ))
        print(f"  Abstract extracted ({len(abstract_text)} chars)")

    # Extract introduction
    if intro_start != -1 and background_start != -1:
        intro_text = full_text[intro_start:background_start].strip()
        special.append(Document(
            page_content=intro_text,
            metadata={"source": filename, "section": "introduction", "page": 1}
        ))
        print(f"  Introduction extracted ({len(intro_text)} chars)")

    return special


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Remove emails, URLs, and very short lines from text.
    Only call AFTER section extraction — never before.
    """
    text = re.sub(r'\S+@\S+', '', text)   # remove emails
    text = re.sub(r'http\S+', '', text)    # remove URLs
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 15:
            clean_lines.append(stripped)
        elif stripped and stripped[0].isdigit():
            # Keep section headers like "3 Model Architecture"
            clean_lines.append(stripped)
    return '\n'.join(clean_lines)


# ── Answer generation ─────────────────────────────────────────────────────────
def answer_question(question: str, vectorstore, llm) -> dict:
    """
    Takes a question + loaded vectorstore + llm.
    Routes to the right retrieval strategy based on question type.
    Returns dict with answer text and sources list.
    """
    if vectorstore is None:
        return {
            "answer": "No documents loaded yet. Please upload PDFs first.",
            "sources": []
        }

    # ── Route by question type ────────────────────────────────────────────────
    summary_keywords = [
        "main idea", "summary", "overview", "about",
        "purpose", "what is this paper", "what is this document"
    ]
    is_summary = any(kw in question.lower() for kw in summary_keywords)

    if is_summary:
        # Summary questions → force abstract + introduction sections
        chunks  = vectorstore.similarity_search(
            question, k=1,
            filter={"section": {"$eq": "abstract"}}
        )
        chunks += vectorstore.similarity_search(
            question, k=1,
            filter={"section": {"$eq": "introduction"}}
        )
    else:
        # All other questions → rewrite query then MMR search
        rewritten = rewrite_query(question, llm)
        chunks    = vectorstore.max_marginal_relevance_search(
            rewritten,
            k=4,
            fetch_k=12,
            lambda_mult=0.5
        )

    if not chunks:
        return {
            "answer": "I could not find relevant information in the loaded documents.",
            "sources": []
        }

    # ── Build context with source labels ─────────────────────────────────────
    context = "\n\n".join([
        f"[From: {doc.metadata.get('source', 'unknown')}, "
        f"Section: {doc.metadata.get('section', '?')}]\n{doc.page_content}"
        for doc in chunks
    ])

    # ── Generate answer ───────────────────────────────────────────────────────
    messages = [
        SystemMessage(content="""You are a helpful research assistant.
Answer using ONLY the provided context.
Always cite which document your answer comes from using (Source: filename).
If the answer is not in the context say 'I could not find this in the documents.'"""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)

    # ── Build deduplicated source list ────────────────────────────────────────
    sources = []
    seen    = set()
    for doc in chunks:
        src = doc.metadata.get('source', 'unknown')
        sec = doc.metadata.get('section', '?')
        key = f"{src}_{sec}"
        if key not in seen:
            sources.append({
                "file":    src,
                "section": sec,
                "preview": doc.page_content[:120] + "..."
            })
            seen.add(key)

    return {
        "answer":  response.content,
        "sources": sources
    }


# ── Query rewriting ───────────────────────────────────────────────────────────
def rewrite_query(question: str, llm) -> str:
    """
    Rewrites a vague user question into a specific search query
    that better matches the vocabulary used in research papers.
    """
    messages = [
        SystemMessage(content="""You are a search query optimizer.
Rewrite the user's question into a short specific search query
for retrieving content from research papers.
Return ONLY the rewritten query, nothing else."""),
        HumanMessage(content=f"Question: {question}")
    ]
    return llm.invoke(messages).content.strip()


# ── Utility ───────────────────────────────────────────────────────────────────
def get_loaded_documents(vectorstore) -> list:
    """Returns sorted list of unique source filenames currently in the DB."""
    if vectorstore is None:
        return []
    results = vectorstore._collection.get()
    sources = set()
    for meta in results['metadatas']:
        if meta and 'source' in meta:
            sources.add(meta['source'])
    return sorted(list(sources))

# ── Conversation-aware answer ─────────────────────────────────────────────────
def answer_with_memory(question: str, chat_history: list,
                       vectorstore, llm) -> dict:
    """
    Like answer_question() but also passes recent chat history
    so the LLM understands follow-up questions.

    chat_history = list of dicts like:
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}]
    """
    if vectorstore is None:
        return {
            "answer": "No documents loaded yet. Please upload PDFs first.",
            "sources": []
        }

    # ── Build conversation context string ─────────────────────────────────────
    # Take last 4 messages (2 exchanges) to keep prompt size reasonable
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    history_text   = ""
    for msg in recent_history:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]  # truncate long answers
        history_text += f"{role}: {content}\n"

    # ── Rewrite question with conversation context ─────────────────────────────
    # If question is a follow-up like "how does it work?",
    # the rewriter uses history to understand what "it" means
    if history_text:
        rewrite_messages = [
            SystemMessage(content="""You are a search query optimizer.
Given a conversation history and a new question, rewrite the question
into a standalone specific search query that captures the full meaning.
If the question is already standalone, just clean it up.
Return ONLY the rewritten query, nothing else."""),
            HumanMessage(content=f"History:\n{history_text}\nNew question: {question}")
        ]
        rewritten = llm.invoke(rewrite_messages).content.strip()
    else:
        rewritten = rewrite_query(question, llm)

    # ── Route and retrieve ─────────────────────────────────────────────────────
    summary_keywords = ["main idea", "summary", "overview",
                        "about", "purpose", "what is this paper"]
    is_summary = any(kw in question.lower() for kw in summary_keywords)

    if is_summary:
        chunks  = vectorstore.similarity_search(
            question, k=1, filter={"section": {"$eq": "abstract"}}
        )
        chunks += vectorstore.similarity_search(
            question, k=1, filter={"section": {"$eq": "introduction"}}
        )
    else:
        chunks = vectorstore.max_marginal_relevance_search(
            rewritten, k=4, fetch_k=12, lambda_mult=0.5
        )

    if not chunks:
        return {
            "answer":  "I could not find relevant information in the documents.",
            "sources": []
        }

    # ── Build context ──────────────────────────────────────────────────────────
    context = "\n\n".join([
        f"[From: {doc.metadata.get('source','unknown')}, "
        f"Section: {doc.metadata.get('section','?')}]\n{doc.page_content}"
        for doc in chunks
    ])

    # ── Generate answer WITH history ───────────────────────────────────────────
    messages = [
        SystemMessage(content="""You are a helpful research assistant.
Answer using ONLY the provided document context.
You may use the conversation history to understand follow-up questions.
Always cite which document your answer comes from (Source: filename).
If the answer is not in the context say 'I could not find this in the documents.'"""),
        HumanMessage(content=
            f"Conversation history:\n{history_text}\n\n"
            f"Document context:\n{context}\n\n"
            f"Question: {question}"
        )
    ]

    response = llm.invoke(messages)

    # ── Build source list ──────────────────────────────────────────────────────
    sources = []
    seen    = set()
    for doc in chunks:
        src = doc.metadata.get('source', 'unknown')
        sec = doc.metadata.get('section', '?')
        key = f"{src}_{sec}"
        if key not in seen:
            sources.append({
                "file":    src,
                "section": sec,
                "preview": doc.page_content[:120] + "..."
            })
            seen.add(key)

    return {"answer": response.content, "sources": sources}