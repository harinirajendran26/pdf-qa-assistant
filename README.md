# PDF Q&A Research Assistant

A production-grade RAG (Retrieval-Augmented Generation) application 
that lets you upload multiple PDFs and ask questions across all of them.
Answers are grounded in your documents with source citations.

## Features
- Upload multiple PDFs simultaneously
- Ask questions across all documents at once
- Answers always cite which PDF and section they came from
- Conversation memory — understands follow-up questions
- Smart retrieval routing — summary questions vs concept questions handled differently
- Section-aware chunking — abstract and introduction preserved as dedicated chunks

## Tech Stack
| Component   | Technology |
| LLM         | Llama 3.3 70B via Groq API |
| Embeddings  | BAAI/bge-base-en-v1.5 (768 dims, local) |
| Vector DB   | ChromaDB (persisted to disk) |
| Framework   | LangChain |
| UI          | Streamlit |

## Architecture
```
User question
    → Question type detection (summary vs concept)
    → Query rewriting with conversation history
    → MMR retrieval from ChromaDB
    → Context injection into LLM prompt
    → Grounded answer with citations
```

## RAG Techniques Used
- **Hybrid chunking** — section extraction + recursive character splitting
- **MMR search** — balances relevance and diversity in retrieved chunks
- **Query rewriting** — rewrites vague questions into document vocabulary
- **Section-aware routing** — routes summary questions to abstract/intro directly
- **Conversation memory** — last 4 messages passed as context for follow-ups
- **Metadata filtering** — filter by section type (abstract, introduction, main)

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/pdf-qa-assistant.git
cd pdf-qa-assistant
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```
Get a free Groq key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py              # Streamlit UI
├── rag_engine.py       # RAG logic (ingestion, retrieval, generation)
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes .env, ChromaDB, venv
└── README.md           # This file
```

## Learning Journey
Built as part of a 10-day RAG learning project progressing from:
- Day 1: First RAG pipeline with 5 hardcoded strings
- Day 2: Real PDF ingestion with chunking
- Day 3: Advanced retrieval (MMR, query rewriting, metadata filtering)
- Day 4: Multi-PDF support with Streamlit UI
- Day 5: Conversation memory, polish, GitHub ready (this project)

## What I Learned
- Chunking strategy matters more than retrieval strategy
- Never clean text before extracting document structure
- Embedding model dimensions must match at build and query time
- Production RAG routes different question types to different retrieval paths
- Windows file locking requires versioned database folders