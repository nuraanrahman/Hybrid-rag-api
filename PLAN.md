# M2 Week 1 Capstone — RAG API Planning Doc

> Keep this open while building. Check off each item as you go.

---

## What We're Building

A production-ready RAG (Retrieval-Augmented Generation) API.

You ask it a question → it searches your documents → it answers with citations.

**One-liner:** "A FastAPI app that ingests documents, finds the most relevant chunks using hybrid search + reranking, and answers questions with source citations."

---

## The Big Picture (Read This First)

```
Your Documents
      ↓
[Chunker] — splits text into pieces
      ↓
[Embedder] — turns chunks into vectors (numbers)
      ↓
Stored in PostgreSQL (pgvector extension)
      ↓
User asks a question
      ↓
[Hybrid Search] — BM25 keyword search + vector similarity search
      ↓
[Reranker] — Cohere picks the best results from those
      ↓
[LLM] — Claude reads the top chunks and writes an answer
      ↓
Answer + Citations returned to user
```

You build this one layer at a time. Each day = one layer.

---

## File Structure

```
app/
  main.py                    ← FastAPI app entry point
  core/
    config.py                ← All API keys and settings live here
    db.py                    ← Database connection (PostgreSQL)
    dependencies.py          ← Auth helpers (who is logged in?)
  routers/
    auth.py                  ← Login / register (copy from Week 2)
    rag.py                   ← The 3 main endpoints
  services/
    chunker.py               ← Day 1: split documents into chunks
    embedder.py              ← Day 2: turn chunks into vectors
    hybrid_retriever.py      ← Day 3: search (keywords + vectors)
    reranker.py              ← Day 4: Cohere picks the best results
    rag_pipeline.py          ← Day 5: ties everything together
  models/
    schemas.py               ← Pydantic shapes (request/response)
    db.py                    ← Database table definitions
```

---

## The 3 Endpoints

### 1. POST /rag/ingest
- **What it does:** Takes a list of text chunks → embeds them → saves to database
- **Protected:** Yes (need to be logged in)
- **Input:** List of `DocumentChunk` objects
- **Output:** `{ "inserted": 50 }`

### 2. POST /rag/search
- **What it does:** Runs hybrid search, optionally reranks, returns ranked chunks
- **Protected:** Yes
- **Input:** `{ "query": "what is RAG?", "top_k": 5, "rerank": true }`
- **Output:** List of chunks with scores

### 3. POST /rag/ask
- **What it does:** Full pipeline — searches docs, feeds to LLM, returns answer
- **Protected:** Yes
- **Input:** `{ "question": "how does chunking work?" }`
- **Output:** `{ "answer": "...", "citations": [...] }`

---

## Demo Data Plan

Two totally different topics — this proves your search actually works semantically.

| Domain | Chunks | Examples |
|--------|--------|---------|
| AI / FastAPI notes | 50 chunks | "What is RAG?", "How does pgvector work?", "FastAPI dependency injection" |
| Muay Thai training | 50 chunks | "Teep technique", "Clinch work", "Training periodization" |

If a question about Muay Thai never returns FastAPI results, your retrieval is working.

---

## Tech Stack

| What | Why |
|------|-----|
| FastAPI | The API framework |
| PostgreSQL + pgvector | Store vectors and do similarity search |
| SQLAlchemy (async) | Talk to the database from Python |
| Cohere API | Reranking (they have a free tier) |
| Claude API | Generate the final answer |
| Langfuse | See every LLM call, token cost, latency |
| pydantic-settings | Load API keys from `.env` safely |

---

## Langfuse Observability (2 Lines)

```python
# Instead of: from anthropic import Anthropic
from langfuse.anthropic import anthropic

# Everything else stays the same — Langfuse traces automatically
```

Then go to cloud.langfuse.com to see every call logged.

---

## Build Order (Day by Day)

```
Day 1 — Chunker
  [ ] chunker.py: split text by sentence/paragraph
  [ ] Test: feed a paragraph, get chunks back

Day 2 — Embedder + Database
  [ ] db.py: Chunk table with a vector column
  [ ] embedder.py: call embedding API, get vectors
  [ ] Test: embed a chunk, store it, retrieve it

Day 3 — Hybrid Search
  [ ] hybrid_retriever.py: BM25 + vector search combined
  [ ] Test: ask a question, get back ranked chunks

Day 4 — Reranker
  [ ] reranker.py: send chunks to Cohere, get reranked list
  [ ] Test: same query, better ordering

Day 5 — Full Pipeline + API
  [ ] rag_pipeline.py: wire all services together
  [ ] rag.py router: expose the 3 endpoints
  [ ] Test: POST /rag/ask with a real question

Day 6 — Demo Data + Polish
  [ ] Write 50 AI/FastAPI chunks
  [ ] Write 50 Muay Thai chunks
  [ ] Ingest both, run demo queries

Day 7 — README + Deploy
  [ ] Fill in README sections below
  [ ] Deploy to Railway / Render
```

---

## README Outline (Fill In After Building)

```markdown
# RAG API

A production RAG API with hybrid BM25+vector search, Cohere reranking,
and citation-aware answers.

## Live URL
https://your-app.railway.app

## Quick Start

# 1. Ingest a document
curl -X POST /rag/ingest ...

# 2. Search
curl -X POST /rag/search ...

# 3. Ask a question
curl -X POST /rag/ask ...

## Architecture
Chunks → Embeddings → pgvector + BM25 → Hybrid Search → Rerank → LLM → Answer

## Tech Stack
FastAPI · PostgreSQL · pgvector · Cohere · Claude · Langfuse
```

---

## API Keys You'll Need

Add these to `.env` before starting:

```
DATABASE_URL=postgresql+asyncpg://...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
SECRET_KEY=...   ← for JWT auth
```

---

## One Rule While Building

> **Don't jump ahead.** Finish each day's service, test it works in isolation, then move on.
> 
> Each service has one job. The pipeline just calls them in order.
