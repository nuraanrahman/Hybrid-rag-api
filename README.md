# Hybrid RAG API

A RAG API I built that combines keyword search and vector search together, with Cohere reranking on top. The goal was to make retrieval actually good — not just cosine similarity on its own, which misses a lot of exact keyword matches.

## Live Demo

> **[RENDER URL HERE]**

---

## How it works

You send in documents, they get chunked and embedded, then stored in Postgres with pgvector. When you ask a question, it runs BM25 (keyword) and vector search in parallel, merges the scores, then passes the top results through Cohere's reranker before feeding them to the LLM.

```
Document
   ↓
Chunker → Embedder → PostgreSQL + pgvector
                              ↓
User Question → BM25 + Vector Search → merge → Cohere Rerank → GPT-4o-mini
                                                                      ↓
                                              { answer, citations, tokens_used }
```

The answer comes back with `[1][2]` style citations so you know exactly which chunks were used.

---

## Stack

| | |
|---|---|
| FastAPI | async API framework |
| PostgreSQL + pgvector | vector storage |
| BM25 (rank-bm25) | in-memory keyword index |
| OpenAI text-embedding-3-small | embeddings |
| Cohere rerank-english-v3.0 | reranking |
| gpt-4o-mini | answer generation |
| JWT + bcrypt | auth |

---

## Running locally

You'll need Docker and your API keys in a `.env` file:

```
OPENAI_API_KEY=...
COHERE_API_KEY=...
SECRET_KEY=any-random-string
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragdb
```

Then:

```bash
docker-compose up
```

---

## Endpoints

### Register + login

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "you", "password": "yourpassword"}'

curl -X POST http://localhost:8000/auth/login \
  -d "username=you&password=yourpassword"
```

Copy the `access_token` from the login response.

---

### POST /rag/ingest

Send in a chunk of text with a source label:

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "RAG combines retrieval with generation so the model answers from your documents instead of just its training data.",
    "source_file": "notes.txt"
  }'
```

---

### POST /rag/search

Search without generating an answer:

```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is RAG?", "top_k": 5, "mode": "hybrid", "rerank": true}'
```

---

### POST /rag/ask

Full pipeline — retrieves, reranks, generates:

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does RAG work?", "top_k": 5}'
```

Returns:

```json
{
  "answer": "RAG works by first retrieving relevant chunks [1], then passing them as context to the LLM [2]...",
  "citations": [
    {"index": 1, "text": "...", "source_file": "notes.txt"},
    {"index": 2, "text": "...", "source_file": "notes.txt"}
  ],
  "tokens_used": 312
}
```

---

### GET /health

```bash
curl http://localhost:8000/health
# {"status": "ok", "rag_ready": true}
```

`rag_ready` flips to `true` once at least one document has been ingested.

---

## Notes

- The demo I tested with used two completely unrelated topics (AI concepts and Muay Thai training) to make sure retrieval wasn't just returning everything — if you ask about clinch work, you shouldn't get chunks about vector databases.
- Hybrid search with reranking outperforms either approach alone, especially on short or keyword-heavy queries.
