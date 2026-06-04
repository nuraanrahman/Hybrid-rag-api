"""
rag.py — the three RAG endpoints.

  POST /rag/ingest  → chunk text, embed, store in Postgres, rebuild BM25 index
  POST /rag/search  → hybrid search with optional reranking
  POST /rag/ask     → full RAG: retrieve → rerank → generate answer
"""
import inspect
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db, AsyncSessionLocal
from app.core.dependencies import get_current_user
from app.models.db import User
from app.models.schemas import (
    IngestRequest, IngestResponse,
    SearchRequest, SearchResponse, ChunkResult,
    AskRequest, AskResponse, SourceCitation,
)
from app.services.chunker import chunk_semantic
from app.services.embedder import embed_chunks, store_chunks, embedding_cost
from app.services.hybrid_retriever import hybrid_retriever
from app.services.reranker import reranker
from app.services.rag_pipeline import generate_answer

router = APIRouter(prefix="/rag", tags=["rag"])


async def call_reranker(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Dispatch to the reranker, handling both sync and async implementations."""
    if inspect.iscoroutinefunction(reranker.rerank):
        return await reranker.rerank(query, candidates, top_n)
    return reranker.rerank(query, candidates, top_n)


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    request: IngestRequest,
    db:      AsyncSession = Depends(get_db),
    user:    User         = Depends(get_current_user),
):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Step 1: chunk the raw text into DocumentChunk objects
    chunks = chunk_semantic(request.text, request.source_file)

    # Step 2: embed all chunks in one API call
    embeddings = await embed_chunks(chunks, model=request.model)

    # Step 3: bulk-insert chunks + embeddings into Postgres
    count = await store_chunks(chunks, embeddings, db, owner_id=user.id, model=request.model)

    # Step 4: rebuild the BM25 index so new chunks are searchable immediately
    async with AsyncSessionLocal() as index_db:
        await hybrid_retriever.build_index(index_db)

    # Estimate token count and cost from response usage (approximated from chunk texts)
    token_count = sum(c.token_count for c in chunks)

    return IngestResponse(
        message         = f"Stored {count} chunks and rebuilt BM25 index",
        chunks_stored   = count,
        tokens_embedded = token_count,
        cost_usd        = embedding_cost(token_count),
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db:      AsyncSession = Depends(get_db),
    user:    User         = Depends(get_current_user),
):
    """
    Search for chunks matching the query.

    With rerank=True (default):
      1. Hybrid search → top-50 candidates
      2. Reranker → top rerank_top_n results with rerank_score added

    With rerank=False:
      Hybrid search → top top_k results (no reranker call)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    start = time.perf_counter()

    # Retrieve more candidates when reranking so the reranker has room to work
    candidate_top_k = 50 if request.rerank else request.top_k

    if request.mode == "bm25":
        raw_results = await hybrid_retriever.bm25_only_search(
            query=request.query, top_k=candidate_top_k, owner_id=user.id, db=db
        )
    else:
        raw_results = await hybrid_retriever.search(
            query=request.query, top_k=candidate_top_k, owner_id=user.id, db=db
        )

    if request.rerank and raw_results:
        raw_results = await call_reranker(request.query, raw_results, top_n=request.rerank_top_n)
    else:
        raw_results = raw_results[:request.top_k]

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(
        f"[search] mode={request.mode} rerank={request.rerank} "
        f"'{request.query[:50]}' {elapsed_ms:.1f}ms"
    )

    results = [
        ChunkResult(
            id             = row["id"],
            text           = row["text"],
            source_file    = row["source_file"],
            page_number    = row.get("page_number"),
            section_header = row.get("section_header"),
            chunk_index    = row["chunk_index"],
            score          = round(float(row.get("score", 0)), 6),
            vector_rank    = row.get("vector_rank"),
            bm25_rank      = row.get("bm25_rank"),
            rerank_score   = row.get("rerank_score"),
        )
        for row in raw_results
    ]

    return SearchResponse(query=request.query, mode=request.mode, results=results)


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    db:      AsyncSession = Depends(get_db),
    user:    User         = Depends(get_current_user),
):
    """
    Full RAG pipeline:
    1. Hybrid search → top-50 candidates
    2. Rerank        → top_k results  (skipped when rerank=False)
    3. Build context string with source citations
    4. GPT-4o-mini   → answer grounded in context only
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    # Step 1: hybrid retrieval — always pull a wide candidate set
    raw_results = await hybrid_retriever.search(
        query=request.question, top_k=50, owner_id=user.id, db=db
    )

    if not raw_results:
        return AskResponse(answer="I don't have that information.", sources=[])

    # Step 2: rerank → top_k
    if request.rerank:
        top_chunks = await call_reranker(request.question, raw_results, top_n=request.top_k)
    else:
        top_chunks = raw_results[:request.top_k]

    # Step 3 + 4: generate answer with numbered citations
    answer, _ = await generate_answer(request.question, top_chunks)

    sources = [
        SourceCitation(
            source_file    = chunk["source_file"],
            section_header = chunk.get("section_header"),
            chunk_preview  = chunk["text"][:200],
        )
        for chunk in top_chunks
    ]

    return AskResponse(answer=answer, sources=sources)
