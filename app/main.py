"""
main.py — FastAPI app with hybrid search + reranking + RAG generation.

Endpoints:
  GET  /health      → liveness check, reports whether BM25 index is ready
  POST /auth/...    → register, login, /me
  POST /rag/ingest  → chunk text, embed, store in Postgres, rebuild BM25 index
  POST /rag/search  → hybrid search with optional reranking
  POST /rag/ask     → full RAG: retrieve → rerank → generate answer with GPT-4o-mini

Startup: create DB tables + pgvector extension, then rebuild BM25 index
         from whatever chunks are already in the database.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.core.db import engine, Base, AsyncSessionLocal
import app.models.db  # noqa: registers ORM models with Base metadata
from app.routers import auth, rag
from app.services.hybrid_retriever import hybrid_retriever
from app.services.reranker import get_reranker

reranker = get_reranker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Creating tables and pgvector extension...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    print("[startup] Building BM25 index from existing chunks...")
    async with AsyncSessionLocal() as db:
        await hybrid_retriever.build_index(db)

    print(f"[startup] Reranker: {type(reranker).__name__}")
    print("[startup] Ready.")
    yield


app = FastAPI(
    title   = "RAG API — M2 Week 1",
    version = "1.0.0",
    lifespan = lifespan,
)

app.include_router(auth.router)
app.include_router(rag.router)

_static = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(_static / "index.html")


@app.get("/health", tags=["health"])
async def health():
    return {
        "status":    "ok",
        "rag_ready": hybrid_retriever.bm25 is not None,
    }
