"""
Pydantic schemas for API request/response validation.
DocumentChunk matches the output from m2-day1-chunker exactly.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
import uuid


# ── Input: what Day 1's chunker produces ──────────────────────────────────────

class DocumentChunk(BaseModel):
    """One chunk of text from a document, as produced by the Day 1 chunker."""
    chunk_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))
    text:           str
    token_count:    int
    source_file:    str
    page_number:    Optional[int] = None
    section_header: Optional[str] = None
    chunk_index:    int
    char_start:     int
    char_end:       int


# ── Auth request / response bodies ───────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ── RAG API request bodies ────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Body for POST /rag/ingest — send raw text to chunk, embed and store."""
    text:        str
    source_file: str
    model:       str = "text-embedding-3-small"

class SearchRequest(BaseModel):
    """Body for POST /rag/search — natural language query."""
    query:        str
    top_k:        int = 10
    source_file:  Optional[str] = None
    mode:         Literal["vector", "bm25", "hybrid"] = "hybrid"
    rerank:       bool = True
    rerank_top_n: int = 5

class AskRequest(BaseModel):
    """Body for POST /rag/ask — full RAG: retrieval + reranking + generation."""
    question: str
    top_k:    int = 8
    rerank:   bool = True


# ── RAG API response bodies ───────────────────────────────────────────────────

class ChunkResult(BaseModel):
    """One search result returned to the caller."""
    id:             str
    text:           str
    source_file:    str
    page_number:    Optional[int]
    section_header: Optional[str]
    chunk_index:    int
    score:          float
    vector_rank:    Optional[int]  = None
    bm25_rank:      Optional[int]  = None
    rerank_score:   Optional[float] = None

class IngestResponse(BaseModel):
    message:         str
    chunks_stored:   int
    tokens_embedded: int
    cost_usd:        float

class SearchResponse(BaseModel):
    query:   str
    mode:    str
    results: list[ChunkResult]

class SourceCitation(BaseModel):
    source_file:    str
    section_header: Optional[str]
    chunk_preview:  str

class AskResponse(BaseModel):
    answer:  str
    sources: list[SourceCitation]
