"""
hybrid_retriever.py — combines BM25 keyword search with vector (cosine) search.

How it works:
  1. Vector search  → finds chunks semantically similar to the query
  2. BM25 search    → finds chunks with exact/close keyword matches
  3. RRF fusion     → merges both ranked lists into one final ranking

Why this beats either alone:
  - BM25 catches exact identifiers (error codes, product names, jargon)
  - Vector catches semantic meaning (synonyms, paraphrasing)
  - RRF rewards chunks that appear high in BOTH lists

The BM25 index is built in memory at startup from all DB chunks.
Call build_index() again after ingesting new data.
"""
import uuid as uuid_lib
from typing import Optional

from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Chunk
from app.services.embedder import embed_chunks
from app.models.schemas import DocumentChunk


class HybridRetriever:
    def __init__(self):
        self.bm25:        Optional[BM25Okapi] = None
        self.chunk_ids:   list[str]           = []  # parallel lists: index N = same chunk
        self.chunk_texts: list[str]           = []
        self.owner_ids:   list[int]           = []  # used to filter per-user

    # ── Index building ────────────────────────────────────────────────────────

    async def build_index(self, db: AsyncSession):
        """
        Load all chunks from the DB and build an in-memory BM25 index.
        Call this at startup and after every /ingest.
        """
        result = await db.execute(select(Chunk.id, Chunk.text, Chunk.owner_id))
        rows = result.fetchall()

        self.chunk_ids   = [str(row.id) for row in rows]
        self.chunk_texts = [row.text for row in rows]
        self.owner_ids   = [row.owner_id for row in rows]

        # BM25 needs tokenized text — simple whitespace split is enough here
        tokenized = [t.lower().split() for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

        print(f"[BM25] Index built with {len(self.chunk_ids)} chunks")

    # ── BM25 search ───────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int, owner_id: int) -> list[tuple[str, float]]:
        """
        Score every chunk with BM25 and return the top_k best matches for this user.
        Returns: list of (chunk_id, bm25_score) sorted best-first.
        """
        if self.bm25 is None or not self.chunk_ids:
            return []

        scores = self.bm25.get_scores(query.lower().split())

        # Filter to only this user's chunks, then sort by score descending
        owned = [
            (self.chunk_ids[i], float(scores[i]))
            for i in range(len(scores))
            if self.owner_ids[i] == owner_id
        ]
        return sorted(owned, key=lambda x: x[1], reverse=True)[:top_k]

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        vector_ids: list[str],
        bm25_ids:   list[str],
        k:          int = 60
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion: merge two ranked lists into one.

        Formula per chunk: score += 1 / (k + rank)
        A chunk that appears at rank 1 in BOTH lists beats one that's
        rank 1 in only one list. k=60 is the research-standard default.

        Returns: list of (chunk_id, rrf_score) sorted best-first.
        """
        scores: dict[str, float] = {}

        for rank, chunk_id in enumerate(vector_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        for rank, chunk_id in enumerate(bm25_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _fetch_chunks_by_ids(
        self,
        chunk_ids: list[str],
        db:        AsyncSession
    ) -> dict[str, dict]:
        """
        Fetch chunk rows from DB by a list of UUID strings.
        Returns a dict: { chunk_id_str: row_dict }
        """
        if not chunk_ids:
            return {}

        # Convert string IDs back to UUID objects for the ORM query
        uuid_ids = [uuid_lib.UUID(cid) for cid in chunk_ids]

        result = await db.execute(select(Chunk).where(Chunk.id.in_(uuid_ids)))
        rows = result.scalars().all()

        return {
            str(row.id): {
                "id":             str(row.id),
                "text":           row.text,
                "source_file":    row.source_file,
                "page_number":    row.page_number,
                "section_header": row.section_header,
                "chunk_index":    row.chunk_index,
            }
            for row in rows
        }

    # ── Public search methods ─────────────────────────────────────────────────

    async def search(
        self,
        query:    str,
        top_k:    int = 10,
        owner_id: int = 0,
        db:       AsyncSession = None
    ) -> list[dict]:
        """
        Full hybrid search: vector + BM25 → RRF fusion.

        Steps:
          1. Embed query, cosine search → top 50 chunk IDs (by vector similarity)
          2. BM25 search                → top 50 chunk IDs (by keyword match)
          3. RRF fusion                 → merge and re-rank, keep top_k
          4. Fetch full rows from DB for the final IDs
          5. Return with text, metadata, rrf_score, vector_rank, bm25_rank
        """
        # Step 1: vector search — get top 50 IDs for this user ordered by cosine distance
        dummy = DocumentChunk(text=query, token_count=0, source_file="", chunk_index=0, char_start=0, char_end=0)
        query_vector = (await embed_chunks([dummy]))[0]
        sql = text("""
            SELECT id::text
            FROM chunks
            WHERE owner_id = :owner_id
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT 50
        """)
        result = await db.execute(sql, {"vec": str(query_vector), "owner_id": owner_id})
        vector_ids = [row[0] for row in result.fetchall()]

        # Step 2: BM25 search — get top 50 IDs ordered by keyword score
        bm25_pairs = self._bm25_search(query, top_k=50, owner_id=owner_id)
        bm25_ids   = [chunk_id for chunk_id, _ in bm25_pairs]

        # Build rank lookup maps (1-based: rank 1 = best)
        vector_rank_map = {cid: rank + 1 for rank, cid in enumerate(vector_ids)}
        bm25_rank_map   = {cid: rank + 1 for rank, cid in enumerate(bm25_ids)}

        # Step 3: RRF fusion — merge both lists, take top_k
        fused     = self._rrf_fusion(vector_ids, bm25_ids)[:top_k]
        final_ids = [cid for cid, _ in fused]
        rrf_score_map = {cid: score for cid, score in fused}

        # Step 4: fetch full chunk data from DB
        chunk_data = await self._fetch_chunks_by_ids(final_ids, db)

        # Step 5: build results in RRF-ranked order
        results = []
        for cid in final_ids:
            if cid not in chunk_data:
                continue
            row = chunk_data[cid]
            results.append({
                **row,
                "score":       round(rrf_score_map[cid], 6),
                "vector_rank": vector_rank_map.get(cid),
                "bm25_rank":   bm25_rank_map.get(cid),
            })
        return results

    async def bm25_only_search(
        self,
        query:    str,
        top_k:    int = 10,
        owner_id: int = 0,
        db:       AsyncSession = None
    ) -> list[dict]:
        """BM25-only search — pure keyword matching, no embeddings needed."""
        bm25_pairs = self._bm25_search(query, top_k=top_k, owner_id=owner_id)
        if not bm25_pairs:
            return []

        top_ids   = [cid for cid, _ in bm25_pairs]
        score_map = {cid: score for cid, score in bm25_pairs}

        chunk_data = await self._fetch_chunks_by_ids(top_ids, db)

        results = []
        for cid in top_ids:
            if cid not in chunk_data:
                continue
            row = chunk_data[cid]
            results.append({
                **row,
                "score":       round(score_map[cid], 6),
                "vector_rank": None,
                "bm25_rank":   None,
            })
        return results


hybrid_retriever = HybridRetriever()
