"""
reranker.py — Cohere reranking with FlashRank fallback.

Uses Cohere's cross-encoder (rerank-english-v3.0) when COHERE_API_KEY is set.
Falls back to FlashRank (ms-marco-MiniLM-L-12-v2, runs locally) otherwise.

Interface:
    rerank(query, candidates, top_n) → list[dict] with "rerank_score" added
"""
import asyncio

from app.core.config import settings


class CohereReranker:
    def __init__(self):
        import cohere
        self.client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

    async def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_n:      int = 5
    ) -> list[dict]:
        response = await asyncio.to_thread(
            self.client.rerank,
            model="rerank-english-v3.0",
            query=query,
            documents=[c["text"] for c in candidates],
            top_n=min(top_n, len(candidates)),
        )
        return [
            candidates[r.index] | {"rerank_score": r.relevance_score}
            for r in response.results
        ]


class FlashRankReranker:
    def __init__(self):
        from flashrank import Ranker
        self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    async def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_n:      int = 5
    ) -> list[dict]:
        from flashrank import RerankRequest
        passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
        request  = RerankRequest(query=query, passages=passages)
        results  = await asyncio.to_thread(self._ranker.rerank, request)
        top      = sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]
        return [
            candidates[r["id"]] | {"rerank_score": float(r["score"])}
            for r in top
        ]


def get_reranker():
    """Return CohereReranker if a real key is configured, else FlashRankReranker."""
    key = settings.COHERE_API_KEY.strip()
    if key and key != "...":
        print("[reranker] Using Cohere rerank-english-v3.0")
        return CohereReranker()
    print("[reranker] No Cohere key — using FlashRank (local)")
    return FlashRankReranker()


reranker = get_reranker()
