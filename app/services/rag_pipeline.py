"""
rag_pipeline.py — LLM answer generation with citation-grounded context.

Takes the top reranked chunks, builds a numbered context block, and calls
GPT-4o-mini to produce an answer that cites sources as [1], [2], etc.

Langfuse is used for observability when keys are configured — it auto-traces
every OpenAI call and logs tokens, latency, and cost to cloud.langfuse.com.
"""
try:
    from langfuse.openai import AsyncOpenAI
except Exception:
    from openai import AsyncOpenAI

from app.core.config import settings

# Module-level client — created once, reused for every request
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Cite sources inline as [1], [2], etc. matching the numbered context entries. "
    "If the answer is not in the context, say 'I don't have that information.'"
)


async def generate_answer(question: str, chunks: list[dict]) -> tuple[str, int]:
    """
    Build a numbered context block from chunks and generate an LLM answer.
    Returns (answer_text, total_tokens_used).
    """
    # Build context with numbered citations and section headers where available
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header   = chunk.get("section_header") or ""
        citation = f"[{i}] {chunk['source_file']}"
        if header:
            citation += f" — {header}"
        context_parts.append(f"{citation}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.0,
    )

    answer = completion.choices[0].message.content
    tokens = completion.usage.total_tokens
    return answer, tokens
