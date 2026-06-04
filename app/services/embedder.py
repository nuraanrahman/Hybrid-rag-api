"""
embedder.py — two jobs:
  1. embed_chunks()  → call OpenAI once with ALL texts, get back vectors
  2. store_chunks()  → bulk-insert chunks + vectors into Postgres in one transaction
"""
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.schemas import DocumentChunk
from app.models.db import Chunk

# Create the async OpenAI client once at import time
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

_COST_PER_TOKEN = 2e-8  # $0.02 / 1M tokens


async def embed_chunks(
    chunks: list[DocumentChunk],
    model:  str = "text-embedding-3-small"
) -> list[list[float]]:
    """
    Send ALL chunk texts to OpenAI in ONE API call.
    Returns a list of vectors in the same order as the input chunks.

    Batching in one call is much faster and cheaper than one call per chunk.
    """
    # Pull out just the text from each chunk
    texts = [chunk.text for chunk in chunks]

    # Single API call — OpenAI returns one embedding per input text
    response = await openai_client.embeddings.create(model=model, input=texts)

    # OpenAI returns results in the same order as input, so we can zip them
    return [item.embedding for item in response.data]


async def store_chunks(
    chunks:     list[DocumentChunk],
    embeddings: list[list[float]],
    db:         AsyncSession,
    owner_id:   int,
    model:      str = "text-embedding-3-small"
) -> int:
    """
    Bulk-insert all chunks and their embeddings in one database transaction.
    Returns the number of rows inserted.
    """
    # Build a Chunk ORM row for each (chunk, embedding) pair
    rows = [
        Chunk(
            owner_id        = owner_id,
            text            = chunk.text,
            token_count     = chunk.token_count,
            source_file     = chunk.source_file,
            page_number     = chunk.page_number,
            section_header  = chunk.section_header,
            chunk_index     = chunk.chunk_index,
            embedding       = embedding,
            embedding_model = model,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Add all rows and commit in one shot — fastest way to bulk insert
    db.add_all(rows)
    await db.commit()

    return len(rows)


def embedding_cost(token_count: int) -> float:
    """Estimate cost in USD for a given number of embedding tokens."""
    return round(token_count * _COST_PER_TOKEN, 8)
