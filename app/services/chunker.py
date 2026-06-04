"""
chunker.py — splits a long text into smaller DocumentChunk pieces.

Why chunking?
  LLMs have a token limit. A whole document won't fit.
  We split it into overlapping pieces so each piece fits,
  and the overlap means ideas that span a boundary aren't lost.
"""
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.models.schemas import DocumentChunk

# Load the tokenizer once at import time — reused for every call
_enc = tiktoken.get_encoding("cl100k_base")


def chunk_semantic(
    text: str,
    source_file: str,
    chunk_size: int = 500,   # target size in characters
    overlap: int = 50,       # how many characters to repeat between chunks
) -> list[DocumentChunk]:
    """
    Split `text` into overlapping chunks using sentence/paragraph boundaries.

    The splitter tries each separator in order — double newline first, then
    single newline, then period, etc. — so it avoids cutting mid-sentence
    whenever possible.
    """
    # Build the splitter with natural language break points
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,   # measure size in characters, not tokens
    )

    # raw_chunks is just a list of plain strings
    raw_chunks = splitter.split_text(text)

    chunks: list[DocumentChunk] = []
    search_from = 0  # tracks where we left off in the original text

    for idx, chunk_text in enumerate(raw_chunks):
        # Find where this chunk sits inside the original text.
        # We search forward from search_from so we don't match an
        # earlier occurrence of the same words.
        char_start = text.find(chunk_text, search_from)
        if char_start == -1:
            # Fallback: search from the beginning (shouldn't happen often)
            char_start = text.find(chunk_text)

        char_end = char_start + len(chunk_text) if char_start != -1 else 0

        # Check if this chunk starts with a markdown heading (## Title)
        section_header = None
        for line in chunk_text.splitlines():
            if line.strip().startswith("#"):
                # Strip the # symbols and whitespace to get just the title
                section_header = line.strip().lstrip("#").strip()
                break  # only care about the first heading in the chunk

        chunks.append(DocumentChunk(
            text=chunk_text,
            token_count=len(_enc.encode(chunk_text)),  # count tokens via tiktoken
            source_file=source_file,
            chunk_index=idx,
            char_start=max(char_start, 0),
            char_end=max(char_end, 0),
            section_header=section_header,
        ))

        # Advance search_from past this chunk (minus overlap so the next
        # chunk can still find the repeated characters)
        if char_start != -1:
            search_from = char_start + max(1, len(chunk_text) - overlap)

    return chunks
