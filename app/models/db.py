"""
SQLAlchemy ORM models for the users and chunks tables.
pgvector gives us the Vector column type so we can store embeddings directly.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.core.db import Base


class User(Base):
    """One row = one registered user."""
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    chunks          = relationship("Chunk", back_populates="owner", cascade="all, delete")


class Chunk(Base):
    """One row = one document chunk + its embedding vector."""
    __tablename__ = "chunks"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    text            = Column(Text, nullable=False)
    token_count     = Column(Integer)
    source_file     = Column(String, nullable=False)
    page_number     = Column(Integer)
    section_header  = Column(String)
    chunk_index     = Column(Integer, nullable=False)
    embedding       = Column(Vector(1536))          # 1536-dim float vector
    embedding_model = Column(String, nullable=False, default="text-embedding-3-small")
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    owner           = relationship("User", back_populates="chunks")
