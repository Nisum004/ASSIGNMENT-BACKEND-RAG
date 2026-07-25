from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkStrategy(StrEnum):
    fixed = "fixed"
    recursive = "recursive"


class IngestionResponse(BaseModel):
    document_id: int
    filename: str
    chunk_strategy: ChunkStrategy
    chunks_stored: int = Field(ge=0)

