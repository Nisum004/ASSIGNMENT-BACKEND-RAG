from app.schemas.ingestion import ChunkStrategy
from app.services.chunking.base import TextChunker
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker


def get_chunker(strategy: ChunkStrategy) -> TextChunker:
    if strategy == ChunkStrategy.fixed:
        return FixedSizeChunker()
    if strategy == ChunkStrategy.recursive:
        return RecursiveChunker()
    raise ValueError(f"Unsupported chunk strategy: {strategy}")

