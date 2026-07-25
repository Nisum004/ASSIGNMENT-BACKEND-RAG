from app.schemas.ingestion import ChunkStrategy
from app.services.chunking.factory import get_chunker


def test_fixed_chunker_splits_text_with_overlap() -> None:
    text = "alpha " * 400

    chunks = get_chunker(ChunkStrategy.fixed).chunk(text)

    assert len(chunks) > 1
    assert all(chunks)


def test_recursive_chunker_preserves_paragraph_boundaries() -> None:
    text = "Intro paragraph.\n\n" + ("Details sentence. " * 120)

    chunks = get_chunker(ChunkStrategy.recursive).chunk(text)

    assert len(chunks) > 1
    assert chunks[0].startswith("Intro paragraph")

