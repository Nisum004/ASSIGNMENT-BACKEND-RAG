from app.services.chunking.base import TextChunker


class FixedSizeChunker(TextChunker):
    def __init__(self, chunk_size: int = 900, overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        clean_text = " ".join(text.split())
        if not clean_text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(clean_text):
            end = start + self.chunk_size
            chunks.append(clean_text[start:end].strip())
            start = max(end - self.overlap, start + 1)
        return [chunk for chunk in chunks if chunk]

