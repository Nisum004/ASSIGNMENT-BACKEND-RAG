from itertools import pairwise

from app.services.chunking.base import TextChunker


class RecursiveChunker(TextChunker):
    def __init__(self, chunk_size: int = 1000, overlap: int = 180) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        chunks = self._split(normalized, self.separators)
        return self._add_overlap([chunk.strip() for chunk in chunks if chunk.strip()])

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        separator = separators[0]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        pieces = text.split(separator)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}{separator}{piece}" if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.extend(self._split(current, separators[1:]))
            current = piece
        if current:
            chunks.extend(self._split(current, separators[1:]))
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for previous, chunk in pairwise(chunks):
            prefix = previous[-self.overlap :]
            overlapped.append(f"{prefix} {chunk}".strip())
        return overlapped

