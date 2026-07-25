import re
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

_WORD_BOUNDARY = re.compile(r" {2,}")


def _looks_letter_spaced(text: str) -> bool:
    """Detect pypdf output where some PDF exporters (notably Canva) cause every
    glyph to be extracted as its own token, e.g. "N i s u m" instead of "Nisum".

    A word boundary still renders as a wider (2+) space in that output, so most
    single-space-separated tokens end up exactly one character long — unlike
    normal prose, where the vast majority of tokens are full words.
    """
    tokens = text.split(" ")
    if len(tokens) < 20:
        return False
    single_char_tokens = sum(1 for token in tokens if len(token) == 1)
    return single_char_tokens / len(tokens) > 0.6


def _collapse_letter_spacing(text: str) -> str:
    """Undo the per-glyph spacing described in `_looks_letter_spaced`, treating
    runs of 2+ spaces as the real word boundary and single spaces as artifacts
    to remove. Not lossless — two real words can rarely end up glued together
    if the PDF's own gap between them was narrower than its own word-gap
    elsewhere — but it turns unusable text into text an embedding model and an
    LLM can actually read.
    """
    lines = text.split("\n")
    fixed_lines = [
        " ".join(word.replace(" ", "") for word in _WORD_BOUNDARY.split(line))
        for line in lines
    ]
    return "\n".join(fixed_lines)


class TextExtractor:
    async def extract(self, file: UploadFile) -> str:
        content = await file.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
            )

        suffix = Path(file.filename or "").suffix.lower()
        if file.content_type == "application/pdf" or suffix == ".pdf":
            return self._extract_pdf(content)
        if file.content_type == "text/plain" or suffix == ".txt":
            return content.decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf and .txt files are supported.",
        )

    def _extract_pdf(self, content: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page.strip() for page in pages if page.strip())
        return _collapse_letter_spacing(text) if _looks_letter_spaced(text) else text
