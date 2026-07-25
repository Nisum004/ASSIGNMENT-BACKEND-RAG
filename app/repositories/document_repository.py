from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_document(self, filename: str, content_type: str, chunk_strategy: str) -> Document:
        document = Document(
            filename=filename,
            content_type=content_type,
            chunk_strategy=chunk_strategy,
        )
        self.db.add(document)
        self.db.flush()
        return document

    def add_chunks(self, document_id: int, chunks: list[tuple[str, int, str]]) -> None:
        for vector_id, chunk_index, text in chunks:
            self.db.add(
                DocumentChunk(
                    document_id=document_id,
                    vector_id=vector_id,
                    chunk_index=chunk_index,
                    text=text,
                )
            )

    def commit(self) -> None:
        self.db.commit()

