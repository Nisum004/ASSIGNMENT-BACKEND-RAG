from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.ingestion import ChunkStrategy, IngestionResponse
from app.services.ingestion_service import DocumentIngestionService

router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse, status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    chunk_strategy: ChunkStrategy = Query(default=ChunkStrategy.recursive),
    db: Session = Depends(get_db),
) -> IngestionResponse:
    service = DocumentIngestionService(db)
    return await service.ingest(file=file, strategy=chunk_strategy)

