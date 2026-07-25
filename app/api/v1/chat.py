from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ConversationalRAGService

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    service = ConversationalRAGService(db)
    return await service.chat(request)

