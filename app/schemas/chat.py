from pydantic import BaseModel, EmailStr, Field

BOOKING_FIELDS = ("name", "email", "date", "time")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class BookingDetails(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    date: str = Field(min_length=4, max_length=50)
    time: str = Field(min_length=2, max_length=50)


class BookingDraft(BaseModel):

    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None

    def is_complete(self) -> bool:
        return all(getattr(self, field) for field in BOOKING_FIELDS)

    def missing_fields(self) -> list[str]:
        return [field for field in BOOKING_FIELDS if not getattr(self, field)]


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict[str, str | int]]
    booking: BookingDetails | None = None
    booking_id: int | None = None
    booking_pending: bool = False

