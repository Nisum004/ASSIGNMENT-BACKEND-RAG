from sqlalchemy.orm import Session

from app.models.booking import Booking
from app.schemas.chat import BookingDetails


class BookingRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, session_id: str, details: BookingDetails) -> Booking:
        booking = Booking(
            session_id=session_id,
            name=details.name,
            email=str(details.email),
            interview_date=details.date,
            interview_time=details.time,
        )
        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)
        return booking

