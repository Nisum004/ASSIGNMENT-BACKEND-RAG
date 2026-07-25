from app.schemas.chat import BookingDetails


def test_booking_details_accepts_required_fields() -> None:
    booking = BookingDetails(
        name="Ada Lovelace",
        email="ada@example.com",
        date="2026-08-01",
        time="10:30",
    )

    assert booking.email == "ada@example.com"

