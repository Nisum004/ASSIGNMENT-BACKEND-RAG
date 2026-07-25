from app.services.text_extractor import _collapse_letter_spacing, _looks_letter_spaced

# Real pypdf output captured from a Canva-exported resume, where every glyph is
# extracted as its own token and a double space marks the true word boundary.
GARBLED_SAMPLE = (
    "N i s u m  Y o n g h a n g L L M / A I / M L  E n t h u s i a s t i c\n"
    "P h o n e :  + 9 7 7 - 9 8 6 3 6 9 2 4 1 8\n"
    "E m a i l :  i z a n a m i t u b e @ g m a i l . c o m"
)

NORMAL_SAMPLE = (
    "Nisum Yonghang is a backend developer with experience in FastAPI,\n"
    "Python, and distributed systems. A B testing and I O bound tasks\n"
    "are also areas of focus."
)


def test_detects_letter_spaced_pdf_output() -> None:
    assert _looks_letter_spaced(GARBLED_SAMPLE) is True


def test_does_not_flag_normal_prose() -> None:
    assert _looks_letter_spaced(NORMAL_SAMPLE) is False


def test_collapse_reconstructs_readable_words() -> None:
    fixed = _collapse_letter_spacing(GARBLED_SAMPLE)

    assert "Phone: +977-9863692418" in fixed
    assert "Email: izanamitube@gmail.com" in fixed
    assert "Nisum" in fixed
