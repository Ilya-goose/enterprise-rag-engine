from src.parser import chunk_text


def test_chunk_text_basic():
    text = "Это простой тестовый текст для проверки работы нашего чанкера"
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert chunks[0].startswith('Это')