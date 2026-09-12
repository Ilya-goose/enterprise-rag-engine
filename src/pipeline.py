from pathlib import Path
from src.loader import load_text_file
from src.parser import chunk_text


def process_document(file_path: str | Path, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    text = load_text_file(file_path)
    if not text:
        return []

    return chunk_text(text, chunk_size, overlap)