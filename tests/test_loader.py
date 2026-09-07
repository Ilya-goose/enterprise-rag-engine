import pytest
from src.loader import load_text_file
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_loader_basic():
    file_path = ROOT_DIR / "data" / "sample.txt"
    text = load_text_file(file_path)

    assert isinstance(text, str)
    assert len(text) > 0
    assert text.startswith("Компания")
    encoded_bytes = text.encode("utf-8")
    assert len(encoded_bytes) > 0


def test_loader_empty():
    file_path = ROOT_DIR / "data" / "empty.txt"
    text = load_text_file(file_path)
    assert isinstance(text, str)
    assert text == ""


def test_loader_invalid_encoding(tmp_path):
    bad_file = tmp_path / "cp1251_file.txt"
    bad_file.write_bytes("Тестовая строка".encode('cp1251'))

    with pytest.raises(UnicodeDecodeError):
        load_text_file(bad_file)



