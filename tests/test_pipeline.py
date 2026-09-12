import pytest
from src.pipeline import process_document
from pathlib import Path


def test_process_document_success(tmp_path):
    good_file = tmp_path / "good_file.txt"
    good_file.write_bytes("Тестовая строка, которая говорит о любви".encode('utf-8'))

    res = process_document(good_file, 10, 4)
    assert isinstance(res, list)
    assert isinstance(res[0], str)


def test_process_document_empty(tmp_path):
    empty_file = tmp_path / "empty_file.txt"
    res = process_document(empty_file, 10, 4)
    assert len(res) == 0


