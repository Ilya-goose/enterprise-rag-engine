from pathlib import Path

def load_text_file(file_path: str) -> str:
    try:
        with open(Path(file_path), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

