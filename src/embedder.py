from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        print(f"Загрузка AI-модели: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Модель успешно загружена!")

    def encode(self, text: str) -> list[float]:
        """Превращает одну строку текста в вектор."""
        # encode возвращает numpy массив, переводим его в обычный list
        return self.model.encode(text).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Превращает список строк в список векторов (для батч-добавления)."""
        return self.model.encode(texts).tolist()