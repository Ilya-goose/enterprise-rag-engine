from numpy import array, linalg, argpartition, argsort, divide, zeros_like, arange, where
from pathlib import Path
import pickle
import uuid


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.metadata = []
        self.vectors = []
        self.columns = {}
        self.id_to_index = {}
        self.is_deleted = []


    def add(self, vector, payload) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Ожидается размерность {self.dim}, получено {len(vector)}")

        if payload is None:
            payload = {}

        self.vectors.append(array(vector))
        self.metadata.append(payload)

        if "id" not in payload.keys():
            doc_id = uuid.uuid4().hex
            payload["id"] = doc_id
        self.id_to_index[payload["id"]] = len(self.vectors) - 1
        self.is_deleted.append(False)


        for key, value in payload.items():
            if key not in self.columns:
                self.columns[key] = [None] * (len(self.vectors) - 1)
            self.columns[key].append(value)

        for key in self.columns:
            if key not in payload:
                self.columns[key].append(None)


    def search(self, query_vector, top_k, filters: dict = None) -> list[dict]:
        if len(query_vector) != self.dim:
            raise ValueError(f"Ожидается размерность {self.dim}, получено {len(query_vector)}")

        if not self.vectors: return []

        valid_k = min(len(self.vectors), top_k)
        if not valid_k: return []

        if not filters or filters is None:
            index_map = arange(len(self.vectors))
        else:
            mask = None
            for key, value in filters.items():
                if key not in self.columns:
                    index_map = []
                    break
                current_mask = (array(self.columns[key]) == value)
                mask = current_mask if mask is None else (mask & current_mask)
            else:
                index_map = where(mask)[0]

        if len(index_map) == 0:
            return []

        vector_matrix = array(self.vectors)
        filtered_vector_matrix = vector_matrix[index_map]
        dot_product = filtered_vector_matrix @ query_vector
        norms = linalg.norm(filtered_vector_matrix, axis=1) * linalg.norm(query_vector)
        scores = divide(dot_product, norms, out=zeros_like(dot_product), where=norms!=0)
        current_valid_k = min(len(scores), top_k)
        # 1. Находим топ-k индексов через argpartition
        top_indices = argpartition(scores, -current_valid_k)[-current_valid_k:]

        # 2. Сортируем ИМЕННО ПО ЗНАЧЕНИЯМ SCORES для этих индексов
        sorted_sub_indices = argsort(scores[top_indices])

        # 3. Применяем обратно и разворачиваем от большего к меньшему
        local_idx_best_scores = top_indices[sorted_sub_indices][::-1]
        global_indices = index_map[local_idx_best_scores]
        return [self.metadata[i] for i in global_indices]

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        # Автоматически создаем папки для файла, если их нет
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        path = Path(filepath)
        with open(path, "rb") as f:
            return pickle.load(f)

    def add_batch(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("Длины векторов и метаданных не совпадают")
        for i in range(len(vectors)):
            self.add(vectors[i], payloads[i])


    def get_stats(self) -> dict:
        return {"count": len(self.vectors), "dimension": self.dim}


    def delete(self, doc_id: str) -> bool:
        if doc_id not in self.id_to_index: return False
        # Получаем индекс по ключу словаря
        idx = self.id_to_index[doc_id]
        # Помечаем элемент как удаленный
        self.is_deleted[idx] = True
        # Удаляем связь из словаря индексов
        del self.id_to_index[doc_id]
        return True





