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

        vector_arr = array(vector)
        norm = linalg.norm(vector_arr)
        if norm > 0:
            vector_arr = vector_arr / norm

        if payload is None:
            payload = {}

        self.vectors.append(vector_arr)
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

        query_vector_arr = array(query_vector)
        norm_query = linalg.norm(query_vector)
        if norm_query > 0:
            query_vector_arr = query_vector_arr / norm_query

        if not self.vectors: return []

        valid_k = min(len(self.vectors), top_k)
        if not valid_k: return []

        # Создаем базовую маску: True только для НЕ удаленных элементов
        active_mask = ~array(self.is_deleted)
        if not filters or filters is None:
            # Если фильтров нет, используем только маску активности
            mask = active_mask
        else:
            # Если фильтры есть, начинаем с маски активности
            mask = active_mask
            for key, value in filters.items():
                if key not in self.columns:
                    # Если запросили неизвестную колонку, ничего не найдем
                    mask = zeros_like(mask, dtype=bool)
                    break
                current_mask = (array(self.columns[key]) == value)
                mask = mask & current_mask
        index_map = where(mask)[0]

        if len(index_map) == 0:
            return []

        vector_matrix = array(self.vectors)
        filtered_vector_matrix = vector_matrix[index_map]
        scores = filtered_vector_matrix @ query_vector_arr
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

    def vacuum(self) -> int:
        if self.is_deleted.count(True) == 0:
            return 0

        new_vectors = []
        new_metadata = []
        new_is_deleted = []
        new_id_to_index = {}

        # 1. Фильтруем живые элементы
        for i in range(len(self.vectors)):
            if not self.is_deleted[i]:
                new_vectors.append(self.vectors[i])
                new_metadata.append(self.metadata[i])
                new_is_deleted.append(False)
                new_id_to_index[self.metadata[i]["id"]] = len(new_vectors) - 1

        # 2. Пересбор колонок
        all_keys = set()
        for meta in new_metadata:
            all_keys.update(meta.keys())

        new_columns = {}
        for key in all_keys:
            new_columns[key] = [meta.get(key, None) for meta in new_metadata]

        # 3. Перезаписываем состояние базы
        count_deleted = len(self.vectors) - len(new_vectors)
        self.vectors = new_vectors
        self.metadata = new_metadata
        self.is_deleted = new_is_deleted
        self.id_to_index = new_id_to_index
        self.columns = new_columns

        return count_deleted







