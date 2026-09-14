from numpy import array, linalg, argpartition, argsort, divide, zeros_like, arange, where


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.metadata = []
        self.vectors = []
        self.columns = {}


    def add(self, vector, payload):
        self.vectors.append(array(vector))
        self.metadata.append(payload)
        for key, value in payload.items():
            if key not in self.columns:
                self.columns[key] = [None] * (len(self.vectors) - 1)
            self.columns[key].append(value)

        for key in self.columns:
            if key not in payload:
                self.columns[key].append(None)


    def search(self, query_vector, top_k, filters: dict = None):
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
        local_idx_best_scores = argsort(argpartition(scores, -valid_k)[-valid_k:])[::-1]
        global_indices = index_map[local_idx_best_scores]
        return [self.metadata[i] for i in global_indices]




