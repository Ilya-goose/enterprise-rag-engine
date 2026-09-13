from numpy import array, linalg, argpartition, argsort, divide, zeros_like


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.metadata = []


    def add(self, vector, chunk):
        self.vectors.append(array(vector))
        self.metadata.append(chunk)


    def search(self, query_vector, top_k):
        if not self.vectors: return []
        valid_k = min(len(self.vectors), top_k)
        if not valid_k: return []

        vector_matrix = array(self.vectors)
        dot_product = vector_matrix @ query_vector
        norms = linalg.norm(vector_matrix, axis=1) * linalg.norm(query_vector)
        scores = divide(dot_product, norms, out=zeros_like(dot_product), where=norms!=0)
        idx_best_scores = argsort(argpartition(scores, -valid_k)[-valid_k:])[::-1]
        return [self.metadata[i] for i in idx_best_scores]




