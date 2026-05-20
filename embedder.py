from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import time
from typing import List, Tuple


class Embedder:
    def __init__(self, model_path: str, **kwargs):
        start = time.time()
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path, **kwargs)
        duration = time.time() - start
        print(f"Model loaded in {duration:.2f} seconds from {model_path}")

    def set_max_seq_length(self, max_len: int):
        """
        Sets the maximum token length for input sequences.
        Works for all SentenceTransformer-compatible models.
        """
        self.max_seq_length = max_len
        self.model.max_seq_length = max_len

        # Ensure tokenizer truncates to the same length
        if hasattr(self.tokenizer, "model_max_length"):
            self.tokenizer.model_max_length = max_len

        print(f"max_seq_length has been set to {max_len} tokens.")


    def calculate(self, data: str | list[str], task_type=None) -> np.ndarray:
        """
        Calculates embeddings for either a single string or a list of strings.
        Delegates task-specific handling to the model before embedding.
        """
        data, kwargs = self._prepare_for_task(data, task_type)

        start = time.time()

        if isinstance(data, str):
            embeddings = self.model.encode(data, **kwargs)
            duration = time.time() - start
            print(f"Embedded 1 text in {duration:.2f} seconds.")
            return embeddings

        elif isinstance(data, list):
            embeddings = self.model.encode(data, show_progress_bar=False, **kwargs)
            duration = time.time() - start
            print(f"Embedded {len(data)} texts in {duration:.2f} seconds.")
            return embeddings

        else:
            raise TypeError("Input must be a string or list of strings")

    def _prepare_for_task(self, data, task_type):
        """
        Default: pass data through, no extra kwargs.
        Subclasses can override this to inject task-specific args.
        """
        return data, {}

    def reduce_mean(self, embeddings: np.ndarray, normalize_result: bool = True) -> np.ndarray:
        mean_embedding = np.mean(embeddings, axis=0)
        if normalize_result:
            mean_embedding = self.normalize([mean_embedding])[0]
        return mean_embedding

    def reduce_max(self, embeddings: np.ndarray, normalize_result: bool = True) -> np.ndarray:
        max_embedding = np.max(embeddings, axis=0)
        if normalize_result:
            max_embedding = self.normalize(max_embedding[np.newaxis, :])[0]
        return max_embedding

    def reduce_min(self, embeddings: np.ndarray, normalize_result: bool = True) -> np.ndarray:
        min_embedding = np.min(embeddings, axis=0)
        if normalize_result:
            min_embedding = self.normalize(min_embedding[np.newaxis, :])[0]
        return min_embedding

    def reduce_sum(self, embeddings: np.ndarray, normalize_result: bool = True) -> np.ndarray:
        sum_embedding = np.sum(embeddings, axis=0)
        if normalize_result:
            sum_embedding = self.normalize(sum_embedding[np.newaxis, :])[0]
        return sum_embedding

    def reduce_prod(self, embeddings: np.ndarray, normalize_result: bool = True) -> np.ndarray:
        prod_embedding = np.prod(embeddings, axis=0)
        if normalize_result:
            prod_embedding = self.normalize(prod_embedding[np.newaxis, :])[0]
        return prod_embedding


    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2)

    @staticmethod
    def calculate_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
        mat = np.array(vectors)
        mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        return np.dot(mat_norm, mat_norm.T)

    @staticmethod
    def calculate_distance_matrix(vectors: list[np.ndarray]) -> np.ndarray:
        mat = np.array(vectors)
        a_squared = np.sum(np.square(mat), axis=1, keepdims=True)
        b_squared = a_squared.T  # çünkü aynı matris
        cross_term = np.dot(mat, mat.T)
        return np.sqrt(a_squared - 2 * cross_term + b_squared)

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Sıfır bölme hatasına karşı koruma
        norms[norms == 0] = 1
        return vectors / norms

    @staticmethod
    def split_text(text: str, max_length: int = 200) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

