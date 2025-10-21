from embedder import Embedder
from pathlib import Path
from enum import Enum, auto

# Supported task types
class TASK_TYPE(Enum):
    SEMANTIC_SIMILARITY = auto()
    CLASSIFICATION = auto()
    CLUSTERING = auto()
    RETRIEVAL_DOCUMENT = auto()
    RETRIEVAL_QUERY = auto()
    CODE_RETRIEVAL_QUERY = auto()
    QUESTION_ANSWERING = auto()
    FACT_VERIFICATION = auto()

class paraphrase_multilingual_MiniLM_L12_v2(Embedder):
    def __init__(self):
        model_path = r"models\paraphrase-multilingual-MiniLM-L12-v2"
        super().__init__(model_path)

    def _prepare_for_task(self, data, task_type):
        kwargs = {}

        # 1. Retrieval-like tasks
        if task_type == TASK_TYPE.RETRIEVAL_DOCUMENT:
            kwargs["prompt_name"] = "document"
        elif task_type == TASK_TYPE.RETRIEVAL_QUERY:
            kwargs["prompt_name"] = "query"

        # 2. Generic semantic embedding tasks handled via prompt
        elif task_type == TASK_TYPE.SEMANTIC_SIMILARITY:
            kwargs["prompt"] = "similarity: "
            print(f"[Warning] Task {task_type.name} not directly supported by MiniLM-L12-v2. Using custom prompt.")
        elif task_type == TASK_TYPE.CLUSTERING:
            kwargs["prompt"] = "clustering: "
            print(f"[Warning] Task {task_type.name} not directly supported by MiniLM-L12-v2. Using custom prompt.")
        elif task_type == TASK_TYPE.CLASSIFICATION:
            kwargs["prompt"] = "classification: "
            print(f"[Warning] Task {task_type.name} not directly supported by MiniLM-L12-v2. Using custom prompt.")
        elif task_type == TASK_TYPE.QUESTION_ANSWERING:
            kwargs["prompt"] = "qa: "
            print(f"[Warning] Task {task_type.name} not directly supported by MiniLM-L12-v2. Using custom prompt.")
        elif task_type == TASK_TYPE.FACT_VERIFICATION:
            kwargs["prompt"] = "fact check: "
            print(f"[Warning] Task {task_type.name} not directly supported by MiniLM-L12-v2. Using custom prompt.")
        elif task_type == TASK_TYPE.CODE_RETRIEVAL_QUERY:
            kwargs["prompt"] = "code retrieval: "
            print(f"[Warning] Task {task_type.name} not directly supported by MiniLM-L12-v2. Using custom prompt.")

        # 3. Unknown tasks
        else:
            print(f"[Warning] Unknown task {task_type.name}. Using default embeddings.")

        return data, kwargs

class bge_m3(Embedder):
    def __init__(self):
        model_path = r"models\bge-m3"
        super().__init__(model_path)

    def _prepare_for_task(self, data, task_type):
        kwargs = {}

        # 1. Supported tasks
        if task_type == TASK_TYPE.RETRIEVAL_DOCUMENT:
            kwargs["prompt_name"] = "document"
        elif task_type == TASK_TYPE.RETRIEVAL_QUERY:
            kwargs["prompt_name"] = "query"

        # 2. Unsupported tasks
        elif task_type == TASK_TYPE.SEMANTIC_SIMILARITY:
            kwargs["prompt"] = "similarity: "
            print(f"[Warning] Task {task_type.name} not directly supported by BGE-M3. Using custom prompt.")
        elif task_type == TASK_TYPE.CLASSIFICATION:
            kwargs["prompt"] = "classification: "
            print(f"[Warning] Task {task_type.name} not directly supported by BGE-M3. Using custom prompt.")
        elif task_type == TASK_TYPE.CLUSTERING:
            kwargs["prompt"] = "clustering: "
            print(f"[Warning] Task {task_type.name} not directly supported by BGE-M3. Using custom prompt.")
        elif task_type == TASK_TYPE.QUESTION_ANSWERING:
            kwargs["prompt"] = "qa: "
            print(f"[Warning] Task {task_type.name} not directly supported by BGE-M3. Using custom prompt.")
        elif task_type == TASK_TYPE.FACT_VERIFICATION:
            kwargs["prompt"] = "fact check: "
            print(f"[Warning] Task {task_type.name} not directly supported by BGE-M3. Using custom prompt.")
        elif task_type == TASK_TYPE.CODE_RETRIEVAL_QUERY:
            kwargs["prompt"] = "code retrieval: "
            print(f"[Warning] Task {task_type.name} not directly supported by BGE-M3. Using custom prompt.")

        # 3. Unknown tasks → fallback
        else:
            print(f"[Warning] Unknown task {task_type.name}. Using default embeddings.")

        return data, kwargs

class distiluse_base_multilingual_cased_v1(Embedder):
    def __init__(self):
        model_path = r"models\distiluse-base-multilingual-cased-v1"
        super().__init__(model_path)

    def _prepare_for_task(self, data, task_type):
        kwargs = {}

        # 1. Supported tasks
        if task_type == TASK_TYPE.RETRIEVAL_DOCUMENT:
            kwargs["prompt_name"] = "document"
        elif task_type == TASK_TYPE.RETRIEVAL_QUERY:
            kwargs["prompt_name"] = "query"

        # 2. Unsupported tasks 
        elif task_type == TASK_TYPE.CLASSIFICATION:
            kwargs["prompt"] = "classification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CLUSTERING:
            kwargs["prompt"] = "clustering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CODE_RETRIEVAL_QUERY:
            kwargs["prompt"] = "code retrieval query: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.QUESTION_ANSWERING:
            kwargs["prompt"] = "question answering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.FACT_VERIFICATION:
            kwargs["prompt"] = "fact verification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")

        # 3. Unknown tasks 
        else:
            print(f"[Warning] Unknown task {task_type.name}. Using default embeddings.")

        return data, kwargs

class multilingual_e5_small(Embedder):
    def __init__(self):
        model_path = r"models\multilingual-e5-small"
        super().__init__(model_path)

    def _prepare_for_task(self, data, task_type):
        kwargs = {}

        # 1. Supported tasks
        if task_type == TASK_TYPE.RETRIEVAL_DOCUMENT:
            kwargs["prompt_name"] = "passage"
        elif task_type == TASK_TYPE.RETRIEVAL_QUERY:
            kwargs["prompt_name"] = "query"
        elif task_type == TASK_TYPE.SEMANTIC_SIMILARITY:
            kwargs["prompt"] = "semantic similarity: "

        # 2. Unsupported tasks
        elif task_type == TASK_TYPE.CLASSIFICATION:
            kwargs["prompt"] = "classification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CLUSTERING:
            kwargs["prompt"] = "clustering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CODE_RETRIEVAL_QUERY:
            kwargs["prompt"] = "code retrieval query: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.QUESTION_ANSWERING:
            kwargs["prompt"] = "question answering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.FACT_VERIFICATION:
            kwargs["prompt"] = "fact verification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")

        # 3. Unknown tasks
        else:
            print(f"[Warning] Unknown task {task_type.name}. Using default embeddings.")

        return data, kwargs


class nomic_embed_text_v1_5(Embedder):
    def __init__(self):
        model_path = r"models\nomic-embed-text-v1.5"
        super().__init__(model_path, trust_remote_code=True)

    def _prepare_for_task(self, data, task_type):
        kwargs = {}

        # 1. Supported tasks
        if task_type == TASK_TYPE.RETRIEVAL_DOCUMENT:
            kwargs["prompt_name"] = "document"
        elif task_type == TASK_TYPE.RETRIEVAL_QUERY:
            kwargs["prompt_name"] = "query"

        # 2. Unsupported tasks 
        elif task_type == TASK_TYPE.CLASSIFICATION:
            kwargs["prompt"] = "classification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CLUSTERING:
            kwargs["prompt"] = "clustering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.SEMANTIC_SIMILARITY:
            kwargs["prompt"] = "semantic similarity: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CODE_RETRIEVAL_QUERY:
            kwargs["prompt"] = "code retrieval query: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.QUESTION_ANSWERING:
            kwargs["prompt"] = "question answering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.FACT_VERIFICATION:
            kwargs["prompt"] = "fact verification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")

        # 3. Unknown tasks
        else:
            print(f"[Warning] Unknown task {task_type.name}. Using default embeddings.")

        return data, kwargs

class nomic_embed_text_v2_moe(Embedder):
    def __init__(self):
        model_path = r"models\nomic-embed-text-v2-moe"
        super().__init__(model_path, trust_remote_code=True)

    def _prepare_for_task(self, data, task_type):
        kwargs = {}


        # 1. Tasks supported
        if task_type == TASK_TYPE.RETRIEVAL_DOCUMENT:
            kwargs["prompt_name"] = "passage"
        elif task_type == TASK_TYPE.RETRIEVAL_QUERY:
            kwargs["prompt_name"] = "query"
        elif task_type == TASK_TYPE.CLASSIFICATION:
            kwargs["prompt"] = "classification: "
        elif task_type == TASK_TYPE.CLUSTERING:
            kwargs["prompt"] = "clustering: "

        # 2. Unsupported tasks
        elif task_type == TASK_TYPE.SEMANTIC_SIMILARITY:
            kwargs["prompt"] = "semantic similarity: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.CODE_RETRIEVAL_QUERY:
            kwargs["prompt"] = "code retrieval query: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.QUESTION_ANSWERING:
            kwargs["prompt"] = "question answering: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")
        elif task_type == TASK_TYPE.FACT_VERIFICATION:
            kwargs["prompt"] = "fact verification: "
            print(f"[Warning] Task {task_type.name} not directly supported. Using custom prompt.")

        # 3. Unknown tasks
        else:
            print(f"[Warning] Unknown task {task_type.name}. Using default embeddings.")

        return data, kwargs






