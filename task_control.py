from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Router

def list_supported_tasks(model_name_or_path):
    try:
        model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"SentenceTransformer ile yüklenemedi: {e}")
        return
    
    tasks = set()

    # 1. Prompt sisteminden
    if hasattr(model, "prompts") and model.prompts:
        tasks.update(model.prompts.keys())
        print(" + " ,model.prompts.keys())
    # 2. Router modülünden
    for _, module in model.named_children():
        if isinstance(module, Router):
            tasks.update(module.routes.keys())
            print("*" , module.routes.keys())

    if tasks:
        print(f"{model_name_or_path} modeli şu task tiplerini destekliyor:")
        for t in sorted(tasks):
            print("-", t)
    else:
        print(f"{model_name_or_path} için task bilgisi bulunamadı (tek amaçlı model olabilir).")

# Örnek kullanım
list_supported_tasks("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
