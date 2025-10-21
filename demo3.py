import os
import pickle
import models
import utils

EMBEDDINGS_FILE = "pdf_embeddings.pkl"
PDF_DIR = "pdfs3"



def build_pdf_embeddings(model, pdf_dir=PDF_DIR):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    pdf_embeddings = {}

    for fname in pdf_files:
        print(f"Processing: {fname}")
        text = utils.pdf_to_text(os.path.join(pdf_dir, fname))
        chunks = utils.split_document_by_structure(text, 2000, 200)
        embeddings = model.calculate(chunks, task_type=models.TASK_TYPE.SEMANTIC_SIMILARITY)
        mean_vector = model.reduce_mean(embeddings)
        pdf_embeddings[fname] = mean_vector

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(pdf_embeddings, f)
    print(f"\n Embeddings saved to {EMBEDDINGS_FILE}\n")



def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return None
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)



def search_query(model, query, pdf_embeddings):
    query_embedding = model.reduce_mean(
        model.calculate([query], task_type=models.TASK_TYPE.SEMANTIC_SIMILARITY)
    )

    results = []
    for fname, emb in pdf_embeddings.items():
        cos_sim = model.cosine_similarity(query_embedding, emb)
        euc_dist = model.euclidean_distance(query_embedding, emb)
        results.append((fname, cos_sim, euc_dist))


    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n Query: {query}\n")
    print(f"{'Rank':<5}{'PDF File':<80}{'Cosine Sim':<12}{'Eucl Dist':<12}")
    print("-" * 110)
    for rank, (fname, cos, euc) in enumerate(results, start=1):
        print(f"{rank:<5}{fname:<80}{cos:<12.4f}{euc:<12.4f}")



def main():
    model = models.paraphrase_multilingual_MiniLM_L12_v2()

    if not os.path.exists(EMBEDDINGS_FILE):
        print("No embeddings found — building now...")
        build_pdf_embeddings(model)
    else:
        print("Found existing embeddings.")

    pdf_embeddings = load_embeddings()
    if pdf_embeddings is None:
        print("Error: Failed to load embeddings.")
        return

    print("Ready to search!")
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        search_query(model, query, pdf_embeddings)


if __name__ == "__main__":
    main()
