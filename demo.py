from embedder import Embedder
from itertools import combinations
import utils



# Model paths
model_paths = {
    "paraphrase-multilingual-MiniLM-L12-v2": "models\paraphrase-multilingual-MiniLM-L12-v2",
    "bge-m3": "models\bge-m3",
    "distiluse-base-multilingual-cased-v1": "models\distiluse-base-multilingual-cased-v1",
    "multilingual-e5-small": "models\multilingual-e5-small",
    "nomic-embed-text-v1.5": "models\nomic-embed-text-v1.5",
    "nomic-embed-text-v2-moe": "models\nomic-embed-text-v2-moe",
    "qwen3-0.6B": "models\Qwen3-Embedding-0.6B"
}

# Listing and choosing models
def select_model():
    print("Kullanılabilir modeller:")
    for i, name in enumerate(model_paths.keys(), 1):
        print(f"{i}. {name}")

    while True:
        try:
            choice = int(input("Bir model numarası seçin: "))
            if 1 <= choice <= len(model_paths):
                selected_name = list(model_paths.keys())[choice - 1]
                return model_paths[selected_name]
            else:
                print("Geçerli bir numara girin.")
        except ValueError:
            print("Sayı girmeniz gerekiyor.")

# Load model with user choice
selected_model_path = select_model()
embedder = Embedder(selected_model_path)


# Sentence embedding demos

texts = [
    "What is the meaning of life?",
    "Hayatın anlamı nedir?",
    "I like this movie a lot"
]

# Embed all texts
vectors = [embedder.calculate(text) for text in texts]

print("Cosine Similarities:")
for i, j in combinations(range(len(vectors)), 2):
    cos_sim = embedder.cosine_similarity(vectors[i], vectors[j])
    print(f"{texts[i]} <-> {texts[j]}: {cos_sim:.4f}")

print("\nEuclidean Distances:")
for i, j in combinations(range(len(vectors)), 2):
    euc_dist = embedder.euclidean_distance(vectors[i], vectors[j])
    print(f"{texts[i]} <-> {texts[j]}: {euc_dist:.4f}")



# Document embedding demos
# Add your own pdfs in pdfs folder
pdf_files = [
    "Speech_recog_1.pdf",
    "Speech_recog_2.pdf",
    "Quantum_sim.pdf",
    "Speech_recog_3.pdf",
]


pdf_means = [
    embedder.reduce_mean(
        embedder.calculate(
            utils.split_document_by_structure(
                utils.pdf_to_text(f"pdfs/{fname}"), 2000 , 200
            )
        )
    )
    for fname in pdf_files
]

cos_similarities = []
euclidean_distances = []

for i in range(3):
    for j in range(i + 1, 3):
        cos_sim = embedder.cosine_similarity(pdf_means[i], pdf_means[j])
        euc_dist = embedder.euclidean_distance(pdf_means[i], pdf_means[j])
        cos_similarities.append(((i, j), cos_sim))
        euclidean_distances.append(((i, j), euc_dist))

# Print Cosine Similarities
for idx, ((i, j), sim) in enumerate(cos_similarities, start=1):
    print(f"Cosine Similarity {idx} ({pdf_files[i]} vs {pdf_files[j]}): {sim:.4f}")

# Print Euclidean Distances
for idx, ((i, j), dist) in enumerate(euclidean_distances, start=1):
    print(f"Euclidean Distance {idx} ({pdf_files[i]} vs {pdf_files[j]}): {dist:.4f}")

