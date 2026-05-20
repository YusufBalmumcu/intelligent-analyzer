from models import nomic_embed_text_v2_moe, paraphrase_multilingual_MiniLM_L12_v2 , TASK_TYPE
import utils
import csv
import json

# Initialize model
model = paraphrase_multilingual_MiniLM_L12_v2()

if hasattr(model, "name"):
    model_name = model.name
elif hasattr(model, "model_name"):
    model_name = model.model_name
elif hasattr(model, "__class__"):  
    model_name = model.__class__.__name__  # fallback to class name
else:
    model_name = "unknown_model"

pdf_files = [
    "Speech_recog_1.pdf",
    "Speech_recog_2.pdf",
    "Quantum_sim.pdf",
]

# Parameters for chunking
chunk_start = 10   # initial chunk size
chunk_step = 5    # how much to increase per iteration
chunk_end = 2200   # maximum chunk size to test
overlap_percent = 10   # overlap percentage (e.g. 5 means 5%)

results = {}

for chunk_size in range(chunk_start, chunk_end + 1, chunk_step):
    overlap_size = int(chunk_size * overlap_percent / 100)  # percentage-based overlap
    print(f"\n=== Evaluating with chunk size = {chunk_size}, overlap = {overlap_size} ===")
    
    # Compute mean embeddings for each PDF
    pdf_means = []
    for fname in pdf_files:
        text = utils.pdf_to_text(f"pdfs/{fname}")
        chunks = utils.split_document_by_structure(text, chunk_size, overlap_size)
        embeddings = model.calculate(chunks, task_type=TASK_TYPE.SEMANTIC_SIMILARITY)
        pdf_means.append(model.reduce_mean(embeddings))
    
    # Compute similarities
    cos_similarities = []
    for i in range(len(pdf_means)):
        for j in range(i + 1, len(pdf_means)):
            cos_sim = float(model.cosine_similarity(pdf_means[i], pdf_means[j]))
            cos_similarities.append(((i, j), cos_sim))

    # Store results for this chunk size
    results[chunk_size] = {
        "cosine": cos_similarities,
    }

    # Print summary for this chunk size
    for idx, ((i, j), sim) in enumerate(cos_similarities, start=1):
        print(f"Cosine Similarity {idx} ({pdf_files[i]} vs {pdf_files[j]}): {sim:.4f}")


# At this point, 'results' holds all metrics across chunk sizes.

# Create a descriptive filename
file_suffix = f"start{chunk_start}_step{chunk_step}_end{chunk_end}_overlap{overlap_percent}_model_{model_name}"
json_filename = f"results_{file_suffix}.json"
csv_filename = f"results_{file_suffix}.csv"

# Save JSON
with open(json_filename, "w") as f:
    json.dump(results, f, indent=4)

# Save CSV
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["chunk_size", "file1", "file2", "cosine"])
    for chunk_size, vals in results.items():
        for (i, j), sim in vals["cosine"]:
            writer.writerow([chunk_size, pdf_files[i], pdf_files[j], sim])