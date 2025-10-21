import importlib.util, os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
embedder_path = os.path.join(root_dir, "embedder.py")
spec = importlib.util.spec_from_file_location("embedder", embedder_path)
embedder = importlib.util.module_from_spec(spec)
sys.modules["embedder"] = embedder
spec.loader.exec_module(embedder)

import models
import utils  

def main():
    model = models.paraphrase_multilingual_MiniLM_L12_v2()

    pdf_files = [
        "Fatemi2019 - Knowledge Hypergraphs_ Prediction beyond Binary Relations.pdf",
        "Goel2019 - Diachronic Embedding for Temporal Knowledge Graph Completion.pdf",
        "Kazemi2018 - SimplE Embedding for Link Prediction in Knowledge Graphs.pdf",
        "Lacroix2020 - Tensor Decompositions for Temporal Knowledge Base Completion.pdf",
        "Li2021 - Search from History and Reason for Future_ Two Stage Reasoning on Temporal Knowledge Graphs.pdf",
        "Messner2021 - Temporal Knowledge Graph Completion Using Box Embeddings.pdf",
        "Omran2018 - Scalable Rule Learning Via Learning Representation.pdf",
        "Ray2021 - Timestamping Documents and Beliefs.pdf",
        "Sadeghian2021 - ChronoR_ Rotation Based Temporal Knowledge Graph Embedding.pdf",
        "Sun2021 - TimeTraveler_ Reinforcement Learning for Temporal Knowledge Graph Forecasting.pdf",
        "Quantum_sim.pdf",
    ]

    # Compute mean embeddings for each PDF
    pdf_means = [
        model.reduce_mean(
            model.calculate(
                utils.split_document_by_structure(
                    utils.pdf_to_text(f"pdfs2/{fname}"), 2000, 200
                ),
                task_type=models.TASK_TYPE.SEMANTIC_SIMILARITY
            )
        )
        for fname in pdf_files
    ]

    # Compute similarities and distances
    cos_similarities = []
    euclidean_distances = []

    
    for i in range(len(pdf_means)):
        for j in range(i + 1, len(pdf_means)):
            cos_sim = model.cosine_similarity(pdf_means[i], pdf_means[j])
            euc_dist = model.euclidean_distance(pdf_means[i], pdf_means[j])
            cos_similarities.append(((i, j), cos_sim))
            euclidean_distances.append(((i, j), euc_dist))

    # Print Cosine Similarities
    for idx, ((i, j), sim) in enumerate(cos_similarities, start=1):
        print(f"Cosine Similarity {idx} ({pdf_files[i]} vs {pdf_files[j]}): {sim:.4f}")

    # Print Euclidean Distances
    for idx, ((i, j), dist) in enumerate(euclidean_distances, start=1):
        print(f"Euclidean Distance {idx} ({pdf_files[i]} vs {pdf_files[j]}): {dist:.4f}")
    

if __name__ == "__main__":
    main()