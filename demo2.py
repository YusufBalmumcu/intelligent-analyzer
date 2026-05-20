import os
import models
import utils  

def main():
    # Modeli başlat
    model = models.paraphrase_multilingual_MiniLM_L12_v2()

    # 1. PDF'lerin bulunduğu klasör yolu
    pdf_folder = "pdfs1"

    # 2. Klasördeki dosyaları otomatik çek, sadece .pdf olanları al ve sırala
    if os.path.exists(pdf_folder):
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        pdf_files.sort() # Çıktıların her seferinde aynı sırada olması için alfabetik sırala
    else:
        print(f"Hata: '{pdf_folder}' klasörü bulunamadı.")
        return

    if not pdf_files:
        print(f"Uyarı: '{pdf_folder}' içinde hiç PDF dosyası bulunamadı.")
        return

    print(f"Bulunan dosyalar ({len(pdf_files)} adet): {pdf_files}")

    # Compute mean embeddings for each PDF
    pdf_means = [
        model.reduce_mean(
            model.calculate(
                utils.split_document_by_structure(
                    utils.pdf_to_text(os.path.join(pdf_folder, fname)), 2000, 200
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
    print("\n--- Cosine Similarities ---")
    for idx, ((i, j), sim) in enumerate(cos_similarities, start=1):
        print(f"{idx}. ({pdf_files[i]} vs {pdf_files[j]}): {sim:.4f}")

    # Print Euclidean Distances
    print("\n--- Euclidean Distances ---")
    for idx, ((i, j), dist) in enumerate(euclidean_distances, start=1):
        print(f"{idx}. ({pdf_files[i]} vs {pdf_files[j]}): {dist:.4f}")
    

if __name__ == "__main__":
    main()