import json
import matplotlib.pyplot as plt

with open("results_start100_step100_end30000_overlap10_model_nomic_embed_text_v2_moe.json", "r") as f:
    results = json.load(f)

diffs = []
sim01 = []

for chunk_size, data in results.items():
    cosine_pairs = {tuple(pair): sim for pair, sim in data["cosine"]}
    
    if (0,1) in cosine_pairs and (0,2) in cosine_pairs:
        diff = cosine_pairs[(0,1)] - cosine_pairs[(0,2)]
        diffs.append((int(chunk_size), diff))
    
    if (0,1) in cosine_pairs:
        sim01.append((int(chunk_size), cosine_pairs[(0,1)]))

# Sort
diffs = sorted(diffs)
sim01 = sorted(sim01)

# 1. Grafik 
x_diff = [c for c, _ in diffs]
y_diff = [d for _, d in diffs]

plt.figure(figsize=(8,5))
plt.plot(x_diff, y_diff, marker="o")
plt.title("Difference: CosSim(0,1) - CosSim(0,2)")
plt.xlabel("Chunk Size")
plt.ylabel("Difference")
plt.grid(True)
plt.show(block=False)  

# 2. Grafik 
x_sim = [c for c, _ in sim01]
y_sim = [s for _, s in sim01]

plt.figure(figsize=(8,5))
plt.plot(x_sim, y_sim, marker="o", color="green")
plt.title("Cosine Similarity of Pair (0,1)")
plt.xlabel("Chunk Size")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.show(block=False) 

input("Press Enter to close all plots...")
