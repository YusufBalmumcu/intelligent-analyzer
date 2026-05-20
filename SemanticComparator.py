import numpy as np
from embedder import Embedder
import models
from models import TASK_TYPE

class SemanticComparator:
    def __init__(self, contents: list[str], model: Embedder = None):
        if model is None:
            self.model = models.paraphrase_multilingual_MiniLM_L12_v2()
        else:
            self.model = model

        self.contents = contents
        self.max_std_multiplier = 1.0

        # 1. İçeriklerin vektörlerini hesapla
        print(f"   > {len(contents)} adet parça (chunk) vektörleştiriliyor...")
        self.embeddings = self.model.calculate(contents, task_type=TASK_TYPE.SEMANTIC_SIMILARITY)

        # 2. Vektörlerin merkezini (Centroid) bul
        self.centroid = self.model.reduce_mean(self.embeddings)

        # 3. Her bir vektörün merkeze olan COSINE uzaklığını hesapla
        # (Cosine Sim: 1 benzer, 0 benzemez. Distance: 0 yakın, 1 uzak)
        distances = []
        for vec in self.embeddings:
            sim = self.model.cosine_similarity(vec, self.centroid)
            dist = 1.0 - sim 
            distances.append(dist)

        # 4. İstatistikleri hesapla
        self.mean_distance = np.mean(distances)
        self.std_dev_distance = np.std(distances)

        print(f"   > Küme İstatistikleri: Ortalama Uzaklık={self.mean_distance:.4f}, Std Sapma={self.std_dev_distance:.4f}")

    def setMaxStd(self, multiplier: float) -> None:
        self.max_std_multiplier = multiplier
        print(f"   > Hassasiyet ayarlandı: {self.max_std_multiplier} sigma")

    def compare(self, content: str) -> bool:
        # 1. Hedefin vektörünü al
        target_embedding = self.model.calculate(content, task_type=TASK_TYPE.SEMANTIC_SIMILARITY)

        # 2. Merkeze olan Cosine Distance hesapla
        sim = self.model.cosine_similarity(target_embedding, self.centroid)
        distance = 1.0 - sim

        # 3. Eşik değerini hesapla
        threshold = self.mean_distance + (self.max_std_multiplier * self.std_dev_distance)

        is_accepted = distance <= threshold
        
        # Sonucu yazdır (Debug)
        status = "KABUL" if is_accepted else "RED"
        print(f"   [Analiz] Uzaklık: {distance:.4f} | Limit: {threshold:.4f} -> {status}")
        
        return is_accepted