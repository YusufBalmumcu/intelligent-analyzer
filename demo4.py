import os
import models
import utils
from SemanticComparator import SemanticComparator

def main():
    # ----------------------------
    # 1. Hazırlık ve Model Yükleme
    # ----------------------------
    print("Model yükleniyor...")
    model = models.paraphrase_multilingual_MiniLM_L12_v2()
    model.max_seq_length = 500

    pdf_folder = "pdfs1"

    if os.path.exists(pdf_folder):
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        pdf_files.sort()
    else:
        print(f"Hata: '{pdf_folder}' klasörü bulunamadı.")
        return

    print(f"Bulunan referans dosyalar ({len(pdf_files)} adet): {pdf_files}")

    # -----------------------------------------
    # 2. Referans Verisini Hazırlama (Chunking)
    # -----------------------------------------
    reference_contents = [] 

    print("\nReferans dokümanlar okunuyor ve parçalanıyor...")
    for fname in pdf_files:
        file_path = os.path.join(pdf_folder, fname)
        
        # PDF'i metne çevir
        raw_text = utils.pdf_to_text(file_path)
        
        if raw_text and raw_text.strip():
            # BURASI DÜZELTİLDİ: Senin istediğin format
            # Metni 2000 karakterlik parçalara, 200 karakter örtüşmeli (overlap) bölüyoruz.
            file_chunks = utils.split_document_by_structure(raw_text, 2000, 200)
            
            # Bu parçaları ana listeye ekle
            reference_contents.extend(file_chunks)

    if not reference_contents:
        print("Hata: İşlenecek metin bulunamadı.")
        return

    print(f"Toplam {len(reference_contents)} adet metin parçası referans alındı.")

    # -------------------------------
    # 3. Semantic Comparator Başlatma
    # -------------------------------
    print("\nSemantic Comparator başlatılıyor...")
    
    # Chunk'ları vererek modeli eğitiyoruz (Merkez ve Sapma hesabı)
    comparator = SemanticComparator(reference_contents, model=model)
    
    # Hassasiyet Ayarı:
    # 2000 karakterlik bloklar büyük olduğu için varyasyon az olabilir.
    comparator.setMaxStd(1.5) 

    # -------------------
    # 4. Test Senaryoları
    # -------------------
    print("\n" + "="*60)
    print(f"--- ANOMALİ / UYGUNLUK TESTİ ---")
    print("="*60)
    
    test_queries = [
        ("Konuşma tanıma sistemlerinde gürültü azaltma ve sinyal işleme teknikleri.", "KABUL (Teknik)"),
        ("Akşama ne pişirsem diye düşünüyorum, makarna güzel olur.", "RED (Yemek)"),
        ("Fenerbahçe bu sene şampiyonluğun en büyük adayı.", "RED (Spor)"),
        ("Derin öğrenme modellerinde backpropagation algoritmasının kullanımı.", "KABUL (Teknik)"),
        ("Merkez bankası faiz kararlarını açıkladı, dolar yükseldi.", "RED (Ekonomi)"),
    ]

    for text, label in test_queries:
        print(f"\nSorgu: '{text}'")
        print(f"Beklenen: {label}")
        
        result = comparator.compare(text)
        
        if result:
            print(">> SONUÇ: ✔️ KABUL EDİLDİ")
        else:
            print(">> SONUÇ: ❌ REDDEDİLDİ")

if __name__ == "__main__":
    main()