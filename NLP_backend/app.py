from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from duckduckgo_search import DDGS
import os
import re
import nltk
from nltk.corpus import stopwords

# Mengunduh kamus NLTK saat server pertama kali menyala (Mencegah lag)
print("⏳ Mengunduh kamus Stopwords NLTK...")
nltk.download('stopwords', quiet=True)
NLTK_STOPWORDS = set(stopwords.words('indonesian'))
print("✅ Kamus NLTK Siap!")

# ... (Kodingan inisiasi FastAPI dan Model IndoBERT di bawahnya) ...

# Menghilangkan warning terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['transformers_verbosity'] = 'error'

# Inisiasi Aplikasi FastAPI
app = FastAPI(title="Hoax Detector API")

# Konfigurasi CORS agar React JS (frontend) bisa menembak API ini tanpa diblokir browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saat produksi nanti, ganti dengan URL React-mu (misal: http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memuat Model IndoBERT secara global (hanya diload 1x saat server menyala)
MODEL_PATH = './saved_model_indobert'
print("⏳ Menyalakan Mesin IndoBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print("✅ Mesin IndoBERT Siap!")

# Format struktur data yang akan diterima dari React JS
class BeritaRequest(BaseModel):
    teks: str

# Endpoint utama API
@app.post("/api/deteksi")
def deteksi_hoaks(request: BeritaRequest):
    input_teks = request.teks
    
    # ==========================================
    # 1. ANALISIS INDOBERT (Gaya Bahasa)
    # ==========================================
    inputs = tokenizer(input_teks, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
    pred_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][pred_class].item() * 100
    
    label = "FAKTA" if pred_class == 0 else "HOAKS"
    
    # ==========================================
    # 2. PENCARIAN REAL-TIME (Cek Fakta Internet)
    # ==========================================
    artikel_referensi = []
    try:
        # Gabungkan NLTK dengan kata-kata khas birokrasi politik yang sering lolos
        custom_words = {"secara", "baru", "dalam", "bersama", "menyusul", "adanya", "sebagai", "terkait", "keputusan", "telah"}
        stopwords_id = NLTK_STOPWORDS.union(custom_words)
        
        # Bersihkan tanda baca dan jadikan huruf kecil
        kata_bersih = re.findall(r'\b\w+\b', input_teks.lower())
        
        # Filter: Bukan stopword DAN panjang kata lebih dari 3 huruf
        kata_kunci_list = [w for w in kata_bersih if w not in stopwords_id and len(w) > 3]
        
        # Ambil 10 kata pertama
        kata_kunci = " ".join(kata_kunci_list[:10])
        
        print(f"🔍 Mencari di internet dengan kata kunci: '{kata_kunci}'")
        
        with DDGS() as ddgs:
            # safesearch='moderate' membantu membuang hasil web spam/random
            hasil_pencarian = ddgs.text(kata_kunci, max_results=3, region='id-id', safesearch='moderate')
            
            for hasil in hasil_pencarian:
                artikel_referensi.append({
                    "judul": hasil['title'],
                    "link": hasil['href'],
                    "cuplikan": hasil['body']
                })
    except Exception as e:
        print(f"Error pencarian: {e}")

    # ==========================================
    # 3. KEMBALIKAN RESPON JSON KE REACT JS
    # ==========================================
    return {
        "status": "success",
        "teks_asli": input_teks,
        "analisis_ai": {
            "prediksi": label,
            "keyakinan": round(confidence, 2),
            "keterangan": f"Gaya penulisan teks ini {round(confidence, 2)}% menyerupai pola {'berita jurnalistik resmi' if label == 'FAKTA' else 'pesan provokatif/manipulatif'}."
        },
        "cek_fakta_internet": artikel_referensi
    }
    
if __name__ == "__main__":
    import uvicorn
    # Menjalankan server di port 7860 yang diwajibkan oleh Hugging Face
    uvicorn.run(app, host="0.0.0.0", port=7860)
    
