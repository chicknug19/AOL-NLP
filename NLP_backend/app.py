from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from duckduckgo_search import DDGS
import os
import re

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
        # Daftar kata sambung bahasa Indonesia yang harus dibuang
        stopwords_id = {"dan", "yang", "di", "ke", "dari", "pada", "ini", "itu", "untuk", "dengan", "adalah", "bahwa", "mengonfirmasi", "adanya", "sebagai", "menyebutkan", "terkait"}
        
        # Bersihkan tanda baca dan jadikan huruf kecil
        kata_bersih = re.findall(r'\b\w+\b', input_teks.lower())
        
        # Ambil kata yang bermakna saja (bukan stopword), batasi 6 kata pertama
        kata_kunci_list = [w for w in kata_bersih if w not in stopwords_id][:6]
        kata_kunci = " ".join(kata_kunci_list)
        
        print(f"🔍 Mencari di internet dengan kata kunci: '{kata_kunci}'") # Untuk ngecek di terminal
        
        # Mencari artikel teratas
        with DDGS() as ddgs:
            hasil_pencarian = ddgs.text(kata_kunci, max_results=3, region='id-id')
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