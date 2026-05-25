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
from collections import Counter

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
    
    artikel_referensi = []
    try:
        # 1. TANGKAP AKRONIM (KTP, KPU, DPR) SEBAGAI PRIORITAS TERTINGGI
        akronim = re.findall(r'\b[A-Z]{2,}\b', input_teks)
        akronim_unik = list(dict.fromkeys([a.lower() for a in akronim]))
        
        # 2. BERSIHKAN KATA (Perhatikan: len >= 3 agar ktp, kpu tidak hilang)
        kata_bersih = re.findall(r'\b[a-z]{3,}\b', input_teks.lower())
        
        # 3. DAFTAR STOPWORDS (Buang kata gaul agar DDG tidak bingung)
        custom_words = {"yang", "dari", "pada", "untuk", "dengan", "dan", "atau", "bisa", "buat", "kalian", "jangan", "aja", "mau", "gak", "ada", "lagi", "itu", "ini", "kok", "tetep", "banget", "sama", "pas", "nanti", "cuma", "sampai", "satu", "kita", "hari", "karena", "masih", "supaya", "sebelum", "dalam", "secara", "telah", "lalu", "akan", "juga", "jadi", "kami", "mereka", "banyak", "semua", "saat", "guys", "nya", "buat"}
        stopwords_id = NLTK_STOPWORDS.union(custom_words)
        
        # 4. AMBIL KATA PENTING SESUAI URUTAN MUNCUL (Bukan Abjad/Frekuensi)
        kata_penting_urut = []
        for w in kata_bersih:
            if w not in stopwords_id and w not in akronim_unik:
                if w not in kata_penting_urut: # Hindari duplikat
                    kata_penting_urut.append(w)
                    
        # 5. GABUNGKAN & POTONG (Akronim + Kata Awal)
        # Ambil 8 kata agar konteksnya panjang dan jelas
        kata_final_list = akronim_unik + kata_penting_urut
        kata_kunci = " ".join(kata_final_list[:8]) + " berita indonesia"
        
        print(f"🔍 [DEBUG] Kata kunci DDG: '{kata_kunci}'")
        
        with DDGS() as ddgs:
            # Cari berita spesifik Indonesia
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
    
