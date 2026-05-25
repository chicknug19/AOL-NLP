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
    
    # ==========================================
    # 2. PENCARIAN REAL-TIME (Cek Fakta Internet)
    # ==========================================
    artikel_referensi = []
    try:
        # 1. TANGKAP AKRONIM (KTP, KPU, PPS dll) SEBAGAI PRIORITAS MUTLAK
        akronim = re.findall(r'\b[A-Z]{2,}\b', input_teks)
        akronim_unik = list(dict.fromkeys(akronim)) # Pertahankan huruf besar
        
        # 2. BERSIHKAN KATA (Ambil huruf kecil, panjang minimal 4 huruf)
        kata_bersih = re.findall(r'\b[a-z]{4,}\b', input_teks.lower())
        
        # 3. DAFTAR STOPWORDS (Buang semua kata gaul perusak pencarian)
        custom_words = {"yang", "dari", "pada", "untuk", "dengan", "atau", "bisa", "buat", "kalian", "jangan", "mau", "gak", "ada", "lagi", "itu", "ini", "kok", "tetep", "banget", "sama", "pas", "nanti", "cuma", "sampai", "satu", "kita", "hari", "karena", "masih", "supaya", "sebelum", "dalam", "secara", "telah", "lalu", "akan", "juga", "jadi", "kami", "mereka", "banyak", "semua", "saat", "guys", "nya", "aja", "kalo", "udah", "dong", "sih"}
        stopwords_id = NLTK_STOPWORDS.union(custom_words)
        
        kata_penting = [w for w in kata_bersih if w not in stopwords_id]
        
        # 4. AMBIL 2 KATA TERPANJANG SAJA
        kata_unik_terurut = sorted(list(set(kata_penting)), key=len, reverse=True)
        kata_terpanjang = kata_unik_terurut[:2] 
        
        # 5. RAKIT KATA KUNCI LASER
        akronim_utama = akronim_unik[:1] if akronim_unik else []
        kata_final_list = akronim_utama + kata_terpanjang
        kata_kunci = " ".join(kata_final_list) + " pemilu indonesia"
        
        print(f"🔍 [DEBUG] Kata kunci DDG: '{kata_kunci}'")
        
        with DDGS() as ddgs:
            # Kita tambahkan backend='html' agar DDG mengira server ini adalah browser jadul (membantu tembus blokir)
            hasil_pencarian = ddgs.text(kata_kunci, max_results=5, region='id-id', backend='html')
            
            for hasil in hasil_pencarian:
                teks_hasil = (hasil.get('title', '') + " " + hasil.get('body', '')).lower()
                syarat_akronim = any(re.search(rf"\b{a.lower()}\b", teks_hasil) for a in akronim_unik)
                syarat_lulus = "pemilu" in teks_hasil or "indonesia" in teks_hasil or syarat_akronim
                
                if syarat_lulus:
                    artikel_referensi.append({
                        "judul": hasil.get('title', 'Tanpa Judul'),
                        "link": hasil.get('href', ''),
                        "cuplikan": hasil.get('body', '')
                    })
                
                if len(artikel_referensi) >= 3:
                    break
                    
        # ==========================================
        # 🚨 JALUR CADANGAN MUTLAK (FAIL-SAFE API)
        # Jika DuckDuckGo diblokir, otomatis pakai Wikipedia Indonesia
        # ==========================================
        if len(artikel_referensi) == 0:
            print("⚠️ DuckDuckGo diblokir Hugging Face. Mengaktifkan Wikipedia API...")
            import requests
            
            # Merakit ulang kata kunci yang lebih ramah untuk Wikipedia
            kata_kunci_wiki = " ".join(kata_final_list) + " pemilu"
            wiki_url = f"https://id.wikipedia.org/w/api.php?action=query&list=search&srsearch={kata_kunci_wiki}&utf8=&format=json"
            
            headers = {'User-Agent': 'SmartHoaxDetector/1.0 (Binus University Project)'}
            resp = requests.get(wiki_url, headers=headers)
            
            if resp.status_code == 200:
                wiki_data = resp.json()
                # Ambil 3 hasil teratas dari Wikipedia
                for item in wiki_data.get('query', {}).get('search', [])[:3]:
                    # Bersihkan tag HTML <span> yang bawaan dari Wikipedia
                    snippet_bersih = re.sub(r'<[^>]+>', '', item['snippet'])
                    artikel_referensi.append({
                        "judul": item['title'] + " - Wikipedia Ensiklopedia Bebas",
                        "link": f"https://id.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                        "cuplikan": snippet_bersih + "..."
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
    
