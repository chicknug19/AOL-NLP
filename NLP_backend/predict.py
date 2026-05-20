import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Menghilangkan warning terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['transformers_verbosity'] = 'error'

# Path ke folder tempat kamu menyimpan model tadi
MODEL_PATH = './saved_model_indobert'

def load_model():
    print("⏳ Membangunkan IndoBERT dari tidurnya (Memuat model)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

def predict_text(text, tokenizer, model):
    # Mengubah teks menjadi format yang dipahami IndoBERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Mematikan kalkulasi gradient (karena kita hanya menebak, bukan melatih)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Mengubah skor mentah (logits) menjadi persentase probabilitas 0-100%
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
    # Mengambil indeks dengan persentase tertinggi (0 = FAKTA, 1 = HOAKS)
    pred_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][pred_class].item() * 100
    
    return pred_class, confidence

def main():
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(f"❌ Error: Model tidak ditemukan di folder '{MODEL_PATH}'. Pastikan path-nya benar.")
        return

    print("\n========================================================")
    print("🤖 INDOBERT HOAX DETECTOR SIAP DIGUNAKAN!")
    print("Ketik 'keluar' atau 'exit' untuk menghentikan program.")
    print("========================================================")

    while True:
        teks_input = input("\n📝 Masukkan teks berita/pesan WA:\n> ")
        
        if teks_input.lower() in ['keluar', 'exit']:
            print("Sampai jumpa! 👋")
            break
            
        if not teks_input.strip():
            print("Teks tidak boleh kosong!")
            continue
            
        pred_class, confidence = predict_text(teks_input, tokenizer, model)
        
        # Labeling (Sesuai dengan kodingan training kita: 0=FAKTA, 1=HOAKS)
        label = "✅ FAKTA" if pred_class == 0 else "🚨 HOAKS"
        
        print("\n📊 HASIL ANALISIS INDOBERT:")
        print(f"Prediksi  : {label}")
        print(f"Keyakinan : {confidence:.2f}%")
        print("-" * 55)

if __name__ == "__main__":
    main()