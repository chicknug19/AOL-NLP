import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Konfigurasi - Dibuat sama persis dengan IndoBERT untuk perbandingan yang adil
MAX_LEN = 128
BATCH_SIZE = 16
MODEL_NAME = "xlm-roberta-base" # Menggunakan Transformer Global Multilingual

class BeritaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def run_xlm_roberta():
    print("Membaca data bersih...")
    df = pd.read_csv('nlp_dataset/data_bersih.csv')
    df = df.dropna(subset=['teks'])
    
    texts = df['teks'].astype(str).tolist()
    labels = df['label'].tolist()
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"Mengunduh/Memuat Tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Melakukan Tokenisasi (Mohon tunggu)...")
    train_encodings = tokenizer(X_train_text, truncation=True, padding='max_length', max_length=MAX_LEN)
    test_encodings = tokenizer(X_test_text, truncation=True, padding='max_length', max_length=MAX_LEN)
    
    train_dataset = BeritaDataset(train_encodings, y_train)
    test_dataset = BeritaDataset(test_encodings, y_test)
    
    print(f"\nMengunduh/Memuat Model {MODEL_NAME} (Versi PyTorch)...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Pengaturan hiperparameter dikunci sama dengan IndoBERT V2 agar adil
    training_args = TrainingArguments(
        output_dir='./results_xlmr',
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir='./logs_xlmr',
        logging_steps=100,
        save_strategy="no" 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    print(f"\n🔥 MEMULAI TRAINING {MODEL_NAME.upper()}...")
    trainer.train()
    
    print("\n✅ Training Selesai! Memprediksi data uji...")
    predictions = trainer.predict(test_dataset)
    y_pred_classes = np.argmax(predictions.predictions, axis=1)
    
    print(f"\n================ HASIL EVALUASI {MODEL_NAME.upper()} ================")
    print(classification_report(y_test, y_pred_classes, target_names=['FAKTA', 'HOAX'], digits=3))
    
    # Kita tidak perlu menyimpan model ini karena tidak akan dinaikkan ke production
    print("✅ Eksperimen XLM-RoBERTa Selesai. Catat skor F1 dan Accuracy-nya untuk tabel PPT-mu!")

if __name__ == "__main__":
    run_xlm_roberta()