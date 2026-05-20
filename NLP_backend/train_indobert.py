import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Konfigurasi
MAX_LEN = 128
BATCH_SIZE = 16
MODEL_NAME = "indobenchmark/indobert-base-p1"

# Di PyTorch, kita perlu membuat cetakan Dataset khusus
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

def run_indobert():
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
    
    # Membungkus data ke dalam format PyTorch
    train_dataset = BeritaDataset(train_encodings, y_train)
    test_dataset = BeritaDataset(test_encodings, y_test)
    
    print(f"\nMengunduh/Memuat Model {MODEL_NAME} (Versi PyTorch)...")
    # Perhatikan kita menggunakan AutoModelForSequenceClassification (Tanpa TF)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Trainer API dari HuggingFace membuat proses training sangat rapi tanpa perlu for-loop manual
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="no" # Agar tidak memakan memori harddisk saat training
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    print("\n🔥 MEMULAI TRAINING INDOBERT (Bisa ditinggal keluar)...")
    trainer.train()
    
    print("\n✅ Training Selesai! Memprediksi data uji...")
    # Prediksi menggunakan Trainer
    predictions = trainer.predict(test_dataset)
    y_pred_classes = np.argmax(predictions.predictions, axis=1)
    
    print("\n================ HASIL EVALUASI IndoBERT ================")
    print(classification_report(y_test, y_pred_classes, target_names=['FAKTA', 'HOAX'], digits=3))
    
    print("\n💾 Menyimpan model IndoBERT...")
    model.save_pretrained('saved_model_indobert')
    tokenizer.save_pretrained('saved_model_indobert')
    print("✅ Model IndoBERT Berhasil Disimpan!")

if __name__ == "__main__":
    run_indobert()