import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle
import os

# Menyembunyikan peringatan TensorFlow agar terminal rapi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# KONFIGURASI HYPERPARAMETER
# ==========================================
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128

def prepare_cnn_data():
    print("Membaca data bersih...")
    df = pd.read_csv('nlp_dataset/data_bersih.csv') # Pastikan nama file sesuai
    df = df.dropna(subset=['teks'])
    
    texts = df['teks'].astype(str).tolist()
    labels = df['label'].tolist()
    
    print("Melakukan Tokenization (Membangun kamus 10.000 kata terbanyak)...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    print("Melakukan Padding (Menyeragamkan panjang menjadi 200 kata)...")
    X_pad = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Membagi porsi 80% latih, 20% uji (random_state=42 WAJIB agar soal ujian sama persis dengan SVM)
    X_train, X_test, y_train, y_test = train_test_split(X_pad, np.array(labels), test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tokenizer

def build_cnn_model():
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        
        # Jantung utamanya: Layer Conv1D sebagai "Pemindai Frasa" (melihat 5 kata sekaligus)
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        
        # Mengambil sinyal paling kuat dari frasa yang terdeteksi
        GlobalMaxPooling1D(), 
        
        Dense(64, activation='relu'),
        Dropout(0.5), # Mencegah overfitting agar tidak sekadar menghafal template
        Dense(1, activation='sigmoid') # 1 output: mendekati 0 = Fakta, mendekati 1 = Hoaks
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_cnn():
    X_train, X_test, y_train, y_test, tokenizer = prepare_cnn_data()
    
    print("\nMembangun arsitektur 1D-CNN...")
    model = build_cnn_model()
    
    # 3 Epoch sudah cukup untuk CNN karena ia cepat belajar dari frasa lokal
    print("\nMemulai Training (Memproses 48.000 data)...")
    model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.1)
    
    print("\nTraining Selesai! Memprediksi data uji...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\n================ HASIL EVALUASI 1D-CNN ================")
    # Target nama diurutkan: 0 = FAKTA, 1 = HOAX
    print(classification_report(y_test, y_pred_classes, target_names=['FAKTA', 'HOAX'], digits=3))
    
    print("\nMenyimpan model 1D-CNN...")
    model.save('saved_model_cnn.keras')
    with open('tokenizer_cnn.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model 1D-CNN Berhasil Disimpan!")

if __name__ == "__main__":
    run_cnn()