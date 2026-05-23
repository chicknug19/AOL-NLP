import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# KONFIGURASI HYPERPARAMETER
# ==========================================
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128

def prepare_bilstm_data():
    print("Membaca data bersih...")
    df = pd.read_csv('nlp_dataset/data_bersih.csv')
    df = df.dropna(subset=['teks'])
    
    texts = df['teks'].astype(str).tolist()
    labels = df['label'].tolist()
    
    print("Melakukan Tokenization (Membangun kamus)...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    X_pad = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Split 80:20 dengan random_state=42 agar soal ujian 100% sama dengan SVM dan CNN
    X_train, X_test, y_train, y_test = train_test_split(X_pad, np.array(labels), test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, tokenizer

def build_bilstm_attention_model():
    input_layer = Input(shape=(MAX_LEN,))
    
    # 1. Layer Pemahaman Kata (Embedding)
    embed_layer = Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM)(input_layer)
    
    # 2. Layer Memori 2 Arah (BiLSTM)
    # return_sequences=True WAJIB agar setiap kata dikirim ke layer Attention
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embed_layer)
    
    # 3. Layer Mekanisme Perhatian (Self-Attention)
    # Model akan membandingkan kata dengan kata lainnya dalam satu kalimat untuk mencari konteks
    attention_out = Attention()([lstm_out, lstm_out])
    
    # 4. Pengumpulan Informasi Penting
    pooled_out = GlobalAveragePooling1D()(attention_out)
    
    # 5. Otak Pengambil Keputusan
    dense_1 = Dense(64, activation='relu')(pooled_out)
    dropout = Dropout(0.5)(dense_1) # Mencegah hafalan mati (overfitting)
    output_layer = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def run_bilstm():
    X_train, X_test, y_train, y_test, tokenizer = prepare_bilstm_data()
    
    print("\nMembangun arsitektur BiLSTM + Attention...")
    model = build_bilstm_attention_model()
    
    print("\nMemulai Training BiLSTM (Akan memakan waktu lebih lama dari CNN)...")
    # Epoch diset 3 agar cukup waktu untuk melihat pergerakan Loss
    model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split=0.1)
    
    print("\nTraining Selesai! Memprediksi data uji...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\n================ HASIL EVALUASI BiLSTM + ATTENTION ================")
    print(classification_report(y_test, y_pred_classes, target_names=['FAKTA', 'HOAX'], digits=3))
    
    print("\nMenyimpan model BiLSTM-Attention...")
    model.save('saved_model_bilstm_attention.keras')
    with open('tokenizer_bilstm_attention.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model BiLSTM-Attention Berhasil Disimpan!")

if __name__ == "__main__":
    run_bilstm()