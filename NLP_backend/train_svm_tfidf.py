import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

def prepare_data():
    print("Membaca data bersih...")
    # Pastikan nama file sesuai dengan yang ada di folder nlp_dataset milikmu
    df = pd.read_csv('nlp_dataset/data_bersih.csv')
    df = df.dropna(subset=['teks'])
    
    texts = df['teks'].astype(str).tolist()
    labels = df['label'].tolist()
    
    # Membagi data latih dan uji SEBELUM proses TF-IDF untuk mencegah kebocoran data (Data Leakage)
    print("Membagi porsi data latih dan data uji...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    return X_train_text, X_test_text, y_train, y_test

def show_top_words(vectorizer, svm_model, n=20):
    """
    Fungsi khusus (Explainable AI) untuk menampilkan kata-kata 
    yang paling kuat memicu indikasi HOAKS atau FAKTA.
    """
    words = vectorizer.get_feature_names_out()
    # LinearSVC memiliki atribut 'coef_' yang menyimpan bobot matematika dari setiap kata
    coef = svm_model.coef_[0]
    
    # Mengurutkan bobot: Positif tinggi = HOAKS, Negatif tinggi = FAKTA
    hoax_indices = np.argsort(coef)[-n:]
    fakta_indices = np.argsort(coef)[:n]
    
    print("\n" + "="*50)
    print("DETEKTIF KATA: KATA PEMICU HOAKS TERKUAT")
    print("="*50)
    hoax_words = [words[i] for i in reversed(hoax_indices)]
    print(", ".join(hoax_words))
    
    print("\n" + "="*50)
    print("DETEKTIF KATA: KATA INDIKATOR FAKTA TERKUAT")
    print("="*50)
    fakta_words = [words[i] for i in fakta_indices]
    print(", ".join(fakta_words))
    print("="*50 + "\n")

def run_svm():
    X_train_text, X_test_text, y_train, y_test = prepare_data()
    
    print("Melakukan Ekstraksi Fitur (TF-IDF)...")
    # TF-IDF mengubah kata menjadi angka pecahan berdasarkan seberapa langka kata tersebut
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    print("\nMemulai Training Model SVM (Sangat cepat, mohon tunggu)...")
    # Menggunakan LinearSVC karena sangat efisien untuk dataset teks yang besar
    model = LinearSVC(random_state=42, dual=False)
    model.fit(X_train, y_train)
    
    print("✅ Training Selesai! Memprediksi data uji...")
    y_pred = model.predict(X_test)
    
    print("\n================ HASIL EVALUASI SVM + TF-IDF ================")
    # Anggap label 0 = FAKTA, label 1 = HOAKS. Sesuaikan jika datasetmu terbalik.
    print(classification_report(y_test, y_pred, target_names=['FAKTA', 'HOAX'], digits=3))
    
    # Tampilkan senjata rahasia kita: Kata-kata biang kerok
    show_top_words(vectorizer, model)
    
    print("Menyimpan model SVM dan Vectorizer...")
    joblib.dump(model, 'saved_model_svm_tfidf.pkl')
    joblib.dump(vectorizer, 'saved_vectorizer_svm_tfidf.pkl')
    print("Model SVM Berhasil Disimpan!")

if __name__ == "__main__":
    run_svm()