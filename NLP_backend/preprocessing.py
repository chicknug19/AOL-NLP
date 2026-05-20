import pandas as pd
import os
import re

def clean_text(text):
    """Fungsi ekstraksi narasi murni & pembuang sampah web scraper"""
    if not isinstance(text, str):
        return ""
    
    # =================================================================
    # 1. PISAU BEDAH DATA HOAKS (TURNBACKHOAX)
    # Target: Hanya mengambil teks di antara [NARASI] dan [PENJELASAN]
    # =================================================================
    # Catatan: Kita pakai regex yang menoleransi typo seperti [NARASI} 
    narasi_match = re.search(r'\[NARASI[\]\}][\s:]*([\s\S]*?)(?:\[PENJELASAN\]|\[REFERENSI\])', text, re.IGNORECASE)
    
    if narasi_match:
        # Jika ketemu format TurnBackHoax, kita BUANG seluruh laporannya, 
        # dan HANYA ambil narasi pembuat hoaks aslinya!
        text = narasi_match.group(1)
    else:
        # Jika formatnya beda, kita buang jejak laporannya secara manual
        text = re.sub(r'(?i)Hasil Periksa Fakta[\s\S]*?Faktanya,.*?(?=\n|$)', '', text)
        text = re.sub(r'(?i)\[KATEGORI\]:?.*?(\n|$)', '', text)
        text = re.sub(r'(?i)\[SUMBER\]:?[\s\S]*?(?=\n\n|\Z)', '', text)
        text = re.sub(r'(?i)\[PENJELASAN\]:?[\s\S]*', '', text) # Buang dari penjelasan sampai akhir

    # =================================================================
    # 2. PISAU BEDAH DATA FAKTA (CNN, KOMPAS, TEMPO)
    # =================================================================
    # Buang watermark lokasi dan media di awal paragraf
    text = re.sub(r'(?i)^[a-zA-Z\s]+,\s*(CNN Indonesia|KOMPAS\.com|TEMPO\.CO|INFO NASIONAL)\s*(--|-)?\s*', '', text)
    
    # Buang sampah iklan (Ads) dan sisipan web scraper
    text = re.sub(r'(?i)ADVERTISEMENT\s*SCROLL TO RESUME CONTENT', '', text)
    text = re.sub(r'(?i)\[Gambas:.*?\]', '', text) # Hapus [Gambas:Video CNN] dll
    
    # Buang rekomendasi artikel di tengah/akhir berita
    text = re.sub(r'(?i)(Lihat|Baca) Juga\s*:.*?(\n|$)', '', text)
    text = re.sub(r'(?i)Pilihan Editor\s*:.*', '', text)
    text = re.sub(r'(?i)Simak selengkapnya dalam video berikut[\s\S]*', '', text)
    
    # Buang jejak kaki jurnalis di akhir artikel
    text = re.sub(r'(?i)(Penulis|Editor|Reporter|Video Jurnalis)\s*:?[\s\S]*', '', text)
    text = re.sub(r'\([a-z]{2,4}/[a-z]{2,4}\)\s*$', '', text) # Hapus (yoa/kid)
    # MENGHAPUS SEMUA NAMA MEDIA DAN SUMBER
    # Memaksa AI buta merek dan hanya fokus pada narasi
    media_blacklist = [
        r'\bkompas\s*\.?\s*com\b', r'\btempo\s*\.?\s*co\b', r'\bcnn\s*indonesia\b',
        r'\bdetik\s*\.?\s*com\b', r'\bantaranews\b', r'\btribunnews\b',
        r'\bliputan6\b', r'\bsuara\s*\.?\s*com\b', r'\bturnbackhoax\b',
        r'\bjakarta\s*kompas\b'
    ]
    
    # =================================================================
    # THE ULTIMATE BLACKLIST: Menghapus Sisa Template & Kebiasaan Jurnalistik
    # =================================================================
    # =================================================================
    # THE ULTIMATE BLACKLIST V2 (BRUTAL MODE)
    # =================================================================
    # =================================================================
    # THE ULTIMATE BLACKLIST V3 (GOD MODE)
    # =================================================================
    word_blacklist = [
        # Sisa pecahan URL & Facebook
        r'\bfafhh\b', r'\bpermalink\b', r'\bbit\b', r'\bly\b', 
        r'\bfacebook\b', r'\bgroups?\b', r'\bid\b', r'\bcom\b', r'\bco\b', r'\bhtml\b',
        
        # Sisa template & analisis TurnBackHoax
        r'\breferensi\b', r'\bsumber\b', r'\bpenjelasan\b', r'\bklarifikasi\b', 
        r'\bkategori\b', r'\bnarasi\b', r'\bselengkapnya\b', r'\bread\b',
        r'\bhoax\b', r'\bhoaks\b', r'\bfoto\b', r'\bvideo\b',
        r'\bkonten\b', r'\bklaim\b', r'\bakun\b', r'\bartikel\b', r'\bpost\b', 
        r'\bmenyesatkan\b', r'\bberedar\b', r'\bdaring\b', r'\bbantah\b', r'\bfakta\b', 
        r'\bmedia\b', r'\bpostingan\b', r'\bberita\b', r'\byg\b', r'\bbahwa\b', r'\bbagian\b',
        
        # Nama hari
        r'\bsenin\b', r'\bselasa\b', r'\brabu\b', r'\bkamis\b', 
        r'\bjumat\b', r'\bsabtu\b', r'\bminggu\b',
        
        # Standalone nama media & kota rilis
        r'\btempo\b', r'\bkompas\b', r'\bcnn\b', r'\btribun\b', r'\bjakarta\b',
        
        # Kata ganti & penunjuk kutipan jurnalistik
        r'\bia\b', r'\bdia\b', r'\bmenurutnya\b', r'\bmengatakan\b', 
        r'\bmenyebut\b', r'\bucap\b', r'\bmenilai\b', r'\bmerespons\b', r'\bkata\b',
        r'\bmengaku\b', r'\bmeminta\b', r'\bmenurut\b', r'\bketerangan\b', r'\bpekan\b'
    ]
    
    for word in word_blacklist:
        text = re.sub(word, ' ', text, flags=re.IGNORECASE)
    
    for media in media_blacklist:
        text = re.sub(media, ' ', text, flags=re.IGNORECASE)

    # =================================================================
    # 3. PEMBERSIHAN KARAKTER (NLP BASIC)
    # =================================================================
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_preprocessing():
    print("=========================================================")
    print("🚀 MEMULAI PROSES PREPROCESSING DATASET RAW EXCEL")
    print("=========================================================")
    
    folder = 'nlp_dataset/'
    
    # Pendefinisian nama file Excel (Asumsi ekstensi .xlsx)
    file_cnn = os.path.join(folder, 'dataset_cnn_10k.xlsx')
    file_kompas = os.path.join(folder, 'dataset_kompas_4k.xlsx')
    file_tempo = os.path.join(folder, 'dataset_tempo_6k.xlsx')
    file_tbh = os.path.join(folder, 'dataset_turnbackhoax_10k.xlsx')
    
    # Cek ketersediaan file
    for f in [file_cnn, file_kompas, file_tempo, file_tbh]:
        if not os.path.exists(f):
            print(f"❌ Error: File {f} tidak ditemukan di folder nlp_dataset/!")
            return

    print("📖 Membaca file Excel menggunakan Pandas (Mohon tunggu)...")
    df_cnn = pd.read_excel(file_cnn)
    df_kompas = pd.read_excel(file_kompas)
    df_tempo = pd.read_excel(file_tempo)
    df_tbh = pd.read_excel(file_tbh)
    
    # Memberikan Label Konstanta
    # 0 = FAKTA, 1 = HOAKS
    df_cnn['label'] = 0
    df_kompas['label'] = 0
    df_tempo['label'] = 0
    df_tbh['label'] = 1
    
    # Menggabungkan seluruh populasi FAKTA
    print("🔗 Menggabungkan dataset Fakta (CNN + Kompas + Tempo)...")
    df_fakta_all = pd.concat([
        df_cnn[['FullText', 'label']], 
        df_kompas[['FullText', 'label']], 
        df_tempo[['FullText', 'label']]
    ], ignore_index=True)
    
    df_hoaks_all = df_tbh[['FullText', 'label']]
    
    # STRATEGI PENYEIMBANGAN DATA (Downsampling Fakta agar seimbang dengan Hoaks)
    total_hoaks = len(df_hoaks_all)
    print(f"⚖️ Menyeimbangkan Kelas: Mengambil {total_hoaks} sampel acak dari {len(df_fakta_all)} data Fakta...")
    df_fakta_sampled = df_fakta_all.sample(n=total_hoaks, random_state=42).reset_index(drop=True)
    
    # Penggabungan Akhir Fakta & Hoaks
    df_gabungan = pd.concat([df_fakta_sampled, df_hoaks_all], ignore_index=True)
    
    # Standarisasi nama kolom menjadi 'teks' agar klop dengan kodingan model lamamu
    df_gabungan.columns = ['teks', 'label']
    
    # Mengacak urutan baris data
    df_gabungan = df_gabungan.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("🧼 Menjalankan fungsi pembersihan teks murni (Anti-Leakage)...")
    df_gabungan['teks'] = df_gabungan['teks'].apply(clean_text)
    
    # Membuang baris yang kosong setelah dibersihkan
    df_gabungan = df_gabungan[df_gabungan['teks'] != ""]
    
    # Menyimpan menjadi file CSV tunggal yang bersih
    path_output = os.path.join(folder, 'data_clean.csv') 
    path_output = 'nlp_dataset/data_bersih.csv' # Nama disamakan agar model lain otomatis membaca
    df_gabungan.to_csv(path_output, index=False)
    
    print("\n=========================================================")
    print(f"✅ BERHASIL! Dataset baru disimpan di: {path_output}")
    print("=========================================================")
    print("\n📊 Distribusi Label Baru (50:50 Sempurna):")
    print(df_gabungan['label'].value_counts())
    print("\n👀 5 Baris Pertama Teks Mentah yang Sudah Bersih:")
    print(df_gabungan.head())

if __name__ == "__main__":
    run_preprocessing()