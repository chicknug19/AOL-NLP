import pandas as pd

# Load raw dataset
df_hoax = pd.read_csv('nlp_dataset/komdigi_hoaks.csv')
df_fakta = pd.read_csv('nlp_dataset/data_fakta.csv')

print("=== 3 DATA HOAKS ACAK ===")
# Mengambil 3 sampel acak dari kolom teks/body_text
for teks in df_hoax['body_text'].dropna().sample(3, random_state=None):
    print(f"- {teks}\n")

print("=== 3 DATA FAKTA ACAK ===")
for teks in df_fakta['content'].dropna().sample(3, random_state=None):
    print(f"- {teks}\n")