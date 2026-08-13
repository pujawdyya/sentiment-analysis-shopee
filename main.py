#Import library 
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)

print("✅ Semua library berhasil diimport!")

#Load dataset
df = pd.read_csv('shopee_reviews.csv')

print("=" * 60)
print("   DATASET ULASAN SHOPEE — GOOGLE PLAY STORE 2026")
print("=" * 60)
print(f"  Jumlah baris    : {df.shape[0]:,}")
print(f"  Jumlah kolom    : {df.shape[1]}")
print("=" * 60)

print("\n📋 Preview Data (5 baris pertama):")
print(df.head())

print(f"\ndf.shape")
print(f"({df.shape[0]}, {df.shape[1]})")

print("\n📋 Info Kolom:")
print(f"{'No':<5} {'Nama Kolom':<35} {'Tipe Data':<15} {'Non-Null'}")
print("-" * 70)
for i, col in enumerate(df.columns):
    print(f"{i:<5} {col:<35} {str(df[col].dtype):<15} {df[col].notna().sum():,}")

print("\n📋 Missing Values per Kolom:")
missing = df.isnull().sum()
missing_df = pd.DataFrame({
    'Kolom'        : missing.index,
    'Missing Count': missing.values,
    'Missing (%)'  : (missing.values / len(df) * 100).round(2)
})
print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0
        else pd.DataFrame({'Info': ['✓ Tidak ada missing values']}))

print("\n📋 Statistik Deskriptif (kolom numerik):")
print(df.describe().round(3))

print("\n📋 Distribusi Skor Bintang (Rating):")
score_dist = df['score'].value_counts().sort_index().reset_index()
score_dist.columns = ['Skor Bintang', 'Jumlah Ulasan']
score_dist['Persentase (%)'] = (score_dist['Jumlah Ulasan'] / len(df) * 100).round(2)
score_dist['Label Sentimen'] = score_dist['Skor Bintang'].map({
    1: 'Negatif', 2: 'Negatif', 3: 'Netral', 4: 'Positif', 5: 'Positif'
})
print(score_dist)

print(f"\n🗓️  Rentang waktu ulasan : {df['at'].min()} s/d {df['at'].max()}")


#Data Cleaning
kolom_pakai = ['reviewId', 'content', 'score', 'at']

# Kolom yang dihapus beserta alasannya
kolom_hapus = {
    'userName'             : 'Identitas personal, tidak relevan',
    'userImage'            : 'URL foto profil, tidak relevan',
    'thumbsUpCount'        : 'Jumlah like, tidak digunakan dalam klasifikasi',
    'reviewCreatedVersion' : 'Tidak relevan + 22.32% kosong',
    'replyContent'         : 'Balasan developer, bukan opini pengguna + 10.74% kosong',
    'repliedAt'            : 'Tanggal balasan developer + 10.74% kosong',
    'appVersion'           : 'Tidak relevan + 22.32% kosong',
}

print("Kolom yang dihapus:")
print(f"{'Nama Kolom':<30} {'Alasan'}")
print("-" * 75)
for kolom, alasan in kolom_hapus.items():
    print(f"  {kolom:<28} {alasan}")

df = df[kolom_pakai].copy()

print(f"\n✅ Data cleaning selesai!")
print(f"   Kolom sebelum : 11")
print(f"   Kolom sesudah : {df.shape[1]}")
print(f"   Kolom tersisa : {df.columns.tolist()}")
print(df.head())


#Pelabelan Sentimen
def label_sentimen(score):
    if score in [1, 2]:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

df['sentimen'] = df['score'].apply(label_sentimen)

print("✅ Pelabelan selesai!")
print()
print("Distribusi Label Sentimen:")
dist = df['sentimen'].value_counts()
for label, count in dist.items():
    bar = '█' * int(count / 100)
    print(f"  {label:<10}: {count:>5,}  ({count/len(df)*100:.1f}%)  {bar}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

colors_map = {'Positif': '#2ecc71', 'Negatif': '#e74c3c', 'Netral': '#f39c12'}
dist_ordered = df['sentimen'].value_counts()
bars = axes[0].bar(dist_ordered.index, dist_ordered.values,
                   color=[colors_map[l] for l in dist_ordered.index],
                   edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, dist_ordered.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val:,}', ha='center', fontweight='bold')
axes[0].set_title('Jumlah Ulasan per Kelas', fontweight='bold')
axes[0].set_ylabel('Jumlah')

sizes = [dist_ordered.get(l, 0) for l in ['Positif', 'Negatif', 'Netral']]
clrs  = ['#2ecc71', '#e74c3c', '#f39c12']
axes[1].pie(sizes, labels=['Positif', 'Negatif', 'Netral'],
            autopct='%1.1f%%', colors=clrs,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Proporsi Kelas Sentimen', fontweight='bold')

plt.suptitle('Distribusi Label Sentimen — Ulasan Shopee', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


#Inisialisasi Stemmer & Stopwords
print("Menginisialisasi stemmer PySastrawi... (tunggu ~10 detik)")

factory    = StemmerFactory()
stemmer    = factory.create_stemmer()

stop_words = set(stopwords.words('indonesian'))
tambahan_stopword = {
    'nya', 'yg', 'ga', 'gak', 'nggak', 'udah', 'udh', 'aja', 'bgt', 'banget',
    'sih', 'deh', 'nih', 'loh', 'dong', 'kan', 'tuh', 'kalo', 'kalau', 'tp',
    'tpi', 'tapi', 'jg', 'juga', 'lg', 'lagi', 'sm', 'sama', 'emg', 'emang',
    'bkn', 'bukan', 'krn', 'karna', 'karena', 'sdh', 'sudah', 'bs', 'bisa',
    'dr', 'dari', 'ke', 'di', 'dgn', 'dengan', 'utk', 'untuk', 'jd', 'jadi',
    'hrs', 'harus', 'msh', 'masih', 'blm', 'belum', 'ad', 'ada',
    'app', 'aplikasi', 'shopee', 'hp', 'handphone', 'android', 'play', 'store'
}
stop_words.update(tambahan_stopword)

print(f"✅ Stemmer siap. Total stopword: {len(stop_words)}")


#Preprocessing Teks
from tqdm import tqdm
tqdm.pandas()

def preprocess(text):
    """Pipeline preprocessing teks Bahasa Indonesia"""
    text   = str(text).lower()                          # 1. Lowercase
    text   = re.sub(r'http\S+|www\S+', '', text)      # 2. Hapus URL
    text   = re.sub(r'[^a-z\s]', '', text)             # 3. Hapus non-alfabet
    text   = re.sub(r'\s+', ' ', text).strip()         # 4. Hapus spasi berlebih
    tokens = text.split()                               # 5. Tokenisasi
    tokens = [w for w in tokens                         # 6. Hapus stopword
              if w not in stop_words and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens]          # 7. Stemming
    return ' '.join(tokens)

print("Memproses teks...")
df['content_clean'] = df['content'].progress_apply(preprocess)

df = df[df['content_clean'].str.strip() != ''].reset_index(drop=True)

print(f"\n✅ Preprocessing selesai!")
print(f"   Ulasan tersisa setelah preprocessing : {len(df):,}")
print(f"   Ulasan tereliminasi (teks kosong)    : {10000 - len(df):,}")
print()
print("Contoh hasil preprocessing:")
for i in [0, 5, 10]:
    row = df.iloc[i]
    print(f"[{row['sentimen']}]")
    print(f"  Asli   : {row['content'][:75]}")
    print(f"  Bersih : {row['content_clean'][:75]}")
    print()


#Ekstraksi Fitur TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,   # 5000 token dengan bobot TF-IDF tertinggi
    ngram_range=(1, 2),  # unigram + bigram
    min_df=2             # abaikan token yang hanya muncul di 1 dokumen
)

X = tfidf.fit_transform(df['content_clean'])
y = df['sentimen']

print(f"✅ TF-IDF selesai!")
print(f"   Dimensi matriks : {X.shape[0]:,} dokumen × {X.shape[1]:,} fitur")
print(f"   Contoh fitur    : {tfidf.get_feature_names_out()[:12].tolist()}")


#Pembagian Data Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # proporsi kelas sama di train & test
)

print("✅ Data berhasil dibagi (Stratified 80:20)")
print(f"   Data latih : {X_train.shape[0]:,} ulasan")
print(f"   Data uji   : {X_test.shape[0]:,} ulasan")
print()
print("Distribusi kelas di data uji:")
print(y_test.value_counts().to_string())


#Pelatihan Model Naive Bayes
model = MultinomialNB(alpha=1.0)   # alpha=1 → Laplace smoothing
model.fit(X_train, y_train)

print("✅ Model Multinomial Naive Bayes berhasil dilatih!")
print(f"   Kelas yang dipelajari : {model.classes_}")


#Evaluasi Model
y_pred  = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"{'='*50}")
print(f"  AKURASI MODEL : {akurasi * 100:.2f}%")
print(f"{'='*50}")
print()
print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Negatif', 'Netral', 'Positif'],
    digits=4
))


#Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Netral', 'Positif'])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Negatif', 'Netral', 'Positif'],
    yticklabels=['Negatif', 'Netral', 'Positif'],
    linewidths=0.5, linecolor='white', annot_kws={"size": 13}
)
plt.title('Confusion Matrix — Naive Bayes\nAnalisis Sentimen Shopee',
          fontweight='bold', fontsize=13)
plt.xlabel('Prediksi', fontweight='bold')
plt.ylabel('Aktual', fontweight='bold')
plt.tight_layout()
plt.show()


#Visualisasi Precision, Recall, F1 per Kelas
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred,
    labels=['Negatif', 'Netral', 'Positif'],
    zero_division=0
)

x      = np.arange(3)
width  = 0.25
labels = ['Negatif', 'Netral', 'Positif']

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
bars2 = ax.bar(x,         recall,    width, label='Recall',    color='#e67e22')
bars3 = ax.bar(x + width, f1,        width, label='F1-Score',  color='#9b59b6')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title('Precision / Recall / F1-Score per Kelas Sentimen',
             fontweight='bold', fontsize=13)
ax.set_ylabel('Nilai')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


#Word Cloud Positif & Negatif
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, label, cmap in zip(axes, ['Positif', 'Negatif'], ['Greens', 'Reds']):
    teks = ' '.join(df[df['sentimen'] == label]['content_clean'])
    wc   = WordCloud(
        width=700, height=350,
        background_color='white',
        colormap=cmap,
        max_words=100
    ).generate(teks)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud — Ulasan {label}', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()


#Top 15 Kata per Kelas
from collections import Counter

def top_kata(sentimen_label, n=15):
    teks    = ' '.join(df[df['sentimen'] == sentimen_label]['content_clean'])
    counter = Counter(teks.split())
    return pd.DataFrame(counter.most_common(n), columns=['Kata', 'Frekuensi'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, label, color in zip(axes, ['Positif', 'Negatif'], ['#2ecc71', '#e74c3c']):
    top = top_kata(label)
    ax.barh(top['Kata'][::-1], top['Frekuensi'][::-1],
            color=color, edgecolor='white', linewidth=0.8)
    for i, (val, nama) in enumerate(zip(top['Frekuensi'][::-1], top['Kata'][::-1])):
        ax.text(val + 5, i, str(val), va='center', fontsize=9)
    ax.set_title(f'Top 15 Kata — Ulasan {label}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Frekuensi')


plt.tight_layout()
plt.show()


#Simpan & Download Hasil (Opsional)
X_all = tfidf.transform(df['content_clean'])
df['prediksi']     = model.predict(X_all)
df['prob_positif'] = model.predict_proba(X_all)[:, list(model.classes_).index('Positif')]
df['prob_negatif'] = model.predict_proba(X_all)[:, list(model.classes_).index('Negatif')]
df['prob_netral']  = model.predict_proba(X_all)[:, list(model.classes_).index('Netral')]

output_cols = ['reviewId', 'content', 'score', 'sentimen',
               'content_clean', 'prediksi',
               'prob_positif', 'prob_negatif', 'prob_netral', 'at']
df[output_cols].to_csv('hasil_prediksi_shopee.csv', index=False)

print("✅ File berhasil diunduh: hasil_prediksi_shopee.csv")