# Analisis Sentimen Pengguna terhadap Aplikasi Shopee

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi analisis sentimen terhadap ulasan pengguna aplikasi Shopee di Google Play Store.

Analisis dilakukan untuk mengetahui sentimen pengguna terhadap aplikasi Shopee berdasarkan ulasan yang diberikan. Sentimen diklasifikasikan menjadi tiga kategori, yaitu **Positif, Netral, dan Negatif**.

Metode klasifikasi yang digunakan dalam proyek ini adalah **Naive Bayes**, dengan pembobotan teks menggunakan **TF-IDF**.

## 🎯 Tujuan

Tujuan dari proyek ini adalah:

- Menganalisis sentimen pengguna terhadap aplikasi Shopee.
- Melakukan preprocessing terhadap data teks ulasan.
- Mengklasifikasikan ulasan ke dalam sentimen Positif, Netral, dan Negatif.
- Mengetahui performa model Naive Bayes dalam melakukan klasifikasi sentimen.

## 📊 Dataset

Dataset yang digunakan berupa **10.000 ulasan pengguna aplikasi Shopee dari Google Play Store**.

Data diperoleh menggunakan library `google-play-scraper`.

Dataset awal terdiri dari 11 kolom, kemudian dilakukan proses cleaning sehingga diperoleh beberapa kolom yang relevan untuk analisis, yaitu:

- `reviewId` — ID ulasan
- `content` — isi ulasan pengguna
- `score` — rating yang diberikan pengguna
- `at` — waktu ulasan

## 🔄 Tahapan Analisis

Tahapan yang dilakukan dalam proyek ini meliputi:

1. **Pengumpulan Data**
   - Mengambil ulasan aplikasi Shopee dari Google Play Store.

2. **Data Understanding**
   - Melihat jumlah data, jumlah kolom, tipe data, missing value, dan statistik data.

3. **Data Cleaning**
   - Menghapus kolom yang tidak diperlukan.
   - Mempertahankan data yang relevan untuk analisis sentimen.

4. **Pelabelan Sentimen**
   
   Label sentimen ditentukan berdasarkan rating:
   
   - Rating 1–2 → **Negatif**
   - Rating 3 → **Netral**
   - Rating 4–5 → **Positif**

5. **Text Preprocessing**
   - Case folding
   - Cleaning
   - Tokenization
   - Stopword removal
   - Stemming menggunakan **Sastrawi**

6. **Feature Extraction**
   - Mengubah teks menjadi representasi numerik menggunakan **TF-IDF**.

7. **Klasifikasi**
   - Menggunakan algoritma **Multinomial Naive Bayes**.

8. **Evaluasi Model**
   - Mengukur performa model menggunakan metrik evaluasi klasifikasi.

## 📈 Distribusi Data

Dari 10.000 data ulasan:

| Sentimen | Jumlah |
|---|---:|
| Positif | 7.261 |
| Negatif | 2.360 |
| Netral | 379 |

Distribusi menunjukkan bahwa sebagian besar ulasan pengguna memiliki sentimen positif.

## 🛠️ Tools & Library

Proyek ini menggunakan:

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Sastrawi
- Matplotlib
- Seaborn
- WordCloud
- Google Play Scraper

## 📁 Struktur Project

```text
sentiment-analysis-shopee/
│
├── main.py
├── shopee_reviews.csv
├── hasil_prediksi_shopee.csv
└── README.md
```

##👩‍💻 Author

Puja Widyasti

Program Studi Informatika
Universitas Sultan Ageng Tirtayasa
