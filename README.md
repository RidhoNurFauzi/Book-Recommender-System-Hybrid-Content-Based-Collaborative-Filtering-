<p align="center">
  <img src="./images/Book.jpg" alt="Book Recommender System" width="100%">
</p>

<h1 align="center">📚 Book Recommender System: Hybrid Content-Based & Collaborative Filtering</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg" alt="Scikit-Learn"></a>
  <a href="https://surpriselib.com/"><img src="https://img.shields.io/badge/Surprise-SVD-green.svg" alt="Surprise"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
Proyek sistem rekomendasi buku end-to-end yang menggabungkan pendekatan <strong>Content-Based Filtering</strong> dan <strong>Collaborative Filtering (SVD)</strong> untuk menghasilkan rekomendasi yang relevan secara konten sekaligus dipersonalisasi, dibangun di atas dataset <strong>Book-Crossing</strong>.
</p>

---

## 📑 Daftar Isi

- [Ringkasan Eksekutif](#-ringkasan-eksekutif)
- [Problem Definition & Karakteristik Dataset](#-problem-definition--karakteristik-dataset)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [Visualisasi & Temuan Analisis](#️-visualisasi--temuan-analisis)
- [Pendekatan Pemodelan](#-pendekatan-pemodelan)
- [Evaluasi Model](#-evaluasi-model)
- [Contoh Hasil Rekomendasi](#-contoh-hasil-rekomendasi)
- [Tech Stack & Alat](#️-tech-stack--alat)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Menjalankan Proyek](#-cara-menjalankan-proyek)
- [Keterbatasan & Pengembangan Selanjutnya](#-keterbatasan--pengembangan-selanjutnya)
- [Kontak](#-kontak)

---

## 📊 Ringkasan Eksekutif

| Parameter | Detail |
|---|---|
| **Dataset & Ukuran** | Book-Crossing — 271.360 buku, 278.858 user, 1.149.780 rating |
| **Sparsity Matriks User-Item** | 99,9967% (setelah thresholding: 12.720 user × 18.318 buku) |
| **Pendekatan** | Hybrid: Content-Based (TF-IDF + Cosine Similarity) → Collaborative Filtering (SVD) |
| **Rating Eksplisit vs Implisit** | 433.671 eksplisit (1–10) vs 716.109 implisit (rating 0) |
| **Model Collaborative** | SVD (library `Surprise`), Cross-Validation 3-fold |
| **RMSE** | 3,5495 |
| **MAE** | 2,8159 |

---

## 📑 Problem Definition & Karakteristik Dataset

### 1. Masalah & Tujuan
Membangun sistem rekomendasi buku yang mampu:
- **Memprediksi rating** yang mungkin diberikan seorang user terhadap buku yang belum pernah ia baca.
- **Menghasilkan daftar Top-N rekomendasi** buku yang paling relevan untuk seorang user, dengan mempertimbangkan baik kemiripan konten buku maupun pola selera komunitas.

### 2. Tantangan Spesifik Dataset Book-Crossing
- **Implicit vs Explicit Feedback**: rating bernilai 0 (62% dari seluruh interaksi) menandakan user berinteraksi dengan buku tanpa memberi skor eksplisit, berbeda maknanya dari rating 1–10.
- **Data Sparsity ekstrem**: 99,9967% sel pada matriks user-item kosong — mayoritas user hanya merating 1 buku, dan mayoritas buku hanya dirating 1 kali.
- **Orphan Data**: 118.644 baris rating mengacu ke ISBN yang tidak terdaftar di tabel metadata buku.
- **Cold Start**: sulit merekomendasikan buku baru atau untuk user baru jika hanya mengandalkan collaborative filtering.

### 3. Struktur Data
- **Books**: ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher.
- **Ratings**: User-ID, ISBN, Book-Rating (skala 0–10).
- **Users**: User-ID, Location, Age.

---

## 🧹 Data Cleaning & Preprocessing

| Isu yang Ditemukan | Tindakan |
|---|---|
| 3 baris `Year-Of-Publication` mengalami pergeseran kolom (data shift) akibat parsing CSV | Diperbaiki manual, kolom dikembalikan ke posisi yang benar |
| Tahun terbit tidak valid (`0`, `2030`, `2050`, `1806`, dll — 4.633 baris) | Dikonversi ke numerik, nilai di luar rentang 1900–2026 dijadikan kosong |
| Umur user tidak masuk akal (`0`, `244`, dll — 1.248 baris) | Difilter ke rentang wajar 5–100 tahun |
| 118.644 baris rating dengan ISBN yang tidak ada di tabel Books (*orphan data*) | Dipisah jadi dua dataset: `df_ratings_content` (orphan dibuang, untuk content-based) dan `df_ratings_cf` (lengkap, untuk collaborative filtering) |
| Rating 0 (implicit) tercampur dengan rating eksplisit | Ditandai lewat kolom `Is-Explicit` agar bisa diperlakukan berbeda |
| Sparsity matriks user-item 99,9967% | Thresholding: hanya user dan buku dengan minimal 10 rating yang dipertahankan (→ 12.720 user, 18.318 buku, 443.196 rating) |

---

## 🖼️ Visualisasi & Temuan Analisis

### 1. Distribusi Rating & Profil Buku
| Histogram User-ID & Book-Rating | 10 Penulis & Penerbit Terbanyak |
|---|---|
| ![Histogram](./images/histogram%20User-ID%20dan%20Book-Rating.png) | ![Top Penulis Penerbit](./images/10%20Penulis%20dengan%20Buku%20Terbanyak%20dan%2010%20Penerbit%20dengan%20Buku%20Terbanyak.png) |

- **Distribusi rating** didominasi nilai 0 (implicit feedback, 62% dari total interaksi). Untuk rating eksplisit, skor tinggi (7–10) jauh lebih sering muncul dibanding skor rendah (1–2) — user cenderung hanya memberi rating saat menyukai buku.
- **Agatha Christie** tercatat sebagai penulis dengan jumlah buku terbanyak, dan **Harlequin** sebagai penerbit dengan jumlah buku terbanyak di dataset ini.

### 2. Long-Tail & Dampak Thresholding
| Distribusi Rating per User & Buku (log scale) | Dampak Thresholding terhadap Ukuran Data |
|---|---|
| ![Long Tail](./images/Distribusi%20Jumlah%20Rating%20per%20User%20%28log%20scale%29%20dan%20Distribusi%20Jumlah%20Rating%20per%20Buku%20%28log%20scale%29.png) | ![Dampak Thresholding](./images/Dampak%20Thresholding%20terhadap%20Ukuran%20Data.png) |

- **Long-tail**: median rating per user maupun per buku sama-sama 1, sementara ada *power user* dengan 13.602 rating dan buku terpopuler dengan 2.502 rating.
- **Thresholding** (minimal 10 rating per user/buku) memangkas data dari 105.283 user & 340.556 buku menjadi 12.720 user & 18.318 buku, tapi tetap mempertahankan 443.196 baris rating (~39% dari total) — trade-off yang sepadan untuk mendapatkan matriks yang lebih padat.

### 3. Profil Demografi User
| Distribusi Umur User (setelah cleaning) |
|---|
| ![Distribusi Umur](./images/Distribusi%20Umur%20User%20%28setelah%20cleaning%29.png) |

- Setelah outlier (umur 0 dan 244 tahun) dibersihkan, mayoritas user berada di rentang usia produktif, dengan median 32 tahun.

### 4. Evaluasi Model
| RMSE & MAE per Fold (Cross-Validation) |
|---|
| ![RMSE MAE](./images/RMSE%20%26%20MAE%20per%20Fold%20%28Cross-Validation%29.png) |

- Skor RMSE dan MAE relatif konsisten di setiap fold, menandakan performa model SVD stabil dan tidak overfit ke salah satu subset data tertentu.

---

## 🧠 Pendekatan Pemodelan

### 1. Content-Based Filtering
- Fitur teks (`content`) dibentuk dari gabungan `Book-Title + Book-Author + Publisher`.
- Diubah menjadi representasi numerik dengan **TF-IDF** (17.479 buku populer × 16.039 kata unik).
- Kemiripan antar buku dihitung dengan **Cosine Similarity** (matriks 17.479 × 17.479).
- Fungsi `recommend_books_cbf` mengembalikan Top-N buku paling mirip secara teks dari sebuah judul referensi.

### 2. Collaborative Filtering
- Model **SVD** (Singular Value Decomposition) dilatih dengan library `Surprise` menggunakan data rating yang telah di-threshold (`df_cf_filtered`).
- Dievaluasi dengan **Cross-Validation 3-fold**.
- Fungsi `recommend_collaborative_filtering` memprediksi rating untuk buku yang belum pernah dinilai seorang user, lalu mengambil Top-N skor prediksi tertinggi.

### 3. Hybrid System
Pendekatan dua fase, menggabungkan kekuatan keduanya:
1. **Fase Content-Based**: ambil 50 buku paling mirip secara konten dengan buku referensi.
2. **Fase Collaborative**: dari 50 kandidat tersebut, model SVD memprediksi skor personal untuk user tertentu, lalu diurutkan berdasarkan skor tertinggi.

Pendekatan ini mengatasi kelemahan masing-masing model tunggal: content-based saja tidak mempertimbangkan selera personal, sementara collaborative saja kesulitan menangani buku dengan interaksi minim.

---

## 📈 Evaluasi Model

| Model | Metrik | Skor |
|---|---|---|
| Collaborative Filtering (SVD) | RMSE | 3,5495 |
| Collaborative Filtering (SVD) | MAE | 2,8159 |

MAE 2,82 berarti rata-rata tebakan rating model meleset sekitar 2,82 poin dari rating asli pada skala 1–10. Mengingat tingginya proporsi implicit feedback dan sparsity data (99,9967%), tingkat error ini berada dalam batas wajar untuk sistem rekomendasi berbasis matriks skala besar.

**Catatan Error Analysis**: hasil content-based sering memunculkan variasi ISBN dari judul buku yang sama (beda edisi/penerbit), karena model menghitung kemiripan murni dari teks. Ini dicatat sebagai arah pengembangan selanjutnya (lihat bagian Keterbatasan).

---

## 📖 Contoh Hasil Rekomendasi

**Content-Based** — referensi: *Harry Potter and the Sorcerer's Stone*
Hasil didominasi seri Harry Potter lainnya (Chamber of Secrets, Goblet of Fire, Prisoner of Azkaban) — seluruhnya ditulis J.K. Rowling.

**Collaborative Filtering** — User-ID 276762
Rekomendasi lintas genre berdasarkan pola user lain: karya Shel Silverstein, seri Harry Potter, dan Michael Ende (*Die unendliche Geschichte*).

**Hybrid** — User-ID 276762, referensi *Harry Potter*
Top rekomendasi: *Harry Potter and the Chamber of Secrets (Book 2)* dengan skor prediksi 5,31 — relevan secara konten sekaligus dipersonalisasi.

---

## 🛠️ Tech Stack & Alat
* **Bahasa**: Python 3.8+
* **Manipulasi & Analisis Data**: `pandas`, `numpy`
* **Content-Based Filtering**: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
* **Collaborative Filtering**: `scikit-surprise` (SVD)
* **Visualisasi Data**: `matplotlib`, `seaborn`

---

## 📁 Struktur Proyek

```text
.
├── data/
│   ├── BX-Books.csv
│   ├── BX-Book-Ratings.csv
│   └── BX-Users.csv
├── images/                           # Hasil visualisasi & grafik proyek
│   ├── Book.jpg
│   ├── histogram User-ID dan Book-Rating.png
│   ├── 10 Penulis dengan Buku Terbanyak dan 10 Penerbit dengan Buku Terbanyak.png
│   ├── Distribusi Jumlah Rating per User (log scale) dan Distribusi Jumlah Rating per Buku (log scale).png
│   ├── Distribusi Jumlah Rating per Buku (log scale).png
│   ├── Dampak Thresholding terhadap Ukuran Data.png
│   ├── Distribusi Umur User (setelah cleaning).png
│   └── RMSE & MAE per Fold (Cross-Validation).png
├── Book_Recommender.ipynb            # Notebook pengerjaan utama
├── requirements.txt                  # Daftar dependensi Python
└── README.md
```

---

## 🚀 Cara Menjalankan Proyek

1. **Clone repository ini**
   ```bash
   git clone https://github.com/<username>/<nama-repo>.git
   cd <nama-repo>
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan notebook**
   ```bash
   jupyter notebook Book_Recommender.ipynb
   ```
   Jalankan seluruh sel secara berurutan (`Kernel → Restart & Run All`) untuk mereproduksi seluruh proses cleaning, pemodelan, dan evaluasi dari awal.

**Sumber Dataset:** [Book-Crossing Dataset (Kaggle)](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)

---

## ⚠️ Keterbatasan & Pengembangan Selanjutnya

1. Hasil content-based sering memunculkan variasi ISBN dari judul buku yang sama (beda edisi/penerbit); ke depan, buku dengan judul serupa bisa dikelompokkan (*grouping*) agar rekomendasi lebih beragam.
2. Kolom `Age` dan `Location` pada tabel Users belum dimanfaatkan sebagai fitur tambahan (demographic-based filtering); berpotensi memperkaya pendekatan hybrid di iterasi berikutnya.
3. Fitur teks content-based masih terbatas pada judul, penulis, dan penerbit — belum mencakup deskripsi/genre buku karena tidak tersedia di dataset asli.
4. Evaluasi collaborative filtering hanya menggunakan RMSE/MAE (rating prediction); belum diukur dengan metrik ranking seperti Precision@K atau NDCG untuk skenario Top-N recommendation.
5. Model belum diuji pada skenario cold-start murni (user atau buku benar-benar baru tanpa histori rating sama sekali).

---

## 📬 Kontak

**Ridho Nur Fauzi**
ML/DL Engineer & Data Scientist

* 🌐 Portfolio: [ridhonurfauzi.netlify.app](https://ridhonurfauzi.netlify.app/)
* 💼 Terbuka untuk diskusi kolaborasi proyek data science & machine learning

---

<p align="center"><em>Dibuat sebagai bagian dari proyek pembelajaran sistem rekomendasi menggunakan dataset Book-Crossing.</em></p>