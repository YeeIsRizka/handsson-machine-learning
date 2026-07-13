# BISINDO Hand Landmark Recognition

Proyek pengenalan bahasa isyarat **BISINDO** (Bahasa Isyarat Indonesia) huruf A–Z menggunakan **MediaPipe Hands** untuk ekstraksi landmark dan **MLP (Multilayer Perceptron)** untuk klasifikasi, dengan model yang dapat di-deploy ke web melalui **TensorFlow.js**.

---

## 📋 Deskripsi

Proyek ini terdiri dari dua komponen utama:

1. **Data Collection** — Script Python untuk mengumpulkan data landmark tangan secara real-time melalui webcam menggunakan MediaPipe Hands
2. **Model Training** — Notebook Google Colab untuk melatih model MLP klasifikasi gesture BISINDO dan mengonversinya ke format TensorFlow.js

### Fitur Utama

- 🖐️ **Deteksi otomatis mode satu/dua tangan** berdasarkan label huruf
- 📐 **Normalisasi dua tahap**: Translation → Scale normalization
- 📊 **Format konsisten 126 fitur** (63 per tangan × 2 tangan) untuk semua label
- 🧠 **Model MLP** dengan arsitektur `128 → 64 → 26` neuron
- 🌐 **Export ke TensorFlow.js** untuk deployment di web browser
- 📈 **Evaluasi lengkap**: Classification report, confusion matrix, grafik akurasi & loss

---

## 📁 Struktur Proyek

```
final/
├── data-collection/                  # Modul pengumpulan data
│   ├── collect_landmarks.py          # Script utama pengumpulan data via webcam
│   └── requirements.txt             # Dependensi untuk data collection
│
├── model/                            # Model hasil pelatihan
│   ├── label_encoder.pkl             # Label encoder (sklearn)
│   ├── saved_models_mlp.zip          # Model TensorFlow SavedModel
│   └── mlp_tfjs_model.zip            # Model TensorFlow.js (untuk web deployment)
│
├── BISINDO0.ipynb                    # Notebook pelatihan model (Google Colab)
├── bisindo0.py                       # Versi script dari notebook
├── bisindo_landmarks_scaled.csv      # Dataset utama (scale normalized landmarks)
├── train.csv                         # Dataset training (80%)
├── test.csv                          # Dataset testing (20%)
├── requirements.txt                  # Dependensi untuk training (Colab environment)
└── README.md                         # Dokumentasi (file ini)
```

---

## 🏷️ Klasifikasi Label

| Mode         | Label Huruf                                      |
|-------------|--------------------------------------------------|
| Satu Tangan | C, E, I, J, L, O, R, U, V, Z                    |
| Dua Tangan  | A, B, D, F, G, H, K, M, N, P, Q, S, T, W, X, Y |

---

## 📊 Format Dataset

File `bisindo_landmarks_scaled.csv` memiliki **127 kolom**:

| Kolom | Deskripsi |
|---|---|
| `label` | Huruf alfabet BISINDO (A–Z) |
| `left_0_x` ... `left_20_z` | 63 fitur landmark tangan kiri (21 titik × 3 koordinat) |
| `right_0_x` ... `right_20_z` | 63 fitur landmark tangan kanan (21 titik × 3 koordinat) |

> **Catatan:** Untuk label satu tangan, fitur tangan yang tidak terdeteksi diisi `0.0`.

### Proses Normalisasi

1. **Translation Normalization** — Setiap landmark dikurangi posisi wrist (titik 0) sehingga wrist menjadi origin `(0, 0, 0)`
2. **Scale Normalization** — Landmark yang sudah ditranslasi dibagi dengan jarak Euclidean antara nilai max dan min, sehingga invariant terhadap ukuran tangan

### Pembagian Dataset

| Split | Proporsi | File |
|-------|----------|------|
| Training | 80% | `train.csv` |
| Testing  | 20% | `test.csv` |

Pembagian dilakukan secara **stratified** (proporsi tiap label terjaga) dengan `random_state=42`.

---

## 🧠 Arsitektur Model

Model menggunakan **Multilayer Perceptron (MLP)** dengan arsitektur berikut:

```
Input (126 fitur)
    │
Dense(128, ReLU)
    │
Dropout(0.25)
    │
Dense(64, ReLU)
    │
Dropout(0.2)
    │
Dense(26, Softmax)  →  Output (A-Z)
```

### Hyperparameter Training

| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 100 (max) |
| Batch Size | 32 |
| Early Stopping | patience = 10, restore best weights |
| ReduceLROnPlateau | factor = 0.5, patience = 5, min_lr = 1e-6 |

---

## ⚙️ Instalasi & Penggunaan

### A. Data Collection (Lokal)

#### Prasyarat
- Python 3.8+
- Webcam

#### Setup & Run

```bash
cd data-collection

# Install dependensi
pip install -r requirements.txt

# Jalankan script
python collect_landmarks.py
```

#### Kontrol Pengambilan Data

| Tombol | Fungsi |
|--------|--------|
| `s` | Mulai pengambilan data (countdown 5 detik) |
| `q` | Keluar dari program |

#### Alur Penggunaan

1. Jalankan script → pilih label huruf → masukkan jumlah sampel
2. Tekan `s` pada window kamera → countdown 5 detik
3. Data diambil otomatis setiap **1 detik**
4. Sampel hanya disimpan jika jumlah tangan terdeteksi sesuai mode label
5. Data di-append ke `bisindo_landmarks_scaled.csv`

---

### B. Model Training (Google Colab)

1. Upload `BISINDO0.ipynb` ke [Google Colab](https://colab.research.google.com/)
2. Upload `bisindo_landmarks_scaled.csv` ke environment Colab
3. Jalankan semua cell secara berurutan
4. Download model hasil training dari folder `model/`

#### Output Training

| File | Format | Kegunaan |
|------|--------|----------|
| `label_encoder.pkl` | Pickle | Mapping label huruf ↔ index numerik |
| `saved_models_mlp.zip` | TensorFlow SavedModel | Inferensi di Python/TensorFlow |
| `mlp_tfjs_model.zip` | TensorFlow.js | Deployment di web browser |

---

## 🔧 Konfigurasi

### Data Collection

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `SAMPLE_INTERVAL` | `1.0` | Interval waktu antar sampel (detik) |
| `PREPARATION_TIME` | `5` | Waktu persiapan sebelum mulai (detik) |
| `min_detection_confidence` | `0.7` | Confidence minimum deteksi tangan |
| `min_tracking_confidence` | `0.7` | Confidence minimum tracking tangan |

---

## 🧪 Pipeline End-to-End

```
┌─────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                       │
│                                                         │
│  Webcam → MediaPipe Hands → Extract 21 Landmarks/Hand  │
│        → Translation Norm → Scale Norm → CSV            │
└──────────────────────────┬──────────────────────────────┘
                           │
                    bisindo_landmarks_scaled.csv
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    MODEL TRAINING                        │
│                                                         │
│  CSV → Stratified Split (80/20)                         │
│      → Label Encoding → MLP Training                   │
│      → Evaluation (Report + Confusion Matrix)           │
│      → Export (SavedModel + TensorFlow.js)              │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 Sumber Dataset Referensi

Dataset ini dikumpulkan secara mandiri melalui webcam. Berikut referensi dataset BISINDO lain yang digunakan sebagai studi:

- [Alfabet BISINDO — Achmad Noer](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)
- [BISINDO Dataset — Yunita Ayu](https://www.kaggle.com/datasets/yunitayupratiwi/bisindo-dataset/data)
- [BISINDO Letter Dataset — Alfredo](https://www.kaggle.com/datasets/alfredolorentiars/bisindo-letter-dataset)
- [BISINDO Hand Sign Detection — Rhio Sutoyo](https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset)
- [Sign Language BISINDO — Bonar Sitorus](https://www.kaggle.com/datasets/bonarsitorus/sign-language-bisindo)
- [BISINDO Final — Sifaq Einstein](https://www.kaggle.com/datasets/sifaqeinstein/bisindo-final)

---

## 🛠️ Tech Stack

| Teknologi | Kegunaan |
|-----------|----------|
| Python | Bahasa pemrograman utama |
| MediaPipe | Deteksi & ekstraksi landmark tangan |
| OpenCV | Akses webcam & visualisasi |
| NumPy | Komputasi normalisasi |
| TensorFlow / Keras | Training model MLP |
| Scikit-learn | Label encoding, train-test split, evaluasi |
| TensorFlow.js | Konversi model untuk web deployment |
| Matplotlib & Seaborn | Visualisasi (grafik, confusion matrix) |
| Google Colab | Environment training |

---

## 📜 Lisensi

Proyek ini dibuat untuk keperluan skripsi/penelitian akademis.

---

## 🤝 Kontribusi

Kontribusi dalam bentuk *issue* maupun *pull request* sangat diterima. Silakan buat issue terlebih dahulu untuk mendiskusikan perubahan yang diinginkan.
