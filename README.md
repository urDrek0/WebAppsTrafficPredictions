# 🚦 Munich Traffic Prediction System (AI-Based)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Status](https://img.shields.io/badge/Status-Deployment%20Ready-brightgreen)

Sistem prediksi kemacetan lalu lintas cerdas yang menggunakan **Machine Learning** untuk memprediksi *Flow* (Arus) dan *Occupancy* (Okupansi) jalan, serta dilengkapi dengan **Fuzzy Logic** untuk mengklasifikasikan status kemacetan (Lancar, Padat Merayap, Macet).

Proyek ini dibangun dari tahap eksplorasi data (EDA), pelatihan model, hingga deployment menggunakan arsitektur *Client-Server* (FastAPI sebagai Backend & Streamlit sebagai Frontend). Atau dapat diakses melalui link berikut.

```link
https://trafficpredictionsdatsa2025.streamlit.app/
```

---

## 📂 Sumber Data (Dataset)

Dataset yang digunakan dalam proyek ini berasal dari **UTD19 (Urban Traffic Data 2019)**.
* **Lokasi:** Munich, Jerman.
* **Fitur Utama:** Waktu (Interval detik ke-0 hingga 86.400), Flow (Jumlah kendaraan), Occupancy (Persentase kepadatan jalan).
* **Preprocessing:** Pembersihan *outliers* menggunakan metode IQR, penghapusan data *error* sensor, dan normalisasi untuk model tertentu (SVR).

---

## 🧠 Model AI yang Digunakan

Sistem ini membandingkan 6 model algoritma regresi yang berbeda untuk performa terbaik:

1.  **XGBoost Regressor** (Recommended) 🏆
2.  **LightGBM** (Light Gradient Boosting Machine) ⚡
3.  **Extra Trees Regressor** 🌳
4.  **Support Vector Regression (SVR)** 📈 *(Menggunakan Scaling)*
5.  **Random Forest Regressor** 🌲
6.  **Polynomial Regression** 📐

Selain itu, sistem menggunakan **Fuzzy Logic (Mamdani)** untuk mengubah angka prediksi menjadi keputusan status jalan yang mudah dipahami manusia.

---

## ⚙️ Struktur Folder Proyek

Pastikan susunan folder Anda terlihat seperti ini agar sistem berjalan lancar:

```text
project_traffic/
│
├── main.py                  # 🧠 Backend Server (FastAPI)
├── dashboard.py             # 💻 Frontend Interface (Streamlit)
├── requirements.txt         # 📦 Daftar Library Python
├── README.md                # 📄 Dokumentasi ini
│
├── data/                    # 📂 Folder Data
│   └── munich.csv           # File dataset bersih (untuk visualisasi grafik)
│
└── models/                  # 📂 Folder Penyimpanan Model (.pkl)
    ├── model_flow_xgb.pkl
    ├── model_occ_xgb.pkl
    ├── model_flow_lgbm.pkl
    ├── model_occ_lgbm.pkl
    ├── model_flow_et.pkl
    ├── model_occ_et.pkl
    ├── model_flow_rf.pkl
    ├── model_occ_rf.pkl
    ├── model_flow_svr.pkl   # Khusus SVR
    ├── model_occ_svr.pkl    # Khusus SVR
    ├── scaler_x_flow.pkl    # Scaler Input Flow (Wajib utk SVR)
    ├── scaler_y_flow.pkl    # Scaler Output Flow (Wajib utk SVR)
    ├── scaler_x_occ.pkl     # Scaler Input Occ (Wajib utk SVR)
    └── scaler_y_occ.pkl     # Scaler Output Occ (Wajib utk SVR)
````

-----

## 🚀 Panduan Instalasi (Lokal)

Ikuti langkah ini untuk menjalankan proyek di laptop/komputer Anda.

### 1\. Clone Repository & Masuk ke Folder

```bash
git clone [https://github.com/username-anda/nama-repo.git](https://github.com/username-anda/nama-repo.git)
cd nama-repo
```

### 2\. Buat Virtual Environment (Venv)

Sangat disarankan menggunakan venv agar library tidak bentrok.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

*(Jika berhasil, akan muncul tulisan `(venv)` di terminal Anda).*

### 3\. Install Dependencies

Install semua library yang dibutuhkan (FastAPI, Streamlit, Scikit-learn, dll).

```bash
pip install -r requirements.txt
```

-----

## ▶️ Cara Menjalankan Aplikasi

Anda perlu membuka **2 Terminal** berbeda untuk menjalankan Backend dan Frontend secara bersamaan.

### Terminal 1: Menyalakan Backend (API)

Ini adalah "Otak" yang memproses prediksi.

```bash
# Pastikan venv sudah aktif
uvicorn main:app --reload
```

*Tunggu hingga muncul pesan: `Application startup complete`.*

### Terminal 2: Menyalakan Frontend (Dashboard)

Ini adalah "Wajah" web yang bisa diakses user.

```bash
# Buka terminal baru, aktifkan venv lagi, lalu:
streamlit run dashboard.py
```

### 🌐 Akses Website

Otomatis browser akan terbuka di alamat:
👉 **http://localhost:8501**

-----

## 🛠️ Alur Kerja Sistem (Pipeline)

1.  **Input:** Pengguna memasukkan waktu (detik ke-sekian atau jam dinding) melalui Slider di Streamlit.
2.  **Request:** Frontend mengirim data JSON ke API FastAPI (`/predict`).
3.  **Processing (Backend):**
      * Server memilih model `.pkl` sesuai request user (misal: XGBoost atau SVR).
      * Jika **SVR**, data di-*scaling* dulu -\> diprediksi -\> di-*inverse transform*.
      * Jika **Tree-based** (XGB, RF, ET), langsung diprediksi.
4.  **Decision:** Hasil prediksi (Flow & Occupancy) masuk ke logika "Status Helper" (Lancar/Macet).
5.  **Response:** Hasil dikirim balik ke Frontend.
6.  **Visualization:**
      * Streamlit menampilkan angka prediksi.
      * Streamlit menghitung ulang **Fuzzy Logic** secara lokal untuk visualisasi grafik himpunan fuzzy & tabel rules.

-----

## 🐛 Troubleshooting (Kendala Umum)

  * **Error `ModuleNotFoundError`:**
    Pastikan Anda sudah menjalankan `pip install -r requirements.txt` dalam kondisi `(venv)` aktif.
  * **Error `Internal Server Error` (Merah):**
    Biasanya karena file `.pkl` di folder `models/` tidak lengkap. Cek kembali kelengkapan file model Anda.
  * **Grafik Kosong:**
    Pastikan file `munich.csv` ada di dalam folder `data/` agar grafik data historis bisa muncul.

-----

**Dibuat untuk Tugas Akhir / Proyek Data Science.**
*Feel free to contribute or give a star\! ⭐*

