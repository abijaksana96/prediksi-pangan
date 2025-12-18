# 🍚 Prediksi Harga Pangan Indonesia dengan Graph Neural Network (GNN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.0%2B-green.svg)](https://pytorch-geometric.readthedocs.io/)

Proyek ini bertujuan untuk memprediksi harga pangan (khususnya **Beras Premium**) di seluruh provinsi Indonesia menggunakan pendekatan **Temporal Graph Convolutional Network (T-GCN)**. Model ini menggabungkan analisis spasial antar-provinsi dan temporal untuk menghasilkan prediksi harga yang akurat.

## 📋 Daftar Isi

- [Gambaran Umum](#-gambaran-umum)
- [Arsitektur Model](#-arsitektur-model)
- [Dataset](#-dataset)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Metodologi](#-metodologi)
- [Hasil dan Evaluasi](#-hasil-dan-evaluasi)
- [Referensi](#-referensi)
- [Lisensi](#-lisensi)

## 🎯 Gambaran Umum

Harga pangan merupakan salah satu indikator penting dalam ekonomi Indonesia. Fluktuasi harga pangan dapat berdampak signifikan pada kesejahteraan masyarakat dan stabilitas ekonomi. Proyek ini mengembangkan model prediksi harga pangan berbasis deep learning yang mempertimbangkan:

- **Hubungan spasial antar-provinsi**: Harga pangan di satu provinsi dapat dipengaruhi oleh provinsi lain
- **Pola temporal**: Tren historis dan pola musiman dalam pergerakan harga
- **Fitur tambahan**: Informasi waktu seperti minggu kalender, bulan, dan kuartal

### Fitur Utama

- ✅ Prediksi harga beras premium untuk 38 provinsi di Indonesia
- ✅ Model T-GCN yang menggabungkan GCN (Graph Convolutional Network) dan GRU (Gated Recurrent Unit)
- ✅ Dukungan untuk berbagai horizon prediksi (7, 14, 30 hari ke depan)
- ✅ Fitur temporal dengan encoding siklus (sine/cosine)
- ✅ Visualisasi hasil prediksi

## 🏗 Arsitektur Model

Model yang digunakan adalah **Temporal Graph Convolutional Network (T-GCN)** yang terdiri dari komponen berikut:

```
Input (batch_size, seq_len, num_features)
           │
           ▼
    ┌─────────────────┐
    │   GCNConv       │  ← Menangkap hubungan spasial antar-provinsi
    │   (per timestep)│
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Calendar Week   │  ← Embedding untuk fitur temporal
    │   Embedding     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      GRU        │  ← Menangkap dependensi temporal
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Fully Connected│  ← Output prediksi harga
    └────────┬────────┘
             │
             ▼
Output (batch_size, num_provinces)
```

### Komponen Model

| Komponen | Deskripsi |
|----------|-----------|
| **GCNConv** | Graph Convolutional layer untuk menangkap hubungan spasial antar-provinsi |
| **Calendar Embedding** | Linear layer untuk memproses fitur siklus waktu (calweek_sin, calweek_cos) |
| **GRU** | Gated Recurrent Unit untuk menangkap pola temporal dalam data time series |
| **FC Layer** | Fully connected layer untuk menghasilkan prediksi akhir |

### Hyperparameter Terbaik

Parameter model optimal yang ditemukan melalui tuning:

```json
{
  "hidden_channels": 160,
  "num_layers": 4,
  "dropout": 0.15,
  "learning_rate": 0.00369,
  "weight_decay": 0.000161,
  "batch_size": 128
}
```

## 📊 Dataset

### Sumber Data

Dataset berisi harga harian **Beras Premium** dari **38 provinsi** di Indonesia dengan periode:
- **Tanggal Mulai**: 17 Mei 2022
- **Tanggal Akhir**: 16 Mei 2025
- **Total Data**: 41,648 baris

### Format Data

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `Tanggal` | datetime | Tanggal pencatatan harga |
| `Komoditas` | string | Jenis komoditas (Beras Premium) |
| `Provinsi` | string | Nama provinsi |
| `Harga` | float | Harga dalam Rupiah (Rp) |

### Provinsi yang Tercakup

Data mencakup 38 provinsi di Indonesia termasuk:
- Aceh, Sumatera Utara, Sumatera Barat, Riau, Jambi, Sumatera Selatan, Bengkulu, Lampung
- DKI Jakarta, Jawa Barat, Jawa Tengah, D.I Yogyakarta, Jawa Timur, Banten
- Bali, Nusa Tenggara Barat, Nusa Tenggara Timur
- Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan, Kalimantan Timur, Kalimantan Utara
- Sulawesi Utara, Sulawesi Tengah, Sulawesi Selatan, Sulawesi Tenggara, Gorontalo, Sulawesi Barat
- Maluku, Maluku Utara, Papua, Papua Barat, dan provinsi-provinsi pemekaran Papua lainnya

## 🛠 Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Jupyter Notebook/Lab (opsional, untuk menjalankan notebook)

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/abijaksana96/prediksi-pangan.git
   cd prediksi-pangan
   ```

2. **Buat virtual environment (disarankan)**
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   # atau
   env\Scripts\activate     # Windows
   ```

3. **Install PyTorch** (sesuaikan dengan sistem Anda)
   ```bash
   # CPU only
   pip install torch torchvision torchaudio
   
   # Atau dengan CUDA (untuk GPU)
   # Kunjungi https://pytorch.org/get-started/locally/ untuk versi yang sesuai
   ```

4. **Install PyTorch Geometric**
   ```bash
   pip install torch-geometric
   pip install torch-scatter torch-sparse
   ```

5. **Install dependencies lainnya**
   ```bash
   pip install -r requirements.txt
   pip install scikit-learn optuna
   ```

### Dependensi Utama

| Package | Versi | Deskripsi |
|---------|-------|-----------|
| `pandas` | 2.2.3 | Manipulasi dan analisis data |
| `numpy` | 2.2.6 | Komputasi numerik |
| `torch` | ≥2.0.0 | PyTorch deep learning framework |
| `torch-geometric` | ≥2.0.0 | Library untuk Graph Neural Networks |
| `scikit-learn` | ≥1.0.0 | Preprocessing dan evaluasi |
| `matplotlib` | 3.10.3 | Visualisasi data |
| `seaborn` | 0.13.2 | Visualisasi statistik |
| `optuna` | ≥3.0.0 | Hyperparameter tuning (opsional) |

## 🚀 Penggunaan

### Menjalankan Notebook

1. **Notebook Utama - Model T-GCN**
   ```bash
   jupyter notebook gnn_model_by_paper.ipynb
   ```
   Notebook ini berisi implementasi utama model T-GCN dengan:
   - Preprocessing data
   - Pembangunan graph berdasarkan korelasi harga
   - Training model
   - Evaluasi hasil

2. **Notebook GCN dengan Fitur Tambahan**
   ```bash
   jupyter notebook gcn_fix.ipynb
   ```
   Notebook ini berisi model GCN dengan fitur tambahan:
   - Delta harga (perubahan harga harian)
   - Rolling average 7 hari
   - Volatilitas 7 hari
   - Fitur musiman (bulan, kuartal)

3. **Notebook Versi Peningkatan**
   ```bash
   jupyter notebook prediksi_harga_pangan_improved.ipynb
   ```
   Berisi implementasi dengan berbagai peningkatan dan hyperparameter tuning menggunakan Optuna.

### Contoh Kode Singkat

```python
import pandas as pd
import torch
from torch_geometric.nn import GCNConv

# Load data
df = pd.read_csv("harga_beras_premium.csv")
df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)

# Pivot data menjadi format (tanggal x provinsi)
df_pivot = df.pivot(index="Tanggal", columns="Provinsi", values="Harga")

# Preprocessing
# ... (lihat notebook untuk detail lengkap)

# Inisialisasi model T-GCN
model = TGCN(
    num_nodes=38,          # Jumlah provinsi
    num_features=1,        # Fitur per node (harga)
    hidden_dim=64,         # Dimensi hidden layer
    output_dim=38,         # Output untuk semua provinsi
    edge_index=edge_index, # Struktur graph
    use_calweek=True       # Gunakan fitur waktu
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
```

### Menggunakan Model Terlatih

```python
# Load model terlatih
model.load_state_dict(torch.load("final_gnn_model.pth"))
model.eval()

# Prediksi
with torch.no_grad():
    predictions = model(X_new)
    
# Inverse transform ke skala Rupiah asli
predictions_rupiah = scaler.inverse_transform(predictions)
```

## 📁 Struktur Proyek

```
prediksi-pangan/
├── README.md                           # Dokumentasi proyek (file ini)
├── requirements.txt                    # Daftar dependensi Python
├── harga_beras_premium.csv             # Dataset utama
├── best_gnn_params.json                # Hyperparameter terbaik
├── final_gnn_model.pth                 # Model terlatih (PyTorch)
├── s41598-025-97724-7.pdf              # Paper referensi
│
├── gnn_model_by_paper.ipynb            # Implementasi T-GCN utama
├── gcn_fix.ipynb                       # GCN dengan fitur tambahan
├── prediksi_harga_pangan_improved.ipynb# Versi peningkatan dengan Optuna
├── harga_pangan.ipynb                  # Eksplorasi awal
├── harga_pangan copy.ipynb             # Eksperimen tambahan
├── harga_pangan copy 2.ipynb           # Eksperimen tambahan
├── percobaan3.ipynb                    # Eksperimen model
└── test.ipynb                          # Testing dan validasi
```

### Deskripsi File

| File | Deskripsi |
|------|-----------|
| `gnn_model_by_paper.ipynb` | Implementasi utama model T-GCN berdasarkan paper referensi |
| `gcn_fix.ipynb` | Model GCN dengan fitur engineering tambahan (delta, rolling, volatility) |
| `prediksi_harga_pangan_improved.ipynb` | Versi peningkatan dengan hyperparameter tuning |
| `harga_beras_premium.csv` | Dataset harga beras premium harian per provinsi |
| `best_gnn_params.json` | Parameter model terbaik hasil tuning |
| `final_gnn_model.pth` | Bobot model terlatih |

## 📐 Metodologi

### 1. Preprocessing Data

```
Data Mentah → Pivot (Tanggal x Provinsi) → Handle Missing Values → Normalisasi
```

- **Pivot**: Mengubah format data dari long ke wide (baris = tanggal, kolom = provinsi)
- **Missing Values**: Menggunakan forward fill untuk mengisi nilai yang hilang
- **Normalisasi**: StandardScaler untuk normalisasi fitur

### 2. Feature Engineering

| Fitur | Deskripsi | Transformasi |
|-------|-----------|--------------|
| Harga Asli | Harga harian per provinsi | Normalisasi |
| Calendar Week | Minggu dalam tahun (1-53) | Sin/Cos encoding |
| Month | Bulan dalam tahun (1-12) | Sin/Cos encoding |
| Quarter | Kuartal dalam tahun (1-4) | Sin/Cos encoding |
| Delta Harga | Perubahan harga dari hari sebelumnya | - |
| Rolling Mean | Rata-rata harga 7 hari | - |
| Volatility | Standar deviasi harga 7 hari | - |

### 3. Konstruksi Graph

Graph dibangun berdasarkan **korelasi harga** antar-provinsi:

```python
def build_edge_index_from_correlation(df, threshold=0.5):
    corr_matrix = df.corr().abs().values
    corr_matrix[corr_matrix < threshold] = 0
    edge_index, edge_weight = dense_to_sparse(torch.tensor(corr_matrix))
    return edge_index
```

- **Node**: Setiap provinsi adalah satu node
- **Edge**: Terhubung jika korelasi harga > threshold (0.5)
- **Total Edges**: ~544 edges untuk 38 nodes

### 4. Pembagian Data

```
Data Total (1096 hari)
├── Training   : Jan 2024 - Aug 2024 (225 hari, ~46%)
├── Validation : Sep 2024 - Jan 2025 (153 hari, ~31%)
└── Testing    : Feb 2025 - Mei 2025 (105 hari, ~23%)
```

### 5. Training

- **Optimizer**: Adam dengan weight decay (L2 regularization)
- **Loss Function**: MSE (Mean Squared Error)
- **Epochs**: 50
- **Early Stopping**: Berdasarkan validation loss

## 📈 Hasil dan Evaluasi

### Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **RMSE** | Root Mean Square Error - mengukur rata-rata kesalahan prediksi |
| **MAE** | Mean Absolute Error - rata-rata absolut kesalahan |
| **MAPE** | Mean Absolute Percentage Error - persentase rata-rata kesalahan |
| **R² Score** | Koefisien determinasi - seberapa baik model menjelaskan variansi data |

### Contoh Hasil

Hasil evaluasi pada test set (contoh untuk satu provinsi):

```
📊 Evaluasi Test Sample - Provinsi Aceh
RMSE (Rp): ~1,464
MAE  (Rp): ~736
MAPE (%): ~4.17%
```

### Visualisasi

Model menghasilkan visualisasi perbandingan harga aktual vs prediksi:

```
Harga (Rp)
    │
 17K│      ╭──────────╮
    │     ╱            ╲
 16K│    ╱              ╲
    │   ╱                ╲
 15K│──╱                  ╲──
    │ Actual (─) vs Predicted (--)
    └────────────────────────────── Waktu
```

## 📚 Referensi

1. **Paper Referensi**: Implementasi model ini terinspirasi dari metodologi dalam paper `s41598-025-97724-7.pdf` yang disertakan dalam repository.

2. **PyTorch Geometric Documentation**: https://pytorch-geometric.readthedocs.io/

3. **Graph Neural Networks for Time Series**: 
   - T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
   - Spatio-Temporal Graph Convolutional Networks

## 📝 Lisensi

Proyek ini dibuat untuk tujuan penelitian dan edukasi. Silakan hubungi pemilik repository untuk penggunaan komersial.

## 👨‍💻 Kontributor

- **abijaksana96** - Pengembang Utama

---

<p align="center">
  Dibuat dengan ❤️ untuk Indonesia
</p>
