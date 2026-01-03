# 🧬 Genetic Disorder Classification using ID3 Algorithm

> Implementasi algoritma ID3 (Iterative Dichotomiser 3) untuk klasifikasi gangguan genetik berbasis machine learning dengan Python

[![Python](https://img.shields.io/badge/Python-100%25-3776AB?style=flat-square&logo=python&logoColor=white)](https://github.com/bagaspng/genDisorder-ID3)
[![ID3 Algorithm](https://img.shields.io/badge/Algorithm-ID3-success?style=flat-square&logo=algolia&logoColor=white)](https://github.com/bagaspng/genDisorder-ID3)
[![Machine Learning](https://img.shields.io/badge/Category-Machine%20Learning-blue?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/bagaspng/genDisorder-ID3)
[![Healthcare](https://img.shields.io/badge/Domain-Healthcare-red?style=flat-square&logo=heartai&logoColor=white)](https://github.com/bagaspng/genDisorder-ID3)


## 📋 Deskripsi

Genetic Disorder Classification adalah sistem machine learning yang menggunakan algoritma **ID3 (Iterative Dichotomiser 3)** untuk mengklasifikasikan jenis gangguan genetik berdasarkan data medis pasien. Proyek ini mengimplementasikan decision tree dari scratch dan menyediakan visualisasi pohon keputusan untuk membantu interpretasi hasil diagnosis.

## ✨ Fitur Utama

### 🎯 **Machine Learning Features**
- 🌳 **ID3 Algorithm** - Implementasi decision tree dari scratch
- 📊 **Information Gain** - Pemilihan fitur optimal berdasarkan entropy
- 🔄 **Data Preprocessing** - Diskretisasi dan normalisasi data medis
- ⚖️ **Dataset Balancing** - Penanganan imbalanced classes

### 🏥 **Medical Domain Features**
- 🧬 **Genetic Disorder Classification** - Klasifikasi berbagai jenis gangguan genetik
- 🩺 **Medical Feature Processing** - Pengolahan data blood cell count dan parameter medis
- 📈 **Clinical Decision Support** - Bantuan diagnosis untuk tenaga medis
- 📋 **Patient Data Analysis** - Analisis komprehensif data pasien

### 🔧 **Technical Features**
- 📁 **Modular Architecture** - Pemisahan algoritma dan main logic
- 🎨 **Tree Visualization** - Visualisasi struktur pohon keputusan
- 📊 **Performance Metrics** - Evaluasi akurasi dan classification report
- 💾 **Result Export** - Penyimpanan hasil prediksi dan model

## 🧠 Algoritma ID3

### 📖 **Konsep Dasar**

ID3 (Iterative Dichotomiser 3) adalah algoritma decision tree yang: 

1. **Information Gain Based** - Memilih split berdasarkan information gain tertinggi
2. **Entropy Calculation** - Menggunakan entropi untuk mengukur ketidakpastian
3. **Recursive Tree Building** - Membangun pohon secara rekursif
4. **Categorical Features** - Optimal untuk fitur kategorikal

### ⚙️ **Cara Kerja Algoritma**

```python
def id3_algorithm_flow():
    """
    1. Hitung entropy dataset
    2. Untuk setiap fitur, hitung information gain
    3. Pilih fitur dengan information gain tertinggi
    4. Split dataset berdasarkan fitur terpilih
    5. Ulangi proses untuk setiap subset
    6. Stop jika semua instance memiliki label sama
    """
    pass
```

### 📊 **Formula Matematika**

#### **Entropy Calculation**
```
Entropy(S) = -Σ(pi * log2(pi))
```

#### **Information Gain**
```
IG(S,A) = Entropy(S) - Σ(|Sv|/|S| * Entropy(Sv))
```

### 📈 **Kompleksitas Algoritma**

| Aspek | Complexity | Keterangan |
|-------|------------|------------|
| **Time Complexity** | O(m * n * log n) | m = features, n = samples |
| **Space Complexity** | O(n) | Untuk menyimpan tree structure |
| **Training Speed** | Fast | Lebih cepat dari C4.5/CART |
| **Interpretability** | ⭐⭐⭐⭐⭐ | Sangat mudah diinterpretasi |

## 🚀 Instalasi & Setup

### 📦 **Prerequisites**

```bash
# Python 3.7+
python --version

# Package manager
pip --version
```

### 🔧 **Install Dependencies**

```bash
# Install required packages
pip install pandas numpy scikit-learn

# Alternative: install from requirements
pip install -r requirements.txt
```

### 📥 **Clone Repository**

```bash
git clone https://github.com/bagaspng/genDisorder-ID3.git
cd genDisorder-ID3
```

### 📋 **Requirements File**

```txt name=requirements.txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

## 🏗️ Struktur Project

```
genDisorder-ID3/
│
├── 📂 src/                    # Source code
│   ├── 📄 main.py            # Main execution script
│   ├── 📄 id3_algorithm.py   # Core ID3 implementation
│   └── 📂 __pycache__/       # Python cache files
│
├── 📂 dataset/               # Medical datasets
│   ├── 📊 train_genetic_disorders.csv    # Training data
│   └── 📊 test_genetic_disorders.csv     # Testing data
│
├── 📂 result/                # Output results
│   └── 🖼️ id3_tree.png      # Decision tree visualization
│
├── 📂 . kaggle/               # Kaggle API configuration
│   └── 🔑 kaggle.json       # API credentials
│
├── 📋 requirements.txt       # Dependencies
└── 📖 README.md             # Documentation (this file)
```

## 🎮 Penggunaan

### 🏃‍♂️ **Quick Start**

```bash
# Navigate to source directory
cd src

# Run the main program
python main.py
```

### 📊 **Expected Output**

```
=== Struktur Pohon Keputusan (ID3) ===
Blood cell count (mcL) = low
    Gender = Female
        → Label:  Alzheimer's
    Gender = Male
        → Label: Diabetes
Blood cell count (mcL) = normal
    Age = Adult
        → Label:  Healthy
    Age = Child
        → Label:  Autism

=== Evaluasi Model pada Data Uji ===
Akurasi: 0.87

=== Prediksi Gangguan Genetik per Sampel ===
Sampel 1: Alzheimer's
Sampel 2: Diabetes
Sampel 3: Healthy
... 

=== Ringkasan Hasil Prediksi ===
- Alzheimer's: 45 kasus dari 200 sampel
- Diabetes: 38 kasus dari 200 sampel
- Healthy: 67 kasus dari 200 sampel
- Autism: 25 kasus dari 200 sampel
- Other disorders: 25 kasus dari 200 sampel
```


## 👨‍💻 Author

**Bagas Pangestu** ([@bagaspng](https://github.com/bagaspng))


### 📊 **Dataset Sources**
- [Kaggle Genetic Disorders Dataset](https://kaggle.com/datasets/genetic-disorders)
- [UCI ML Repository - Medical Datasets](https://archive.ics.uci.edu/ml/)

---

<div align="center">

**🧬 Advancing Healthcare through AI & Machine Learning 🏥**

[![GitHub stars](https://img.shields.io/github/stars/bagaspng/genDisorder-ID3?style=social)](https://github.com/bagaspng/genDisorder-ID3/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/bagaspng/genDisorder-ID3?style=social)](https://github.com/bagaspng/genDisorder-ID3/network/members)

**Made with ❤️ for the Medical AI Community**

</div>
