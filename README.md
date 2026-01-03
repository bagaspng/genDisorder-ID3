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



## 🏥 Medical Domain Application

### 🧬 **Genetic Disorders Covered**

Sistem ini dapat mengklasifikasikan berbagai jenis gangguan genetik: 

| Category | Disorders | Characteristics |
|----------|-----------|-----------------|
| **Neurological** | Alzheimer's, Autism | Memory, cognitive function |
| **Metabolic** | Diabetes, Hemochromatosis | Metabolism disorders |
| **Developmental** | Cystic Fibrosis, Tay-Sachs | Growth and development |
| **Blood Disorders** | Thalassemia, Sickle Cell | Blood cell abnormalities |




## 🚀 Extensions & Improvements

### 🔮 **Future Enhancements**

#### **Algorithm Improvements**
- [ ] **C4.5 Algorithm** - Handle continuous variables better
- [ ] **Random Forest** - Ensemble of decision trees
- [ ] **Gradient Boosting** - Advanced ensemble methods
- [ ] **Pruning Strategies** - Post-pruning for generalization

#### **Medical Domain Enhancements**
- [ ] **SNOMED CT Integration** - Medical terminology standardization
- [ ] **ICD-10 Mapping** - International disease classification
- [ ] **Clinical Guidelines** - Evidence-based decision rules
- [ ] **Multi-language Support** - International medical terms

#### **Technical Features**
```python
# Planned enhancements
class AdvancedGeneticClassifier:
    def __init__(self):
        self.models = {
            'id3': ID3Classifier(),
            'c45': C45Classifier(),
            'random_forest': RandomForestClassifier(),
            'gradient_boost': GradientBoostClassifier()
        }
    
    def ensemble_predict(self, X):
        """Use ensemble of multiple algorithms"""
        predictions = {}
        for name, model in self. models.items():
            predictions[name] = model.predict(X)
        
        return self.weighted_ensemble_vote(predictions)
    
    def explain_prediction(self, patient_data):
        """Provide detailed explanation of prediction"""
        return {
            'decision_path': self. trace_decision_path(patient_data),
            'feature_importance': self. calculate_shap_values(patient_data),
            'similar_cases': self.find_similar_patients(patient_data),
            'confidence_interval': self.bootstrap_confidence(patient_data)
        }
```

### 🎨 **Visualization Enhancements**

```python
def create_interactive_tree_viz():
    """Create interactive web-based tree visualization"""
    
    import plotly.graph_objects as go
    import dash
    from dash import dcc, html
    
    # Create interactive plotly figure
    fig = go.Figure()
    
    # Add interactive nodes and edges
    # Allow zooming, panning, node details on hover
    
    # Create Dash web app
    app = dash.Dash(__name__)
    app.layout = html. Div([
        dcc.Graph(figure=fig),
        html.Div(id='node-details')
    ])
    
    return app
```

## 📚 Educational Content

### 🎓 **Learning Objectives**

Dengan menggunakan proyek ini, Anda akan mempelajari: 

1. **Machine Learning Fundamentals**
   - Decision tree algorithms
   - Information theory (entropy, information gain)
   - Model evaluation metrics

2. **Medical Informatics**
   - Healthcare data preprocessing
   - Medical feature engineering
   - Clinical decision support systems

3. **Python Programming**
   - pandas for data manipulation
   - numpy for numerical computations
   - scikit-learn for evaluation metrics

### 📖 **Theoretical Background**

#### **Information Theory Concepts**
```python
def information_theory_primer():
    """
    Entropy:  Measure of uncertainty in a dataset
    - High entropy:  uniform distribution (high uncertainty)
    - Low entropy: skewed distribution (low uncertainty)
    
    Information Gain:  Reduction in entropy after splitting
    - Measures how well a feature separates the data
    - ID3 selects feature with highest information gain
    """
    
    # Example calculation
    def entropy_example():
        # Dataset with 8 positive, 2 negative examples
        p_pos = 8/10
        p_neg = 2/10
        entropy = -(p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))
        return entropy  # ≈ 0.72 bits
```

### 🧪 **Hands-on Exercises**

```python
# Exercise 1: Implement different splitting criteria
def implement_gini_index(y):
    """Alternative to entropy:  Gini impurity"""
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

# Exercise 2: Add feature selection
def select_top_features(X, y, k=10):
    """Select top k features based on information gain"""
    feature_scores = []
    for feature in X.columns:
        gain = info_gain(X, y, feature)
        feature_scores.append((feature, gain))
    
    # Sort by gain and select top k
    top_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)[:k]
    return [feature for feature, _ in top_features]

# Exercise 3: Implement cost-sensitive learning
def cost_sensitive_id3(X, y, features, cost_matrix):
    """ID3 with different misclassification costs"""
    # Modify entropy calculation based on cost matrix
    # Higher cost for missing serious disorders (false negatives)
    pass
```

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

### 🔄 **Development Setup**

```bash
# Fork and clone repository
git clone https://github.com/your-username/genDisorder-ID3.git
cd genDisorder-ID3

# Create virtual environment
python -m venv genetic_env
source genetic_env/bin/activate  # Linux/macOS
# genetic_env\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### 📋 **Contribution Guidelines**

```python
# Code style guidelines
def function_naming_convention():
    """
    1. Use snake_case for functions and variables
    2. Use PascalCase for classes
    3. Add comprehensive docstrings
    4. Include type hints where appropriate
    """
    pass

def medical_data_handling_ethics():
    """
    IMPORTANT: Medical Data Ethics
    1. Never commit real patient data
    2. Use synthetic/anonymized datasets only
    3. Follow HIPAA guidelines
    4. Add privacy protection measures
    """
    pass
```

### 🧪 **Testing Framework**

```python
import unittest
import pandas as pd
import numpy as np

class TestID3Algorithm(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self. test_data = pd. DataFrame({
            'feature1': ['A', 'A', 'B', 'B'],
            'feature2': ['X', 'Y', 'X', 'Y'],
            'target': ['pos', 'neg', 'pos', 'pos']
        })
    
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        y = self.test_data['target']
        calculated_entropy = entropy(y)
        expected_entropy = 0.811  # Manual calculation
        self.assertAlmostEqual(calculated_entropy, expected_entropy, places=3)
    
    def test_information_gain(self):
        """Test information gain calculation"""
        X = self.test_data[['feature1', 'feature2']]
        y = self.test_data['target']
        gain = info_gain(X, y, 'feature1')
        self.assertGreater(gain, 0)
    
    def test_tree_building(self):
        """Test ID3 tree construction"""
        X = self.test_data[['feature1', 'feature2']]
        y = self.test_data['target']
        tree = id3(X, y, ['feature1', 'feature2'])
        self.assertIn('feature', tree)

if __name__ == '__main__':
    unittest.main()
```

### 💡 **Enhancement Ideas**

- [ ] **Web Interface** - Flask/Django web application
- [ ] **Mobile App** - React Native/Flutter app
- [ ] **API Integration** - RESTful API for external systems
- [ ] **Real-time Processing** - Stream processing capabilities
- [ ] **Federated Learning** - Distributed training across hospitals

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**⚠️ Medical Disclaimer**: This software is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and medical professional oversight.

## 👨‍💻 Author

**Bagas Pangestu** ([@bagaspng](https://github.com/bagaspng))

- 📧 Email: bagaspangestu0407@gmail.com
- 💼 LinkedIn: [Bagas Pangestu](https://linkedin.com/in/bagaspng)
- 🌐 Portfolio: [bagaspng.dev](https://bagaspng.dev)
- 🎓 Expertise: Machine Learning, Healthcare AI, Decision Trees

## 🙏 Acknowledgments

- **Medical Community** - For providing valuable domain knowledge
- **Machine Learning Researchers** - For foundational algorithms
- **Open Source Community** - For tools and libraries
- **Healthcare Data Providers** - For enabling medical AI research

## 📚 References

### 📖 **Academic Papers**
- Quinlan, J. R. (1986). "Induction of Decision Trees"
- Breiman, L.  et al. (1984). "Classification and Regression Trees"
- Mitchell, T. (1997). "Machine Learning"

### 🌐 **Medical Resources**
- [NCBI Genetic Testing Registry](https://www.ncbi.nlm.nih.gov/gtr/)
- [OMIM - Online Mendelian Inheritance in Man](https://omim.org/)
- [ClinVar - Clinical Significance of Variants](https://www.ncbi.nlm. nih.gov/clinvar/)

### 📊 **Dataset Sources**
- [Kaggle Genetic Disorders Dataset](https://kaggle.com/datasets/genetic-disorders)
- [UCI ML Repository - Medical Datasets](https://archive.ics.uci.edu/ml/)

## 📞 Support

Need help?  Contact us: 

- 📖 **Documentation**: [Project Wiki](https://github.com/bagaspng/genDisorder-ID3/wiki)
- 🐛 **Issues**: [Report Bugs](https://github.com/bagaspng/genDisorder-ID3/issues)
- 💬 **Discussions**: [Q&A Forum](https://github.com/bagaspng/genDisorder-ID3/discussions)
- 📧 **Email**: bagaspangestu0407@gmail.com

---

<div align="center">

**🧬 Advancing Healthcare through AI & Machine Learning 🏥**

[![GitHub stars](https://img.shields.io/github/stars/bagaspng/genDisorder-ID3? style=social)](https://github.com/bagaspng/genDisorder-ID3/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/bagaspng/genDisorder-ID3?style=social)](https://github.com/bagaspng/genDisorder-ID3/network/members)

**Made with ❤️ for the Medical AI Community**

</div>
