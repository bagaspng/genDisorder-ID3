import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from id3_algorithm import id3, print_tree

# --- 1. Fungsi Diskretisasi Fitur Numerik ---
def discretize_blood_cell_count(series):
    # Ubah nilai numerik menjadi kategori: low, normal, high
    bins = [-np.inf, 4.5, 11, np.inf]  # contoh batasan (mcL, ribuan per mikroliter)
    labels = ['low', 'normal', 'high']
    # Tangani nilai yang tidak bisa dikonversi ke float
    series = pd.to_numeric(series, errors='coerce')
    return pd.cut(series, bins=bins, labels=labels).astype(str).fillna('missing')

# --- 2. Fungsi Balancing Dataset ---
def balance_per_label(df, label_col, max_per_label=100):
    # Ambil maksimal max_per_label data dari setiap label
    balanced = []
    for label, group in df.groupby(label_col):
        balanced.append(group.sample(n=min(len(group), max_per_label), random_state=42))
    return pd.concat(balanced).reset_index(drop=True)

def predict_all(tree, X):
    preds = []
    for _, row in X.iterrows():
        node = tree
        while 'label' not in node:
            feat = node['feature']
            val = row[feat]
            if val in node['nodes']:
                node = node['nodes'][val]
            else:
                # fallback: ambil label mayoritas di node
                node = list(node['nodes'].values())[0]
        preds.append(node['label'])
    return preds

def main():
    # --- 3. Load Data dan Ambil Sample ---
    df_train = pd.read_csv("../dataset/train_genetic_disorders.csv")
    df_test = pd.read_csv("../dataset/test_genetic_disorders.csv")

    # --- 4. Hapus Kolom Identitas ---
    drop_cols = [
        'Patient Id', 'Patient First Name', 'Family Name', "Father's name",
        'Institute Name', 'Location of Institute'
    ]
    df_train = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns], errors='ignore')
    df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns], errors='ignore')

    # --- 5. Diskretisasi Fitur Numerik ---
    for df in [df_train, df_test]:
        if "Blood cell count (mcL)" in df.columns:
            df["Blood cell count (mcL)"] = discretize_blood_cell_count(df["Blood cell count (mcL)"])
        # Tambahkan diskretisasi fitur numerik lain jika perlu

    # --- 6. Tangani NaN dan Konversi ke String ---
    df_train = df_train.fillna('missing').astype(str)
    df_test = df_test.fillna('missing').astype(str)

    # --- 7. Drop Baris dengan Target NaN ---
    df_train = df_train[df_train["Genetic Disorder"] != 'missing']

    # --- 8. Balance Dataset ---
    df_train = balance_per_label(df_train, "Genetic Disorder", max_per_label=100)

    # --- 9. Siapkan Fitur dan Target ---
    X_train = df_train.drop(columns=["Genetic Disorder", "Disorder Subclass"], errors='ignore')
    y_train = df_train["Genetic Disorder"]
    X_test = df_test[X_train.columns]

    # --- 10. Latih Pohon ID3 ---
    features = list(X_train.columns)
    tree = id3(X_train, y_train, features)

    # --- 11. Visualisasi Pohon (opsional, tampilkan di terminal) ---
    print("\n=== Struktur Pohon Keputusan (ID3) ===")
    print_tree(tree)

    # --- 12. Prediksi Data Uji ---
    predictions = predict_all(tree, X_test)

    # --- 13. Evaluasi Jika Label Uji Tersedia ---
    if "Genetic Disorder" in df_test.columns and not df_test["Genetic Disorder"].isnull().all():
        y_test = df_test["Genetic Disorder"]
        print("\n=== Evaluasi Model pada Data Uji ===")
        print("Akurasi:", accuracy_score(y_test, predictions))
        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, predictions))
    else:
        print("\nLabel data uji tidak tersedia. Hanya menampilkan prediksi.")

    # --- 14. Tampilkan Prediksi per Sampel ---
    print("\n=== Prediksi Gangguan Genetik per Sampel ===")
    for i, pred in enumerate(predictions, 1):
        print(f"Sampel {i}: {pred}")

    # --- 15. Ringkasan Hasil Prediksi ---
    counter = Counter(predictions)
    print("\n=== Ringkasan Hasil Prediksi ===")
    for label, jumlah in counter.items():
        print(f"- {label}: {jumlah} kasus dari {len(predictions)} sampel")

    # --- 16. Kesimpulan Otomatis ---
    print("\n=== Kesimpulan ===")
    most_common = counter.most_common(1)[0]
    # Cari fitur paling sering digunakan di root pohon
    fitur_utama = tree['feature'] if 'feature' in tree else '(tidak diketahui)'
    print(
        f"Sebagian besar sampel diprediksi sebagai '{most_common[0]}' "
        f"({most_common[1]} dari {len(predictions)} kasus). "
        f"Hal ini kemungkinan dipengaruhi oleh fitur '{fitur_utama}' yang menjadi akar pohon keputusan."
    )

if __name__ == "__main__":
    main()