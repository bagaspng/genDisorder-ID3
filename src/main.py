import pandas as pd
from collections import Counter
from id3_algorithm import id3, print_tree

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
                node = list(node['nodes'].values())[0]
        preds.append(node['label'])
    return preds

def main():
    # Ambil sample kecil dari data
    df_train = pd.read_csv("../dataset/train_genetic_disorders.csv").head(50)
    df_test = pd.read_csv("../dataset/test_genetic_disorders.csv").head(20)

    # Hapus kolom identitas jika ada
    drop_cols = [
        'Patient Id', 'Patient First Name', 'Family Name', "Father's name",
        'Institute Name', 'Location of Institute'
    ]
    df_train = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns], errors='ignore')
    df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns], errors='ignore')

    # Drop baris dengan target NaN
    df_train = df_train.dropna(subset=["Genetic Disorder"])

    # Siapkan fitur dan target
    X_train = df_train.drop(columns=["Genetic Disorder", "Disorder Subclass"], errors='ignore')
    y_train = df_train["Genetic Disorder"]

    # Samakan fitur pada data uji dan latih
    X_test = df_test[X_train.columns]

    # Isi NaN pada fitur dengan string 'missing', lalu ubah semua ke string
    X_train = X_train.fillna('missing').astype(str)
    X_test = X_test.fillna('missing').astype(str)

    # Latih pohon ID3
    features = list(X_train.columns)
    tree = id3(X_train, y_train, features)

    # Prediksi data uji
    predictions = predict_all(tree, X_test)

    # Ringkasan hasil prediksi
    print("\n=== Ringkasan Hasil Prediksi ===")
    counter = Counter(predictions)
    for label, jumlah in counter.items():
        print(f"- {label}: {jumlah} kasus dari {len(predictions)} sampel")

    print("\n=== Kesimpulan ===")
    most_common = counter.most_common(1)[0]
    print(f"Sebagian besar sampel diprediksi sebagai '{most_common[0]}' ({most_common[1]} dari {len(predictions)} kasus).")

    # Tampilkan prediksi per sampel
    print("\n=== Prediksi per Sampel ===")
    for i, pred in enumerate(predictions, 1):
        print(f"Sampel {i}: {pred}")

if __name__ == "__main__":
    main()