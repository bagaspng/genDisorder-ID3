import pandas as pd
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

def export_tree_to_dot(tree, dot_file_path):
    def node_id():
        node_id.counter += 1
        return f"node{node_id.counter}"
    node_id.counter = -1

    def recurse(tree):
        curr_id = node_id()
        if 'label' in tree:
            label = tree['label'].replace('"', "'")
            lines.append(f'{curr_id} [label="{label}", shape=box, style=filled, color=lightblue];')
            return curr_id
        else:
            label = tree['feature'].replace('"', "'")
            lines.append(f'{curr_id} [label="{label}", shape=ellipse, style=filled, color=lightyellow];')
            for val, subtree in tree['nodes'].items():
                child_id = recurse(subtree)
                edge_label = str(val).replace('"', "'")
                lines.append(f'{curr_id} -> {child_id} [label="{edge_label}"];')
            return curr_id

    lines = ['digraph ID3Tree {', 'node [fontname="Arial"];']
    recurse(tree)
    lines.append('}')
    with open(dot_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nPohon ID3 diekspor ke file DOT: {dot_file_path}")

def main():
    df_train = pd.read_csv("../dataset/train_genetic_disorders.csv")
    df_test = pd.read_csv("../dataset/test_genetic_disorders.csv")

    drop_cols = [
        'Patient Id', 'Patient First Name', 'Family Name', "Father's name",
        'Institute Name', 'Location of Institute'
    ]
    df_train = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns], errors='ignore')
    df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns], errors='ignore')

    df_train = df_train.dropna(subset=["Genetic Disorder"])

    X_train = df_train.drop(columns=["Genetic Disorder", "Disorder Subclass"], errors='ignore')
    y_train = df_train["Genetic Disorder"]

    X_test = df_test[X_train.columns]

    X_train = X_train.fillna('missing').astype(str)
    X_test = X_test.fillna('missing').astype(str)

    features = list(X_train.columns)
    tree = id3(X_train, y_train, features)

    print("Pohon keputusan:")
    print_tree(tree)

    predictions = predict_all(tree, X_test)
    print("\nPrediksi untuk data uji:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred}")

    # Export ke Graphviz DOT
    export_tree_to_dot(tree, "../result/id3_tree.dot")

if __name__ == "__main__":
    main()