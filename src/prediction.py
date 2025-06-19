from sklearn.metrics import accuracy_score

def predict(tree, sample):
    if 'label' in tree:
        return tree['label']
    feature = tree['feature']
    val = sample[feature]
    if val in tree['nodes']:
        return predict(tree['nodes'][val], sample)
    else:
        return -1  # fallback label

def predict_all(tree, X):
    return X.apply(lambda row: predict(tree, row), axis=1)

def evaluate(y_true, y_pred):
    print("Akurasi:", accuracy_score(y_true, y_pred))
