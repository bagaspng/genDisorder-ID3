import numpy as np

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

def info_gain(X, y, feature):
    total_entropy = entropy(y)
    vals = np.unique(X[feature])
    weighted_entropy = 0
    for val in vals:
        y_sub = y[X[feature] == val]
        weighted_entropy += (len(y_sub)/len(y)) * entropy(y_sub)
    return total_entropy - weighted_entropy

def id3(X, y, features):
    if len(np.unique(y)) == 1:
        return {'label': y.iloc[0]}
    if not features:
        return {'label': y.mode()[0]}
    
    gains = [info_gain(X, y, f) for f in features]
    best = features[np.argmax(gains)]

    tree = {'feature': best, 'nodes': {}}
    for val in np.unique(X[best]):
        subset = X[best] == val
        subtree = id3(X[subset], y[subset], [f for f in features if f != best])
        tree['nodes'][val] = subtree

    return tree

def print_tree(tree, indent=""):
    if 'label' in tree:
        print(indent + f"→ Label: {tree['label']}")
    else:
        for val, subtree in tree['nodes'].items():
            print(indent + f"{tree['feature']} = {val}")
            print_tree(subtree, indent + "  ")
