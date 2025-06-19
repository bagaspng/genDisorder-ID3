import pandas as pd

# Baca dataset
df = pd.read_csv("../dataset/train_genetic_disorders.csv")

# Lihat 5 data pertama
print(df.head())
