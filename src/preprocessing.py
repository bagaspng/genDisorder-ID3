import pandas as pd

def load_data():
    df_train = pd.read_csv("../dataset/train_genetic_disorders.csv")
    df_test = pd.read_csv("../dataset/test_genetic_disorders.csv")
    return df_train, df_test

def preprocess(df):
    df = df.drop(columns=["Patient Id", "Name", "Contact"], errors="ignore")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes
        df = df.fillna(method='ffill')  # bisa juga df.fillna(df.mean(numeric_only=True))
    return df

def discretize_numeric(df, col, bins=3):
    df[col + "_binned"] = pd.cut(df[col], bins=bins, labels=False)
    df.drop(columns=[col], inplace=True)
    return df

def discretize_column(df, colname, bins=4):
    df[colname] = pd.cut(df[colname], bins=bins, labels=False)
    return df
