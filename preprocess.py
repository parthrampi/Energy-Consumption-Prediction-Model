import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.dropna()

    X = df[['voltage', 'current', 'temperature']]
    y = df['power']

    return X, y
