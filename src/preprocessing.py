import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_data(path, url=None):

    if os.path.exists(path):
        print("Loading local data...")
        return pd.read_csv(path, sep=None, engine='python')
    
    elif url:
        print("Downloading data...")
        df = pd.read_csv(url, sep=None, engine='python', encoding='latin1')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    
    else:
        raise ValueError("No data source available")

def scale_features(df):
    """
    Standardize numerical features using Z-score
    """
    scaler = StandardScaler()
    
    scaled_array = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(
        scaled_array,
        columns=df.columns,
        index=df.index
    )
    
    return df_scaled, scaler

def detect_outliers_iqr(df):
    """
    Detect outliers using IQR method
    Returns:
        df_clean: data without outliers
        df_outliers: detected outliers
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)

    df_clean = df[mask]
    df_outliers = df[~mask]

    return df_clean, df_outliers