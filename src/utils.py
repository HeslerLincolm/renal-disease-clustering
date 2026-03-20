import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def save_parquet(df, path):
    """
    Save DataFrame in parquet format
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine='fastparquet', index=False)

def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')

def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()

    return result, end - start