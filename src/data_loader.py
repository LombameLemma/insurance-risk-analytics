pd.read_csv(path)
import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Error loading data: {e}")