import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ideal_parser import read_ideal_data

def inspect_data():
    file_path = "D:\\iDeal\\ChartData\\VIP\\01\\VIP'VIP-X030-T.01"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"Reading file: {file_path}")
    try:
        df = read_ideal_data(file_path)
        if df is not None and len(df) > 0:
            print(f"Data loaded: {len(df)} bars.")
            print("\nHead:")
            print(df.head())
            print("\nTail:")
            print(df.tail())
            print("\nColumn Stats:")
            print(df[['Open', 'High', 'Low', 'Close']].describe())
        else:
            print("Loaded DataFrame is empty.")
    except Exception as e:
        print(f"Error reading data: {e}")

if __name__ == "__main__":
    inspect_data()
