
import pandas as pd
from src.data.ideal_parser import read_ideal_data
from datetime import datetime

file_path = "D:\\iDeal\\ChartData\\VIP\\B\\VIP'VIP-X030-T.B"

try:
    print(f"Reading {file_path}...")
    df = read_ideal_data(file_path)
    
    print(f"Rows: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    # Calculate time differences
    df['Diff'] = df['DateTime'].diff()
    
    print("\nTime Differences Analysis:")
    print(df['Diff'].value_counts().head())
    
    # Check if mostly 7 days
    mode_diff = df['Diff'].mode()[0]
    print(f"\nMost common difference: {mode_diff}")
    
except Exception as e:
    print(f"Error: {e}")
