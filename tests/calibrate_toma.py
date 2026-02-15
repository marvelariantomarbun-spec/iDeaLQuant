
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.indicators.trend import TOMA
from src.indicators.core import EMA

def parse_turkish_float(x):
    if isinstance(x, str):
        # Remove thousands separator first (if any, though not seen in sample)
        # Replace decimal comma with dot
        return float(x.replace(',', '.'))
    return x

def load_bardata(filepath):
    print(f"Loading BarData from {filepath}...")
    # Format: Tarih;Saat;Acilis;Yuksek;Dusuk;Kapanis;...
    # 07.02.2024;09:25:00;...
    
    df = pd.read_csv(filepath, sep=';')
    
    # Combine Date and Time
    df['Datetime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    
    # Parse Close Price
    # Headers might be garbled due to encoding, let's use index if needed or try to find 'Kapanis'
    # Based on sample: 5th index (0-based) is Close? 
    # Tarih, Saat, Acilis, Yuksek, Dusuk, Kapanis
    # 0,     1,    2,      3,      4,     5
    
    # Let's clean headers
    df.columns = [c.strip() for c in df.columns]
    
    # Find Close column
    close_col = [c for c in df.columns if 'Kapan' in c or 'Close' in c]
    if not close_col:
        # Fallback to index 5
        print("Warning: 'Kapanis' column not found by name, using index 5.")
        close_series = df.iloc[:, 5].apply(parse_turkish_float)
    else:
        close_series = df[close_col[0]].apply(parse_turkish_float)
        
    df['Close'] = close_series
    return df[['Datetime', 'Close']].set_index('Datetime')

def load_reference(filepath):
    print(f"Loading Reference from {filepath}...")
    # Format: Date;Close;TOMA;Trend
    # 2024-02-07 09:25;9670;0;1
    
    df = pd.read_csv(filepath, sep=';')
    df['Datetime'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')
    
    # Parse TOMA (already dot decimal in sample)
    # But checking just in case
    df['TOMA_Ref'] = df['TOMA'].apply(lambda x: float(str(x).replace(',', '.')))
    df['Trend_Ref'] = df['Trend'].astype(int)
    
    return df[['Datetime', 'TOMA_Ref', 'Trend_Ref']].set_index('Datetime')

def main():
    bardata_path = r"D:\Projects\IdealQuant\data\VIP_X030T_1dk_362780bardata.csv"
    ref_path = r"d:\Projects\IdealQuant\reference\IdealQuant_TOMA_Data.csv"
    
    # 1. Load Data
    try:
        bars = load_bardata(bardata_path)
        refs = load_reference(ref_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Align Data
    # Inner join on index (Datetime) to ensure we compare same bars
    data = bars.join(refs, how='inner')
    
    if data.empty:
        print("Error: No overlapping dates found between BarData and Reference!")
        print(f"BarData Range: {bars.index.min()} - {bars.index.max()}")
        print(f"Ref Range: {refs.index.min()} - {refs.index.max()}")
        return
        
    print(f"Aligned Data Points: {len(data)}")
    
    # 3. Calculate Python TOMA
    # Create list from Closes
    closes = data['Close'].tolist()
    
    # Parameters from reference code: Period=3, Percent=2.0 (Defaults)
    # Need to check if user changed params in generated file. 
    # Assumed defaults 3, 2.0 based on manual code text.
    
    print("Calculating Python TOMA (Period=3, Opt=2.0)...")
    toma_py, trend_py = TOMA(closes, period=3, percent=2.0)
    
    data['TOMA_Py'] = toma_py
    data['Trend_Py'] = trend_py
    
    # 4. Compare
    # Ignore first few bars where EMA initializes (or align with Ref's 0s)
    # Ref has 0s at start. Py will have EMA values.
    # Let's compare where Ref != 0
    
    valid_data = data[data['TOMA_Ref'] != 0].copy()
    
    if valid_data.empty:
        print("Warning: Reference TOMA is all zeros?")
        return
        
    valid_data['Diff'] = valid_data['TOMA_Ref'] - valid_data['TOMA_Py']
    valid_data['AbsDiff'] = valid_data['Diff'].abs()
    
    mae = valid_data['AbsDiff'].mean()
    max_error = valid_data['AbsDiff'].max()
    rmse = np.sqrt((valid_data['Diff'] ** 2).mean())
    
    print("\n--- Calibration Results ---")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Max Error: {max_error:.4f}")
    
    # Trend Match %
    trend_match = (valid_data['Trend_Ref'] == valid_data['Trend_Py']).mean() * 100
    print(f"Trend Match: {trend_match:.2f}%")
    
    # Show worst mismatches used for debugging
    worst = valid_data.nlargest(5, 'AbsDiff')
    print("\nWorst Mismatches:")
    print(worst[['Close', 'TOMA_Ref', 'TOMA_Py', 'Diff', 'Trend_Ref', 'Trend_Py']])
    
    # Plot - Skipped due to missing lib
    # print("\nPlot saved to d:\Projects\IdealQuant\toma_calibration_plot.png")

if __name__ == "__main__":
    main()
