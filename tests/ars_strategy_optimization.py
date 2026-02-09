import sys
import os
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.ars_trend_strategy import backtest_ars_trend

def load_and_filter_data(file_path):
    print(f"Loading {file_path}...")
    # Use latin-1 for Turkish characters
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    # Map columns to internal names
    # Tarih;Saat;Al;Yksek;Dk;Kapan;Ortalama;Hacim;Lot
    column_map = {
        'Tarih': 'Date',
        'Saat': 'Time',
        'Al': 'Acilis',
        'Yksek': 'Yuksek',
        'Dk': 'Dusuk',
        'Kapan': 'Kapanis',
        'Lot': 'Volume'
    }
    # Sometimes naming varies slightly on different exports
    # Let's try to find indices if mapping fails
    cols = df.columns.tolist()
    if 'Al' not in cols and 'Acilis' not in cols:
        # Fallback to column position if headers are garbled
        df.columns = ['Date', 'Time', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Avg', 'Volume', 'Lot']
    else:
        df = df.rename(columns=column_map)
    
    # Combine Date and Time
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    
    # Filter for 2024-01-01 to 2025-12-31
    mask = (df['DateTime'] >= '2024-01-01') & (df['DateTime'] <= '2025-12-31')
    filtered_df = df.loc[mask].copy()
    
    # Ensure numeric types
    for col in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis']:
        # Force everything to string, remove thousand separator (dot), replace decimal (comma) with dot
        filtered_df[col] = filtered_df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            
    filtered_df = filtered_df.sort_values('DateTime')
    print(f"  Final columns types: {filtered_df[['Acilis', 'Yuksek', 'Dusuk', 'Kapanis']].dtypes.to_dict()}")
    return filtered_df

def run_optimization(file_path, timeframe_label):
    bars = load_and_filter_data(file_path)
    if len(bars) == 0:
        print(f"No data for {timeframe_label} in requested period.")
        return None
    
    print(f"Starting optimization for {timeframe_label} ({len(bars)} bars)...")
    
    # Search grid (ULTRA WIDE SCAN)
    ema_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200, 233, 377, 500]
    k_values = [round(x, 2) for x in np.arange(0.2, 20.2, 0.2)]
    
    results = []
    
    # Parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_params = {
            executor.submit(backtest_ars_trend, bars, ep, k): (ep, k)
            for ep in ema_periods for k in k_values
        }
        
        count = 0
        total = len(future_to_params)
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                res = future.result()
                res['ema_period'] = params[0]
                res['k_value'] = params[1]
                results.append(res)
            except Exception as e:
                print(f"Error for {params}: {e}")
            
            count += 1
            if count % 200 == 0:
                print(f"  {timeframe_label} Progress: {count}/{total}...", end="\r")
    
    df_res = pd.DataFrame(results)
    
    # Filter for significant trade count
    df_filtered = df_res[df_res['num_trades'] >= 40]
    
    if df_filtered.empty:
        best = df_res.sort_values(by='net_profit', ascending=False).iloc[0]
    else:
        # Show Top PF candidates
        df_top_pf = df_filtered[df_filtered['net_profit'] > 0.0].sort_values(by='pf', ascending=False).head(10)
        df_top_pf['monthly_trades'] = df_top_pf['num_trades'] / 24.0
        print(f"\nTop PF results for {timeframe_label}:")
        print(df_top_pf[['ema_period', 'k_value', 'pf', 'max_dd_points', 'net_points', 'num_trades', 'monthly_trades']].to_string(index=False))
        
        best = df_top_pf.iloc[0]
    
    return best

def main():
    data_files = {
        '5m': r"D:\Projects\IdealQuant\data\XU030_5dk_200000bar.csv",
        '15m': r"D:\Projects\IdealQuant\data\XU030_15dk_68081bar.csv",
        '60m': r"D:\Projects\IdealQuant\data\XU030_60dk_32944bar.csv"
    }
    
    final_results = {}
    for label, path in data_files.items():
        res = run_optimization(path, label)
        if res is not None:
            final_results[label] = res
            
    print("\n" + "="*125)
    print("ULTRA WIDE OPTIMIZATION COMPLETED (FINAL REPORT)")
    print("="*125)
    print(f"{'Per':4} | {'EMA':3} | {'K':5} | {'Points':10} | {'PF':6} | {'MaxDD-Pts':10} | {'Trades':6} | {'Monthly':7}")
    print("-" * 125)
    for label, best in final_results.items():
        m_trades = best['num_trades'] / 24.0
        print(f"{label:4} | {int(best['ema_period']):3} | {best['k_value']:5.2f} | {best['net_points']:10.2f} | {best['pf']:6.4f} | {best['max_dd_points']:10.2f} | {int(best['num_trades']):6} | {m_trades:7.2f}")

if __name__ == "__main__":
    main()
