import sys
import os
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.ars_pulse_strategy import backtest_ars_pulse

def load_and_filter_data(file_path):
    print(f"Loading {file_path}...")
    # Use latin-1 for Turkish characters
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    # Map columns to internal names
    column_map = {
        'Tarih': 'Date',
        'Saat': 'Time',
        'Al': 'Acilis',
        'Yksek': 'Yuksek',
        'Dk': 'Dusuk',
        'Kapan': 'Kapanis',
        'Lot': 'Volume'
    }
    cols = df.columns.tolist()
    if 'Al' not in cols and 'Acilis' not in cols:
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
        filtered_df[col] = filtered_df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            
    filtered_df = filtered_df.sort_values('DateTime')
    return filtered_df

def run_optimization(file_path, timeframe_label):
    bars = load_and_filter_data(file_path)
    if len(bars) < 100:
        print(f"Not enough data for {timeframe_label}")
        return None
    
    print(f"Starting pulse optimization for {timeframe_label} ({len(bars)} bars)...")
    
    print(f"Starting Pulse STAGE 1 (Satellite) for {timeframe_label} ({len(bars)} bars)...")
    
    # Stage 1: Broad Scan
    ema_periods = [1, 5, 21, 55, 144, 377, 610, 987] # Fibonacci broad
    k_values = [round(x, 2) for x in np.arange(0.1, 10.1, 1.0)]
    adx_ths = [20, 30]
    netlot_ths = [5, 15]
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_params = {}
        for ep in ema_periods:
            for k in k_values:
                for adx in adx_ths:
                    for nl in netlot_ths:
                        params = {'ema_period': ep, 'k_value': k, 'adx_th': adx, 'netlot_th': nl}
                        future_to_params[executor.submit(backtest_ars_pulse, bars, params)] = params
        
        count = 0; total = len(future_to_params)
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]; res = future.result(); res.update(params); results.append(res)
            count += 1
            if count % 100 == 0: print(f"  {timeframe_label} Stage 1 Progress: {count}/{total}...", end="\r")
    
    df_s1 = pd.DataFrame(results)
    df_filtered = df_s1[df_s1['num_trades'] >= 20]
    if df_filtered.empty: return df_s1.sort_values(by='pf', ascending=False).iloc[0]
    
    # Select Top Candidates for Stage 2
    top_candidates = df_filtered.sort_values(by='pf', ascending=False).head(3)
    print(f"\n  Top candidates from Stage 1 for {timeframe_label}:")
    print(top_candidates[['ema_period', 'k_value', 'adx_th', 'netlot_th', 'pf']].to_string(index=False))
    
    # --- STAGE 2: Drone Scan (Narrow) ---
    print(f"\nStarting Pulse STAGE 2 (Drone) for {timeframe_label}...")
    s2_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_params = {}
        for idx, row in top_candidates.iterrows():
            ep_base = int(row['ema_period'])
            k_base = row['k_value']
            adx_base = int(row['adx_th'])
            nl_base = int(row['netlot_th'])
            
            # Focused EMA and K
            ep_range = range(max(1, ep_base - 10), ep_base + 11, 2)
            k_range = [round(x, 2) for x in np.arange(max(0.1, k_base - 1.0), k_base + 1.1, 0.1)] # K step 0.1 = raw 0.001
            
            for ep in ep_range:
                for k in k_range:
                    p = {'ema_period': ep, 'k_value': k, 'adx_th': adx_base, 'netlot_th': nl_base}
                    future_to_params[executor.submit(backtest_ars_pulse, bars, p)] = p
                    
        total2 = len(future_to_params); count2 = 0
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]; res = future.result(); res.update(params); s2_results.append(res)
            count2 += 1
            if count2 % 100 == 0: print(f"  {timeframe_label} Stage 2 Progress: {count2}/{total2}...", end="\r")

    df_res = pd.DataFrame(s2_results)
    df_final_filtered = df_res[df_res['num_trades'] >= 30]
    
    if df_final_filtered.empty:
        best = df_res.sort_values(by='pf', ascending=False).iloc[0]
    else:
        best = df_final_filtered.sort_values(by='pf', ascending=False).iloc[0]
        
    print(f"\nBest for {timeframe_label}: EMA={int(best['ema_period'])}, K={best['k_value']:.2f}, ADX={int(best['adx_th'])}, NetLot={int(best['netlot_th'])}, PF={best['pf']:.4f}")
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
    print("ARS PULSE (STRICT) OPTIMIZATION COMPLETED")
    print("="*125)
    print(f"{'Per':4} | {'EMA':3} | {'K':5} | {'ADX':3} | {'NL':3} | {'Points':10} | {'PF':6} | {'MaxDD-Pts':10} | {'Trades':6}")
    print("-" * 125)
    for label, best in final_results.items():
        print(f"{label:4} | {int(best['ema_period']):3} | {best['k_value']:5.2f} | {int(best['adx_th']):3} | {int(best['netlot_th']):3} | {best['net_points']:10.2f} | {best['pf']:6.4f} | {best['max_dd_points']:10.2f} | {int(best['num_trades']):6}")

if __name__ == "__main__":
    main()
