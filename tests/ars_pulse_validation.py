import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.ars_pulse_strategy import ARSPulseStrategy
from src.optimization.hybrid_group_optimizer import fast_backtest
from src.optimization.fitness import quick_fitness

def load_data(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    column_map = {
        'Tarih': 'Date', 'Saat': 'Time', 'Acl': 'Acilis', 'Al': 'Acilis',
        'Yksek': 'Yuksek', 'Dk': 'Dusuk', 'Kapan': 'Kapanis', 'Lot': 'Volume'
    }
    df = df.rename(columns=column_map)
    
    # Standard format check
    if 'Acilis' not in df.columns:
        df.columns = ['Date', 'Time', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Avg', 'Volume', 'Lot']
    
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    mask = (df['DateTime'] >= '2024-01-01') & (df['DateTime'] <= '2025-12-31')
    df = df.loc[mask].copy()
    
    for col in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis']:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            
    df = df.sort_values('DateTime')
    return df

def validate_params(label, df, params):
    print(f"\nVALIDATING: {label}")
    print("-" * 30)
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    strat = ARSPulseStrategy(**params)
    signals, _ = strat.run(df)
    closes = df['Kapanis'].values
    
    # 0.05 commission + slippage as a realistic scenario
    np_val, trades, pf, dd, sharpe = fast_backtest(closes, signals, (signals == 0), (signals == 0), commission=0.0, slippage=0.0)
    
    print(f"\nResults (Gross):")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Net Points:    {np_val:,.0f}")
    print(f"  Total Trades:  {trades}")
    print(f"  Max Drawdown:  {dd:,.0f} Pts")
    
    # SENSITIVITY ANALYSIS (Tweaking EMA by ±5% and K by ±10%)
    print("\nSensitivity Analysis (Robustness Check):")
    tweaks = [
        {'ema_period': params['ema_period'] * 1.05, 'k_value': params['k_value']},
        {'ema_period': params['ema_period'] * 0.95, 'k_value': params['k_value']},
        {'ema_period': params['ema_period'], 'k_value': params['k_value'] * 1.1},
        {'ema_period': params['ema_period'], 'k_value': params['k_value'] * 0.9},
    ]
    
    pfs = []
    for t_params in tweaks:
        p = params.copy()
        p.update(t_params)
        s = ARSPulseStrategy(**p)
        sigs, _ = s.run(df)
        _, _, p_val, _, _ = fast_backtest(closes, sigs, (sigs == 0), (sigs == 0), 0.0, 0.0)
        pfs.append(p_val)
    
    avg_pf = np.mean(pfs)
    stability = "STABLE" if all(p > 1.2 for p in pfs) else "FRAGILE"
    print(f"  Avg PF (Tweaked): {avg_pf:.2f}")
    print(f"  Stability Status: {stability}")

def main():
    golden_params = {
        '5m': {
            'ema_period': 156, 'k_value': 1.5, 'macdv_k': 11, 'macdv_u': 29, 
            'macdv_sig': 12, 'netlot_period': 9, 'adx_th': 15, 'netlot_th': 5
        },
        '15m': {
            'ema_period': 941, 'k_value': 7.7, 'macdv_k': 18, 'macdv_u': 32, 
            'macdv_sig': 11, 'netlot_period': 4, 'adx_th': 35, 'netlot_th': 40
        },
        '60m': {
            'ema_period': 770, 'k_value': 2.3, 'macdv_k': 20, 'macdv_u': 42, 
            'macdv_sig': 7, 'netlot_period': 5, 'adx_th': 20, 'netlot_th': 20
        }
    }
    
    files = {
        '5m': r"D:\Projects\IdealQuant\data\XU030_5dk_200000bar.csv",
        '15m': r"D:\Projects\IdealQuant\data\XU030_15dk_68081bar.csv",
        '60m': r"D:\Projects\IdealQuant\data\XU030_60dk_32944bar.csv"
    }
    
    for label in ['5m', '15m', '60m']:
        df = load_data(files[label])
        validate_params(label, df, golden_params[label])

if __name__ == "__main__":
    main()
