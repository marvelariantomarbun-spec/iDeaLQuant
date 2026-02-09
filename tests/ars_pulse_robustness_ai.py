import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.ars_pulse_strategy import ARSPulseStrategy
from src.robust.monte_carlo import MonteCarloSimulator
from src.optimization.hybrid_group_optimizer import fast_backtest

def load_data_numeric(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    column_map = {'Tarih': 'Date', 'Saat': 'Time', 'Acl': 'Acilis', 'Al': 'Acilis', 'Yksek': 'Yuksek', 'Dk': 'Dusuk', 'Kapan': 'Kapanis', 'Lot': 'Volume'}
    df = df.rename(columns=column_map)
    if 'Date' not in df.columns:
        df.columns = ['Date', 'Time', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Avg', 'Volume', 'Lot']
    
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    
    for col in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    
    df = df.dropna(subset=['Acilis', 'Yuksek', 'Dusuk', 'Kapanis'])
    return df.sort_values('DateTime')

def run_validation(label, df, params):
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS VALIDATION: {label}")
    print(f"{'='*60}")
    
    # Use 2023-2025 but split IS/OOS
    # Warmup is handled by running the full DF and then slicing signals
    strat = ARSPulseStrategy(**params)
    signals_full, _ = strat.run(df)
    
    # Join signals to DF for easy slicing
    df['Signal'] = signals_full
    
    # 1. Split Data: In-Sample (2024), Out-of-Sample (2025)
    df_is = df.loc[(df['DateTime'] >= '2024-01-01') & (df['DateTime'] <= '2024-12-31')].copy()
    df_oos = df.loc[(df['DateTime'] >= '2025-01-01') & (df['DateTime'] <= '2025-12-31')].copy()
    
    print(f"In-Sample (2024) Bars:  {len(df_is)}")
    print(f"Out-of-Sample (2025) Bars: {len(df_oos)}")
    
    # IS Results
    np_is, tr_is, pf_is, dd_is, _ = fast_backtest(
        df_is['Kapanis'].to_numpy(), 
        df_is['Signal'].to_numpy(), 
        (df_is['Signal'] == 0).to_numpy(), 
        (df_is['Signal'] == 0).to_numpy(), 
        0.0, 0.0
    )
    
    # OOS Results
    np_oos, tr_oos, pf_oos, dd_oos, _ = fast_backtest(
        df_oos['Kapanis'].to_numpy(), 
        df_oos['Signal'].to_numpy(), 
        (df_oos['Signal'] == 0).to_numpy(), 
        (df_oos['Signal'] == 0).to_numpy(), 
        0.0, 0.0
    )
    
    print(f"\n[PART 1] IS/OOS RESULTS")
    print(f"  IS (2024)  -> PF: {pf_is:.2f} | Profit: {np_is:,.0f} | Trades: {tr_is}")
    print(f"  OOS (2025) -> PF: {pf_oos:.2f} | Profit: {np_oos:,.0f} | Trades: {tr_oos}")
    
    # 2. Monte Carlo on 2024-2025 combined
    print(f"\n[PART 2] MONTE CARLO SIMULATION")
    df_v = pd.concat([df_is, df_oos])
    sigs_v = df_v['Signal'].values
    closes = df_v['Kapanis'].values
    
    trades_pnl = []
    pos = 0
    entry = 0
    for i in range(1, len(sigs_v)):
        if sigs_v[i] != sigs_v[i-1]:
            if pos != 0:
                pnl = (closes[i] - entry) if pos == 1 else (entry - closes[i])
                trades_pnl.append(pnl)
            pos = sigs_v[i]
            entry = closes[i]
            
    if len(trades_pnl) >= 5:
        mc = MonteCarloSimulator(trades_pnl, initial_capital=100000)
        mc_res = mc.run_simulation(num_simulations=2000)
        mc.print_report(mc_res)
    else:
        print(f"Insufficient trades for MC ({len(trades_pnl)}). Strategy is extremely selective.")

def main():
    golden_params = {
        '5m': {'ema_period': 156, 'k_value': 1.5, 'macdv_k': 11, 'macdv_u': 29, 'macdv_sig': 12, 'netlot_period': 9, 'adx_th': 15, 'netlot_th': 5},
        '15m': {'ema_period': 941, 'k_value': 7.7, 'macdv_k': 18, 'macdv_u': 32, 'macdv_sig': 11, 'netlot_period': 4, 'adx_th': 35, 'netlot_th': 40}
    }
    
    files = {
        '5m': r"D:\Projects\IdealQuant\data\XU030_5dk_200000bar.csv",
        '15m': r"D:\Projects\IdealQuant\data\XU030_15dk_68081bar.csv"
    }

    for label in ['5m', '15m']:
        df = load_data_numeric(files[label])
        run_validation(label, df, golden_params[label])

if __name__ == "__main__":
    main()
