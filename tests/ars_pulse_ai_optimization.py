import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.genetic_optimizer import GeneticOptimizer, GeneticConfig
from src.optimization.bayesian_optimizer import BayesianOptimizer
from src.optimization.fitness import FitnessConfig

def load_and_filter_data(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    column_map = {
        'Tarih': 'Date', 'Saat': 'Time', 'Al': 'Acilis',
        'Yksek': 'Yuksek', 'Dk': 'Dusuk', 'Kapan': 'Kapanis', 'Lot': 'Volume'
    }
    cols = df.columns.tolist()
    if 'Al' not in cols and 'Acilis' not in cols:
        df.columns = ['Date', 'Time', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Avg', 'Volume', 'Lot']
    else:
        df = df.rename(columns=column_map)
    
    # Combine Date and Time
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    mask = (df['DateTime'] >= '2024-01-01') & (df['DateTime'] <= '2025-12-31')
    filtered_df = df.loc[mask].copy()
    
    # Ensure numeric types
    for col in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis']:
        filtered_df[col] = filtered_df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            
    filtered_df = filtered_df.sort_values('DateTime')
    return filtered_df

def run_ai_optimization(file_path, timeframe_label):
    bars = load_and_filter_data(file_path)
    if len(bars) < 100: return None
    
    print(f"\n{'='*60}")
    print(f"AI OPTIMIZATION: {timeframe_label}")
    print(f"{'='*60}")
    
    # --- ROUND 1: GENETIC (Exploration) ---
    print("\n[ROUND 1] Genetic Exploration...")
    gen_config = GeneticConfig(population_size=100, generations=20)
    gen_opt = GeneticOptimizer(bars, config=gen_config, strategy_index=2, n_parallel=os.cpu_count())
    gen_res = gen_opt.run(verbose=True)
    
    # --- ROUND 2: BAYESIAN (Precision) ---
    print("\n[ROUND 2] Bayesian Precision (Fine-tuning around best Genetic)...")
    best_gen_params = gen_res['best_params']
    
    # Narrow ranges around best genetic (±20% for numerical, ±2 for discrete)
    narrowed = {}
    from src.optimization.genetic_optimizer import STRATEGY3_PARAMS
    for name, original_vals in STRATEGY3_PARAMS.items():
        val = best_gen_params[name]
        min_orig, max_orig, step, is_int = original_vals
        if is_int:
            narrowed[name] = (max(min_orig, val - 5), min(max_orig, val + 5))
        else:
            narrowed[name] = (max(min_orig, val - 1.0), min(max_orig, val + 1.0))
            
    bay_opt = BayesianOptimizer(bars, n_trials=100, strategy_index=2, n_parallel=os.cpu_count(), narrowed_ranges=narrowed)
    bay_res = bay_opt.run(verbose=True)
    
    return bay_res['best_result'], bay_res['best_params']

def main():
    data_files = {
        '5m': r"D:\Projects\IdealQuant\data\XU030_5dk_200000bar.csv",
        #'15m': r"D:\Projects\IdealQuant\data\XU030_15dk_68081bar.csv",
        #'60m': r"D:\Projects\IdealQuant\data\XU030_60dk_32944bar.csv"
    }
    
    for label, path in data_files.items():
        res, params = run_ai_optimization(path, label)
        if res:
            print(f"\nFINAL GOLDEN PARAMS FOR {label}:")
            for k, v in params.items(): print(f"  {k}: {v}")
            print(f"  Final PF: {res['pf']:.4f}")

if __name__ == "__main__":
    main()
