import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ideal_parser import read_ideal_data, resample_bars
from src.indicators.core import ARS, EMA, SMA

def calculate_fitness(bars, ars_params):
    """
    Calculate a basic fitness (Net Return approximation)
    ars_params: (ema_period, k, signal_period)
    """
    ema_p, k, sig_p = ars_params
    
    # Typical price
    typical = bars['Tipik'].values
    closes = bars['Kapanis'].values
    
    # Calculate ARS
    # ARS signature: ARS(typical, ema_period=3, k=0.0123)
    ars_vals = ARS(typical.tolist(), ema_period=int(ema_p), k=k)
    ars_vals = np.array(ars_vals)
    
    # Calculate Signal Line (SMA of ARS)
    signal_line = SMA(ars_vals.tolist(), period=int(sig_p))
    signal_line = np.array(signal_line)
    
    # Simple strategy: Long when ARS > Signal, Short when ARS < Signal
    signals = np.zeros(len(ars_vals))
    signals[ars_vals > signal_line] = 1
    signals[ars_vals < signal_line] = -1
    
    # Shift signals to avoid lookahead (signal is generated at bar N, trade at N+1 Open/Close)
    # Return Close[i] / Close[i-1] - 1
    returns = np.zeros(len(closes))
    returns[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    
    # Shift signal
    shifted_signals = np.zeros(len(signals))
    shifted_signals[1:] = signals[:-1]
    
    strat_returns = shifted_signals * returns
    total_return = np.sum(strat_returns)
    return total_return

def optimize_timeframe(df_1m, timeframe_min):
    print(f"\nOptimizing for {timeframe_min}m...")
    bars = resample_bars(df_1m, timeframe_min)
    
    # Use last 6 months or so (to keep it fast but relevant)
    # Estimate bars: 6 months * 20 days * 400 bars/day = ~48000 bars
    # We take last 100,000 for robustness
    bars = bars.tail(100000)
    print(f"  Data range: {bars['DateTime'].min()} to {bars['DateTime'].max()} ({len(bars)} bars)")
    
    best_fitness = -float('inf')
    best_params = None
    
    # Search grid
    ema_periods = [3, 5, 8, 13]
    ks = [0.005, 0.010, 0.0123, 0.015, 0.020, 0.025]
    sig_periods = [5, 10, 15, 21]
    
    total_iters = len(ema_periods) * len(ks) * len(sig_periods)
    current = 0
    
    for ep in ema_periods:
        for k in ks:
            for sp in sig_periods:
                fitness = calculate_fitness(bars, (ep, k, sp))
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = (ep, k, sp)
                current += 1
                if current % 20 == 0:
                    print(f"  Progress: {current}/{total_iters}...", end="\r")
    
    print(f"\nBest for {timeframe_min}m: EMA={best_params[0]}, K={best_params[1]:.4f}, Signal={best_params[2]} | Fitness: {best_fitness:.4f}")
    return best_params

def main():
    file_path = "D:\\iDeal\\ChartData\\VIP\\01\\VIP'VIP-X030-T.01"
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return
        
    print(f"Loading data from {file_path}...")
    df_1m = read_ideal_data(file_path)
    
    results = {}
    for tf in [5, 15, 60]:
        results[tf] = optimize_timeframe(df_1m, tf)
    
    print("\n" + "="*30)
    print("FINAL OPTIMIZED PARAMETERS")
    print("="*30)
    for tf, params in results.items():
        print(f"{tf}m: EMA={params[0]}, K={params[1]:.4f}, Signal={params[2]}")

if __name__ == "__main__":
    main()
