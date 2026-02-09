# -*- coding: utf-8 -*-
"""Bayesian 5 deneme test"""
from datetime import datetime
import pandas as pd
from src.data.ideal_parser import load_ideal_data
from src.optimization.bayesian_optimizer import BayesianOptimizer

df = load_ideal_data(r'D:\iDeal\ChartData', 'VIP', 'X030-T', '1')
start = datetime(2024, 1, 1)
end = datetime(2026, 1, 30, 23, 59)
df = df[(df['DateTime'] >= start) & (df['DateTime'] <= end)].copy()
print(f'Veri: {len(df)} bar')

optimizer = BayesianOptimizer(df, n_trials=5, strategy_index=0)
results = optimizer.run(verbose=False)

if results:
    best = results[0] if isinstance(results, list) else results
    print(f"Net Profit: {best.get('net_profit', 0):.0f}")
    print(f"Trades: {best.get('trades', 0)}")
    print(f"PF: {best.get('pf', 0):.2f}")
else:
    print('Sonuc yok')
