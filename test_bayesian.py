# -*- coding: utf-8 -*-
"""Bayesian düzgün test"""
from datetime import datetime
import pandas as pd
from src.data.ideal_parser import load_ideal_data
from src.optimization.bayesian_optimizer import BayesianOptimizer

print("Veri yükleniyor...")
df = load_ideal_data(r'D:\iDeal\ChartData', 'VIP', 'X030-T', '1')
start = datetime(2024, 1, 1)
end = datetime(2026, 1, 30, 23, 59)
df = df[(df['DateTime'] >= start) & (df['DateTime'] <= end)].copy()
print(f'Veri: {len(df)} bar')

print("\nBayesian optimizer başlatılıyor (10 deneme, Strateji 1)...")
optimizer = BayesianOptimizer(df, n_trials=10, strategy_index=0)
result = optimizer.run(verbose=True)

print("\n" + "=" * 60)
if result.get('best_result'):
    best = result['best_result']
    print("=== EN İYİ SONUÇ ===")
    print(f"Net Profit: {best.get('net_profit', 0):.0f}")
    print(f"Trades: {best.get('trades', 0)}")
    print(f"PF: {best.get('pf', 0):.2f}")
    print(f"Max DD: {best.get('max_dd', 0):.0f}")
else:
    print('Sonuç yok')
print("=" * 60)
