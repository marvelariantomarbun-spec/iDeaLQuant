# -*- coding: utf-8 -*-
"""RSI ve ATR fark analizi"""
import sys
sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
import numpy as np
from indicators.core import RSI, ATR

# Veri oku
ideal = pd.read_csv('d:/Projects/IdealQuant/data/ideal_ind.csv', sep=';', decimal='.')
raw = pd.read_csv('d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

close = raw['Kapanis'].values.astype(float)
high = raw['Yuksek'].values.astype(float)
low = raw['Dusuk'].values.astype(float)

py_rsi = np.array(RSI(close, 14))
py_atr = np.array(ATR(high, low, close, 14))

bar_indices = ideal['BarNo'].values.astype(int)

print("=" * 80)
print("RSI14 DETAYLI KARŞILAŞTIRMA")
print("=" * 80)
print(f"{'Bar':>8} | {'IdealData':>12} | {'Python':>12} | {'Fark':>10} | {'Fark %':>10}")
print("-" * 80)

for i in range(min(10, len(bar_indices))):
    bar = bar_indices[i]
    ideal_val = ideal['RSI14'].iloc[i]
    py_val = py_rsi[bar]
    diff = abs(ideal_val - py_val)
    pct = (diff / ideal_val * 100) if ideal_val != 0 else 0
    print(f"{bar:>8} | {ideal_val:>12.4f} | {py_val:>12.4f} | {diff:>10.4f} | {pct:>9.4f}%")

print("\n" + "=" * 80)
print("ATR14 DETAYLI KARŞILAŞTIRMA")
print("=" * 80)
print(f"{'Bar':>8} | {'IdealData':>12} | {'Python':>12} | {'Fark':>10} | {'Fark %':>10}")
print("-" * 80)

for i in range(min(10, len(bar_indices))):
    bar = bar_indices[i]
    ideal_val = ideal['ATR14'].iloc[i]
    py_val = py_atr[bar]
    diff = abs(ideal_val - py_val)
    pct = (diff / ideal_val * 100) if ideal_val != 0 else 0
    print(f"{bar:>8} | {ideal_val:>12.4f} | {py_val:>12.4f} | {diff:>10.4f} | {pct:>9.4f}%")

# En büyük farkı bul
print("\n" + "=" * 80)
print("EN BÜYÜK FARKLAR")
print("=" * 80)

rsi_diffs = []
atr_diffs = []
for i, bar in enumerate(bar_indices):
    rsi_diffs.append((bar, abs(ideal['RSI14'].iloc[i] - py_rsi[bar]), ideal['RSI14'].iloc[i], py_rsi[bar]))
    atr_diffs.append((bar, abs(ideal['ATR14'].iloc[i] - py_atr[bar]), ideal['ATR14'].iloc[i], py_atr[bar]))

rsi_max = max(rsi_diffs, key=lambda x: x[1])
atr_max = max(atr_diffs, key=lambda x: x[1])

print(f"RSI Max Fark: Bar {rsi_max[0]}, Fark: {rsi_max[1]:.6f}, IdealData: {rsi_max[2]:.4f}, Python: {rsi_max[3]:.4f}")
print(f"ATR Max Fark: Bar {atr_max[0]}, Fark: {atr_max[1]:.6f}, IdealData: {atr_max[2]:.4f}, Python: {atr_max[3]:.4f}")
