# -*- coding: utf-8 -*-
"""Quick test runner for smart optimizer"""
import sys
import os

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Starting...", flush=True)

import pandas as pd
import numpy as np
from time import time

print("Imports OK", flush=True)

from src.indicators.core import EMA, ATR, ADX, SMA, ARS, RVI, Qstick, NetLot

print("Indicators imported", flush=True)

# Load data
csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
print(f"Loading: {csv_path}", flush=True)

df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']
for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
df.dropna(inplace=True)

print(f"Data loaded: {len(df)} bars", flush=True)

# Quick ARS test
tipik = df['Tipik'].values.tolist()
ars = ARS(tipik, 10, 0.01)
print(f"ARS calculated: {len(ars)} values", flush=True)
print(f"ARS sample [100:105]: {ars[100:105]}", flush=True)

# Quick optimization (small grid)
print("\n=== Mini Optimization Test ===", flush=True)

ars_emas = [3, 5, 7]
ars_ks = [0.005, 0.01, 0.015]

closes = df['Kapanis'].values
best_pf = 0
total_tests = 0
start = time()

for p in ars_emas:
    for k in ars_ks:
        ars_vals = np.array(ARS(tipik, int(p), float(k)))
        
        # Simple logic: Long when price > ARS, Short when price < ARS
        signals = np.zeros(len(closes))
        signals[closes > ars_vals] = 1
        signals[closes < ars_vals] = -1
        
        # Very simple PnL (no real backtest)
        pos = 0
        pnl = 0
        for i in range(1, len(signals)):
            if pos == 0 and signals[i] == 1:
                pos = 1
                entry = closes[i]
            elif pos == 0 and signals[i] == -1:
                pos = -1
                entry = closes[i]
            elif pos == 1 and signals[i] != 1:
                pnl += closes[i] - entry
                pos = 0
            elif pos == -1 and signals[i] != -1:
                pnl += entry - closes[i]
                pos = 0
        
        total_tests += 1
        print(f"  ARS({p}, {k}) -> PnL: {pnl:.0f}", flush=True)

elapsed = time() - start
print(f"\nDone! {total_tests} tests in {elapsed:.1f} sec", flush=True)
