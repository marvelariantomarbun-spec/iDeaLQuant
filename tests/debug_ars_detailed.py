# -*- coding: utf-8 -*-
"""
ARS v2 Debug - Farkın nereden kaynaklandığını tespit et
"""

import sys, io, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 1. Referans veri yükle
df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_ars_v2_data.csv", sep=';')
df_ind.columns = [c.strip() for c in df_ind.columns]

for col in ['Close', 'ARS', 'Momentum', 'HHV', 'LLV', 'RSI']:
    if col in df_ind.columns and df_ind[col].dtype == object:
        df_ind[col] = df_ind[col].str.replace(',', '.').apply(pd.to_numeric, errors='coerce')

df_ind['DateTime'] = pd.to_datetime(df_ind['Date'] + ' ' + df_ind['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
df_ind = df_ind.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

# 2. Ham fiyat verisi
df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

# 3. Python strateji hesapla
config = StrategyConfigV2(
    ars_ema_period = 3,
    ars_atr_period = 10,
    ars_atr_mult = 0.5,
   ars_min_band = 0.002,
    ars_max_band = 0.015,
    momentum_period = 5,
    breakout_period = 10,
    rsi_period = 14
)

opens = df_raw['Acilis'].values.tolist()
highs = df_raw['Yuksek'].values.tolist()
lows = df_raw['Dusuk'].values.tolist()
closes = df_raw['Kapanis'].values.tolist()
typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
times = df_raw['DateTime'].tolist()

strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, config)

# 4. En büyük farkın olduğu yeri bul (index 3447 civarı)
calc_map = {t: i for i, t in enumerate(times)}

print("=" * 100)
print("ARS DETAYLI KARŞILAŞTIRMA - En Büyük Farkın Olduğu Bölge (Index 3440-3455)")
print("=" * 100)
print(f"{'Index':<6} {'DateTime':<17} {'Py_ARS':<12} {'ID_ARS':<12} {'Fark':<10} {'Round_Step':<10}")
print("-" * 100)

from indicators.core import EMA, ATR

ema = EMA(typical, 3)
atr = ATR(highs, lows, closes, 10)

# İndexleri eşleştir
matched_indices = []
for i in range(len(df_ind)):
    row = df_ind.iloc[i]
    dt = row['DateTime']
    if dt in calc_map:
        idx = calc_map[dt]
        matched_indices.append((i, idx, dt, strategy.ars[idx], row['ARS']))

# Index 3440-3455 arası göster
for i_ref, idx, dt, py_ars, id_ars in matched_indices:
    if 3440 <= i_ref <= 3455:
        diff = abs(py_ars - id_ars)
        round_step = max(0.01, atr[idx] * 0.1)
        
        status = "✅" if diff < 0.05 else "❌"
        print(f"{i_ref:<6} {str(dt):<17} {py_ars:<12.4f} {id_ars:<12.4f} {diff:<10.6f} {round_step:<10.6f} {status}")

print("=" * 100)

# En büyük farkı bul
max_diff_idx = 0
max_diff_val = 0
for i_ref, idx, dt, py_ars, id_ars in matched_indices:
    diff = abs(py_ars - id_ars)
    if diff > max_diff_val:
        max_diff_val = diff
        max_diff_idx = i_ref

print(f"\nEN BÜYÜK FARK:")
print(f"Referans Index: {max_diff_idx}")
print(f"Fark: {max_diff_val:.6f}")

# O indexi detaylı göster
for i_ref, idx, dt, py_ars, id_ars in matched_indices:
    if i_ref == max_diff_idx:
        print(f"DateTime: {dt}")
        print(f"Python ARS: {py_ars:.6f}")
        print(f"IdealData ARS: {id_ars:.6f}")
        print(f"Fiyat (Close): {closes[idx]:.2f}")
        print(f"EMA: {ema[idx]:.6f}")
        print(f"ATR: {atr[idx]:.6f}")
        print(f"Round Step: {max(0.01, atr[idx] * 0.1):.6f}")
        break

print("=" * 100)
