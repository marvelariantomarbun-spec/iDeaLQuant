# -*- coding: utf-8 -*-
"""
Ref 330-340 arası detaylı analiz - fark neden başlıyor?
"""

import sys, io, os
import pandas as pd
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from indicators.core import EMA, ATR

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_ars_v2_data.csv", sep=';')
df_ind.columns = [c.strip() for c in df_ind.columns]
for col in ['Close', 'ARS']:
    if col in df_ind.columns and df_ind[col].dtype == object:
        df_ind[col] = df_ind[col].str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
df_ind['DateTime'] = pd.to_datetime(df_ind['Date'] + ' ' + df_ind['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
df_ind = df_ind.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
highs = df_raw['Yuksek'].tolist()
lows = df_raw['Dusuk'].tolist()
closes = df_raw['Kapanis'].tolist()
times = df_raw['DateTime'].tolist()

ema = EMA(typical, 3)
atr = ATR(highs, lows, closes, 10)

n = len(typical)
py_ars = [0.0] * n
py_ars[0] = ema[0]

for i in range(1, n):
    if ema[i] != 0:
        dynamic_k = (atr[i] / ema[i]) * 0.5
        dynamic_k = max(0.002, min(0.015, dynamic_k))
    else:
        dynamic_k = 0.002
    
    alt_band = ema[i] * (1 - dynamic_k)
    ust_band = ema[i] * (1 + dynamic_k)
    
    if alt_band > py_ars[i - 1]:
        raw_ars = alt_band
    elif ust_band < py_ars[i - 1]:
        raw_ars = ust_band
    else:
        raw_ars = py_ars[i - 1]
    
    round_step = max(0.01, atr[i] * 0.1)
    py_ars[i] = math.floor(raw_ars / round_step + 0.5) * round_step

calc_map = {t: i for i, t in enumerate(times)}

print("=" * 120)
print("REF 320-345 ARASI DETAYLI ANALİZ")
print("=" * 120)
print(f"{'Ref':<5} {'DateTime':<20} {'Py_ARS':<16} {'ID_ARS':<16} {'Fark':<12} {'RStep':<10} {'Durum'}")
print("-" * 120)

for ref_idx in range(320, 345):
    if ref_idx >= len(df_ind):
        break
        
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt in calc_map:
        idx = calc_map[dt]
        diff = abs(py_ars[idx] - id_ars)
        round_step = max(0.01, atr[idx] * 0.1)
        
        if diff < 0.01:
            status = "✓"
        elif diff < 0.05:
            status = "⚠️"
        else:
            status = f"❌ {diff:.4f}"
        
        print(f"{ref_idx:<5} {str(dt):<20} {py_ars[idx]:<16.6f} {id_ars:<16.6f} {diff:<12.6f} {round_step:<10.4f} {status}")

# Ref 334 özel analiz
print("\n" + "=" * 120)
print("REF 333-334 DETAYLI KARŞILAŞTIRMA")
print("=" * 120)

for ref_idx in [333, 334]:
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt in calc_map:
        idx = calc_map[dt]
        
        print(f"\n--- Ref {ref_idx} ---")
        print(f"DateTime: {dt}")
        print(f"Python ARS:    {py_ars[idx]:.8f}")
        print(f"IdealData ARS: {id_ars:.8f}")
        print(f"Fark:          {abs(py_ars[idx] - id_ars):.8f}")
        
        if idx > 0:
            print(f"\nPython prev ARS: {py_ars[idx-1]:.8f}")
            
            # Önceki IdealData değeri
            if ref_idx > 0:
                prev_row = df_ind.iloc[ref_idx-1]
                print(f"IdealData prev ARS: {prev_row['ARS']:.8f}")
                print(f"Prev fark: {abs(py_ars[idx-1] - prev_row['ARS']):.8f}")
            
            curr_ema = ema[idx]
            curr_atr = atr[idx]
            dynamic_k = max(0.002, min(0.015, (curr_atr / curr_ema) * 0.5))
            
            alt_band = curr_ema * (1 - dynamic_k)
            ust_band = curr_ema * (1 + dynamic_k)
            round_step = max(0.01, curr_atr * 0.1)
            
            print(f"\nHesaplama:")
            print(f"  EMA: {curr_ema:.6f}")
            print(f"  ATR: {curr_atr:.6f}")
            print(f"  Dynamic K: {dynamic_k:.6f}")
            print(f"  Alt Band: {alt_band:.6f}")
            print(f"  Üst Band: {ust_band:.6f}")
            print(f"  Round Step: {round_step:.6f}")
            
            # Histerizis
            if alt_band > py_ars[idx - 1]:
                py_decision = f"alt_band ({alt_band:.6f})"
            elif ust_band < py_ars[idx - 1]:
                py_decision = f"ust_band ({ust_band:.6f})"
            else:
                py_decision = f"prev_ars ({py_ars[idx - 1]:.6f})"
            
            print(f"\nPython histerizis kararı: {py_decision}")

print("\n" + "=" * 120)
