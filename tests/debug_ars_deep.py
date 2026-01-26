# -*- coding: utf-8 -*-
"""
ARS Derin Analiz - Farkın tam olarak nereden başladığını bul
"""

import sys, io, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indicators.core import EMA, ATR
import math

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 1. Referans veri yükle (IdealData export)
df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_ars_v2_data.csv", sep=';')
df_ind.columns = [c.strip() for c in df_ind.columns]
for col in ['Close', 'ARS']:
    if col in df_ind.columns and df_ind[col].dtype == object:
        df_ind[col] = df_ind[col].str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
df_ind['DateTime'] = pd.to_datetime(df_ind['Date'] + ' ' + df_ind['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
df_ind = df_ind.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

# 2. Ham fiyat verisi
df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

# Veri hazırla
typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
highs = df_raw['Yuksek'].tolist()
lows = df_raw['Dusuk'].tolist()
closes = df_raw['Kapanis'].tolist()
times = df_raw['DateTime'].tolist()

# EMA ve ATR hesapla
ema = EMA(typical, 3)
atr = ATR(highs, lows, closes, 10)

# ARS'yi Python'da elle hesapla (debug için)
n = len(typical)
py_ars = [0.0] * n
py_ars[0] = ema[0]

for i in range(1, n):
    if ema[i] != 0:
        dynamic_k = (atr[i] / ema[i]) * 0.5  # ATR_Mult = 0.5
        dynamic_k = max(0.002, min(0.015, dynamic_k))  # min/max band
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

# Eşleştirme
calc_map = {t: i for i, t in enumerate(times)}

print("=" * 120)
print("ADIM ADIM ARS FARKI ANALİZ - Farkın nereden başladığını bul")
print("=" * 120)

# İlk farkın olduğu yeri bul
first_diff_found = False
first_diff_idx = -1

for ref_idx in range(len(df_ind)):
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt in calc_map:
        idx = calc_map[dt]
        diff = abs(py_ars[idx] - id_ars)
        
        if diff > 0.05 and not first_diff_found:
            first_diff_found = True
            first_diff_idx = ref_idx
            print(f"\n>>> İLK BÜYÜK FARK BULUNDU: Referans Index {ref_idx}")
            print(f"    DateTime: {dt}")
            print(f"    Python ARS: {py_ars[idx]:.6f}")
            print(f"    IdealData ARS: {id_ars:.6f}")
            print(f"    Fark: {diff:.6f}")
            break

# İlk farktan önceki 30 barı göster
if first_diff_idx > 0:
    print(f"\n{'Ref':<5} {'DateTime':<20} {'Py_ARS':<14} {'ID_ARS':<14} {'Fark':<10} {'EMA':<14} {'ATR':<10} {'DynK':<8} {'RStep':<8}")
    print("-" * 120)
    
    start_idx = max(0, first_diff_idx - 30)
    for ref_idx in range(start_idx, first_diff_idx + 10):
        if ref_idx >= len(df_ind):
            break
        row = df_ind.iloc[ref_idx]
        dt = row['DateTime']
        id_ars = row['ARS']
        
        if dt in calc_map:
            idx = calc_map[dt]
            diff = abs(py_ars[idx] - id_ars)
            
            dynamic_k = (atr[idx] / ema[idx]) * 0.5 if ema[idx] != 0 else 0.002
            dynamic_k = max(0.002, min(0.015, dynamic_k))
            round_step = max(0.01, atr[idx] * 0.1)
            
            marker = ">>>" if ref_idx == first_diff_idx else ""
            status = "❌" if diff > 0.05 else "✅" if diff > 0.01 else ""
            
            print(f"{ref_idx:<5} {str(dt):<20} {py_ars[idx]:<14.6f} {id_ars:<14.6f} {diff:<10.6f} {ema[idx]:<14.6f} {atr[idx]:<10.6f} {dynamic_k:<8.6f} {round_step:<8.4f} {status} {marker}")

print("\n" + "=" * 120)

# Önceki barla karşılaştırma
if first_diff_idx > 0:
    print("\nÖNCEKİ BAR ANALİZ:")
    print("-" * 80)
    
    prev_ref = first_diff_idx - 1
    curr_ref = first_diff_idx
    
    prev_row = df_ind.iloc[prev_ref]
    curr_row = df_ind.iloc[curr_ref]
    
    prev_dt = prev_row['DateTime']
    curr_dt = curr_row['DateTime']
    
    if prev_dt in calc_map and curr_dt in calc_map:
        prev_idx = calc_map[prev_dt]
        curr_idx = calc_map[curr_dt]
        
        print(f"Önceki bar (ref {prev_ref}):")
        print(f"  Python ARS: {py_ars[prev_idx]:.6f}")
        print(f"  IdealData ARS: {prev_row['ARS']:.6f}")
        print(f"  Fark: {abs(py_ars[prev_idx] - prev_row['ARS']):.6f}")
        
        print(f"\nŞimdiki bar (ref {curr_ref}):")
        print(f"  Python ARS: {py_ars[curr_idx]:.6f}")
        print(f"  IdealData ARS: {curr_row['ARS']:.6f}")
        print(f"  Fark: {abs(py_ars[curr_idx] - curr_row['ARS']):.6f}")
        
        # Hesaplama detayları
        curr_ema = ema[curr_idx]
        curr_atr = atr[curr_idx]
        dynamic_k = (curr_atr / curr_ema) * 0.5 if curr_ema != 0 else 0.002
        dynamic_k = max(0.002, min(0.015, dynamic_k))
        
        alt_band = curr_ema * (1 - dynamic_k)
        ust_band = curr_ema * (1 + dynamic_k)
        
        print(f"\nHesaplama Detayları:")
        print(f"  EMA: {curr_ema:.6f}")
        print(f"  ATR: {curr_atr:.6f}")
        print(f"  Dynamic K: {dynamic_k:.6f}")
        print(f"  Alt Band: {alt_band:.6f}")
        print(f"  Üst Band: {ust_band:.6f}")
        print(f"  Önceki Python ARS: {py_ars[prev_idx]:.6f}")
        print(f"  Önceki IdealData ARS: {prev_row['ARS']:.6f}")
        
        # Histerizis kontrolü
        print(f"\nHisterizis kontrolü (Python):")
        print(f"  alt_band ({alt_band:.4f}) > prev_ars ({py_ars[prev_idx]:.4f})? {alt_band > py_ars[prev_idx]}")
        print(f"  ust_band ({ust_band:.4f}) < prev_ars ({py_ars[prev_idx]:.4f})? {ust_band < py_ars[prev_idx]}")
        
        print(f"\nHisterizis kontrolü (IdealData ile):")
        print(f"  alt_band ({alt_band:.4f}) > prev_id_ars ({prev_row['ARS']:.4f})? {alt_band > prev_row['ARS']}")
        print(f"  ust_band ({ust_band:.4f}) < prev_id_ars ({prev_row['ARS']:.4f})? {ust_band < prev_row['ARS']}")

print("\n" + "=" * 120)
