# -*- coding: utf-8 -*-
"""
Tam hesaplama karşılaştırması - ref 89 ve 90 için adım adım
"""

import sys, io, os
import pandas as pd
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indicators.core import EMA, ATR

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Veri yükle
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

# ARS manuel hesapla
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

# Test: IdealData ARS değerlerini kullanarak Python hesaplama
print("=" * 120)
print("HİPOTEZ TESTİ: IdealData'nın prev_ars değerini kullanarak Python hesapla")
print("=" * 120)

calc_map = {t: i for i, t in enumerate(times)}

for ref_idx in [89, 90, 91]:
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt not in calc_map:
        continue
        
    idx = calc_map[dt]
    
    print(f"\n--- REF {ref_idx} ({dt}) ---")
    print(f"IdealData ARS: {id_ars:.6f}")
    print(f"Python ARS:    {py_ars[idx]:.6f}")
    print(f"Fark:          {abs(py_ars[idx] - id_ars):.6f}")
    
    # Bu bar için hesaplama detayları
    curr_ema = ema[idx]
    curr_atr = atr[idx]
    
    dynamic_k = (curr_atr / curr_ema) * 0.5 if curr_ema != 0 else 0.002
    dynamic_k = max(0.002, min(0.015, dynamic_k))
    
    alt_band = curr_ema * (1 - dynamic_k)
    ust_band = curr_ema * (1 + dynamic_k)
    
    # Python'un önceki değeri
    py_prev_ars = py_ars[idx - 1]
    
    # IdealData'nın önceki değeri
    if ref_idx > 0:
        id_prev_ars = df_ind.iloc[ref_idx - 1]['ARS']
    else:
        id_prev_ars = 0
    
    print(f"\nHesaplama parametreleri:")
    print(f"  EMA:        {curr_ema:.6f}")
    print(f"  ATR:        {curr_atr:.6f}")
    print(f"  Dynamic K:  {dynamic_k:.6f}")
    print(f"  Alt Band:   {alt_band:.6f}")
    print(f"  Üst Band:   {ust_band:.6f}")
    print(f"  Round Step: {max(0.01, curr_atr * 0.1):.6f}")
    
    print(f"\nÖnceki ARS değerleri:")
    print(f"  Python prev_ars:    {py_prev_ars:.6f}")
    print(f"  IdealData prev_ars: {id_prev_ars:.6f}")
    print(f"  Fark:               {abs(py_prev_ars - id_prev_ars):.6f}")
    
    # Histerizis PYTHON prev ile
    print(f"\nHisterizis (Python prev kullanarak):")
    if alt_band > py_prev_ars:
        py_raw = alt_band
        py_decision = "alt_band"
    elif ust_band < py_prev_ars:
        py_raw = ust_band
        py_decision = "ust_band"
    else:
        py_raw = py_prev_ars
        py_decision = "prev_ars"
    print(f"  Karar: {py_decision} -> raw_ars = {py_raw:.6f}")
    
    # Histerizis IDEALDATA prev ile
    print(f"\nHisterizis (IdealData prev kullanarak):")
    if alt_band > id_prev_ars:
        id_raw = alt_band
        id_decision = "alt_band"
    elif ust_band < id_prev_ars:
        id_raw = ust_band
        id_decision = "ust_band"
    else:
        id_raw = id_prev_ars
        id_decision = "prev_ars"
    print(f"  Karar: {id_decision} -> raw_ars = {id_raw:.6f}")
    
    # Yuvarlama sonuçları
    round_step = max(0.01, curr_atr * 0.1)
    py_rounded = math.floor(py_raw / round_step + 0.5) * round_step
    id_rounded = math.floor(id_raw / round_step + 0.5) * round_step
    
    print(f"\nYuvarlama sonrası:")
    print(f"  Python ARS (hesaplanan):    {py_rounded:.6f}")
    print(f"  IdealData ARS (hesaplanan): {id_rounded:.6f}")
    print(f"  IdealData ARS (gerçek):     {id_ars:.6f}")

print("\n" + "=" * 120)
print("\nSONUÇ ANALİZİ:")
print("=" * 120)

# Ref 90 için özel kontrol
ref_90 = df_ind.iloc[90]
dt_90 = ref_90['DateTime']
idx_90 = calc_map[dt_90]

# Önceki bar bilgileri
id_prev = df_ind.iloc[89]['ARS']  # 14167.98
py_prev = py_ars[idx_90 - 1]      # 14167.976065

# Ref 90 hesaplama
curr_ema = ema[idx_90]
curr_atr = atr[idx_90]
dynamic_k = max(0.002, min(0.015, (curr_atr / curr_ema) * 0.5))

alt_band = curr_ema * (1 - dynamic_k)
ust_band = curr_ema * (1 + dynamic_k)
round_step = max(0.01, curr_atr * 0.1)

print(f"\nRef 90 için kritik karşılaştırma:")
print(f"  IdealData önceki ARS: {id_prev:.6f}")
print(f"  Python önceki ARS:    {py_prev:.6f}")
print(f"  Fark:                 {abs(id_prev - py_prev):.6f}")

# Her iki prev ile hesapla
if alt_band > py_prev:
    py_decision = alt_band
elif ust_band < py_prev:
    py_decision = ust_band
else:
    py_decision = py_prev

if alt_band > id_prev:
    id_decision = alt_band
elif ust_band < id_prev:
    id_decision = ust_band
else:
    id_decision = id_prev

py_final = math.floor(py_decision / round_step + 0.5) * round_step
id_final = math.floor(id_decision / round_step + 0.5) * round_step

print(f"\nRaw ARS kararları:")
print(f"  Python prev kullanarak:    {py_decision:.6f} -> {py_final:.6f}")
print(f"  IdealData prev kullanarak: {id_decision:.6f} -> {id_final:.6f}")
print(f"\nGerçek IdealData ARS: {ref_90['ARS']:.6f}")

# EMA ve ATR farkı olabilir mi kontrol
print(f"\n\nEMA/ATR Kontrolü:")
print(f"  Python EMA: {curr_ema:.6f}")
print(f"  Python ATR: {curr_atr:.6f}")
print(f"  IdealData close: {ref_90['Close']:.6f}")
print(f"  Python close:    {closes[idx_90]:.6f}")

print("\n" + "=" * 120)
