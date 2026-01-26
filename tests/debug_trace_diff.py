# -*- coding: utf-8 -*-
"""
Tam takip: İlk farkın kaynağını bul
Ref 0'dan başlayarak Python vs IdealData ARS farkını takip et
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

calc_map = {t: i for i, t in enumerate(times)}

print("=" * 130)
print("İLK 100 REF İÇİN FARK TAKİBİ - Fark nereden başlıyor?")
print("=" * 130)
print(f"{'Ref':<5} {'Py_ARS':<16} {'ID_ARS':<16} {'Fark':<12} {'RStep':<10} {'Durum'}")
print("-" * 130)

first_significant_diff = None

for ref_idx in range(min(100, len(df_ind))):
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt in calc_map:
        idx = calc_map[dt]
        diff = abs(py_ars[idx] - id_ars)
        round_step = max(0.01, atr[idx] * 0.1)
        
        # Fark durumu
        if diff < 0.001:
            status = "✅ Perfect"
        elif diff < 0.01:
            status = "✓ Good"
        elif diff < 0.05:
            status = "⚠️ Small diff"
        else:
            status = f"❌ BAD {diff:.4f}"
            if first_significant_diff is None:
                first_significant_diff = ref_idx
        
        # Sadece fark > 0.001 olanları göster
        if diff > 0.001:
            print(f"{ref_idx:<5} {py_ars[idx]:<16.6f} {id_ars:<16.6f} {diff:<12.6f} {round_step:<10.4f} {status}")

print("=" * 130)

if first_significant_diff:
    print(f"\nİLK ANLAMLI FARK: Ref {first_significant_diff}")
else:
    print("\nİlk 100 bar'da anlamlı fark yok!")

# İlk 10 bar'ı detaylı göster
print("\n" + "=" * 130)
print("İLK 10 BAR DETAYLI ANALİZ")
print("=" * 130)

for ref_idx in range(min(10, len(df_ind))):
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt in calc_map:
        idx = calc_map[dt]
        
        print(f"\n--- Ref {ref_idx} (raw idx {idx}) ---")
        print(f"DateTime: {dt}")
        print(f"Python ARS:    {py_ars[idx]:.8f}")
        print(f"IdealData ARS: {id_ars:.8f}")
        print(f"Fark:          {abs(py_ars[idx] - id_ars):.8f}")
        
        if idx > 0:
            print(f"Python prev:   {py_ars[idx-1]:.8f}")
