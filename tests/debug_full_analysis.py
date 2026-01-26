# -*- coding: utf-8 -*-
"""
Tam 5000 bar analizi - nerede hala fark var?
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

print("=" * 100)
print("TÜM 5000 BAR İÇİN ANALİZ")
print("=" * 100)

# İstatistikler
diffs = []
bad_count = 0
first_bad = None

for ref_idx in range(len(df_ind)):
    row = df_ind.iloc[ref_idx]
    dt = row['DateTime']
    id_ars = row['ARS']
    
    if dt in calc_map:
        idx = calc_map[dt]
        diff = abs(py_ars[idx] - id_ars)
        diffs.append(diff)
        
        if diff > 0.05:
            bad_count += 1
            if first_bad is None:
                first_bad = (ref_idx, dt, py_ars[idx], id_ars, diff)

print(f"Toplam karşılaştırılan bar: {len(diffs)}")
print(f"Max fark: {max(diffs):.6f}")
print(f"Ortalama fark: {sum(diffs)/len(diffs):.6f}")
print(f"Kötü bar sayısı (>0.05): {bad_count}")

if first_bad:
    print(f"\nİlk kötü bar:")
    print(f"  Ref: {first_bad[0]}")
    print(f"  DateTime: {first_bad[1]}")
    print(f"  Python ARS: {first_bad[2]:.6f}")
    print(f"  IdealData ARS: {first_bad[3]:.6f}")
    print(f"  Fark: {first_bad[4]:.6f}")
else:
    print("\n✅ TÜM BARLAR BAŞARILI! (< 0.05 fark)")

# Dağılım
print("\nFark dağılımı:")
print(f"  < 0.001: {len([d for d in diffs if d < 0.001])}")
print(f"  0.001 - 0.01: {len([d for d in diffs if 0.001 <= d < 0.01])}")
print(f"  0.01 - 0.05: {len([d for d in diffs if 0.01 <= d < 0.05])}")
print(f"  >= 0.05: {len([d for d in diffs if d >= 0.05])}")

print("=" * 100)
