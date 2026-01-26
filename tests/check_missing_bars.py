# -*- coding: utf-8 -*-
"""
Eksik bar kontrolü - IdealData export'unda eksik bar var mı?
"""

import sys, io, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# IdealData export
df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_ars_v2_data.csv", sep=';')
df_ind.columns = [c.strip() for c in df_ind.columns]
df_ind['DateTime'] = pd.to_datetime(df_ind['Date'] + ' ' + df_ind['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
df_ind = df_ind.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

# Ham veri
df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

# Ref 89 ve 90 arasındaki barları kontrol et
dt_89 = df_ind.iloc[89]['DateTime']
dt_90 = df_ind.iloc[90]['DateTime']

print("=" * 100)
print("EKSİK BAR KONTROLÜ")
print("=" * 100)

print(f"\nIdealData Ref 89: {dt_89}")
print(f"IdealData Ref 90: {dt_90}")

# Python raw verisinde bu iki tarih arasında kaç bar var?
raw_between = df_raw[(df_raw['DateTime'] > dt_89) & (df_raw['DateTime'] < dt_90)]
print(f"\nPython'da bu iki tarih arasındaki bar sayısı: {len(raw_between)}")

if len(raw_between) > 0:
    print("\n>>> EKSİK BARLAR BULUNDU!")
    print(f"{'DateTime':<25} {'Close':<12}")
    print("-" * 40)
    for _, row in raw_between.iterrows():
        print(f"{str(row['DateTime']):<25} {row['Kapanis']:<12.2f}")

# Raw veride ref 89 ve 90'ın indexlerini bul
idx_89 = df_raw[df_raw['DateTime'] == dt_89].index[0] if len(df_raw[df_raw['DateTime'] == dt_89]) > 0 else -1
idx_90 = df_raw[df_raw['DateTime'] == dt_90].index[0] if len(df_raw[df_raw['DateTime'] == dt_90]) > 0 else -1

print(f"\nRaw veri indexleri:")
print(f"  dt_89 ({dt_89}) -> raw index: {idx_89}")
print(f"  dt_90 ({dt_90}) -> raw index: {idx_90}")
print(f"  Aradaki bar sayısı: {idx_90 - idx_89 - 1}")

# Daha geniş aralıkta eksik bar kontrolü (ref 85-95)
print("\n" + "=" * 100)
print("GENİŞ ARALIKTA EKSİK BAR KONTROLÜ (Ref 85-95)")
print("=" * 100)

total_missing = 0
for i in range(85, 95):
    if i >= len(df_ind) - 1:
        break
    dt_curr = df_ind.iloc[i]['DateTime']
    dt_next = df_ind.iloc[i+1]['DateTime']
    
    # Raw veride ara
    idx_curr = df_raw[df_raw['DateTime'] == dt_curr].index
    idx_next = df_raw[df_raw['DateTime'] == dt_next].index
    
    if len(idx_curr) > 0 and len(idx_next) > 0:
        gap = idx_next[0] - idx_curr[0] - 1
        if gap > 0:
            total_missing += gap
            print(f"Ref {i} -> {i+1}: {gap} eksik bar ({dt_curr} -> {dt_next})")
            # Eksik barları göster
            missing = df_raw[(df_raw['DateTime'] > dt_curr) & (df_raw['DateTime'] < dt_next)]
            for _, row in missing.iterrows():
                print(f"    EKSIK: {row['DateTime']} Close={row['Kapanis']:.2f}")

print(f"\nToplam eksik bar: {total_missing}")

# İlk 100 ref için toplam eksik bar
print("\n" + "=" * 100)
print("İLK 100 REF İÇİN TOPLAM EKSİK BAR")
print("=" * 100)

total_missing_100 = 0
for i in range(min(99, len(df_ind) - 1)):
    dt_curr = df_ind.iloc[i]['DateTime']
    dt_next = df_ind.iloc[i+1]['DateTime']
    
    idx_curr = df_raw[df_raw['DateTime'] == dt_curr].index
    idx_next = df_raw[df_raw['DateTime'] == dt_next].index
    
    if len(idx_curr) > 0 and len(idx_next) > 0:
        gap = idx_next[0] - idx_curr[0] - 1
        if gap > 0:
            total_missing_100 += gap

print(f"İlk 100 referans bar arasında toplam {total_missing_100} eksik bar var!")

print("\n" + "=" * 100)
