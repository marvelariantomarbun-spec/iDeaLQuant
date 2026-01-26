# -*- coding: utf-8 -*-
"""
Vade Geçişi Gap Kontrolü
Sapmanın olduğu tarihlerde vade geçişi var mı?
"""

import sys, io, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ham fiyat verisi
df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

# Referans veri
df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_ars_v2_data.csv", sep=';')
df_ind.columns = [c.strip() for c in df_ind.columns]
for col in ['Close', 'ARS']:
    if col in df_ind.columns and df_ind[col].dtype == object:
        df_ind[col] = df_ind[col].str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
df_ind['DateTime'] = pd.to_datetime(df_ind['Date'] + ' ' + df_ind['Time'], format='%d.%m.%Y %H:%M', errors='coerce')

print("=" * 120)
print("VADE GEÇİŞİ GAP KONTROLÜ - Index 3447 Civarı (22 Ocak 2026)")
print("=" * 120)

# Index 3447'deki tarihi bul
target_idx = 3447
target_row = df_ind.iloc[target_idx]
target_date = target_row['DateTime']

print(f"\nHedef Index: {target_idx}")
print(f"Hedef DateTime: {target_date}")
print(f"Hedef ARS (IdealData): {target_row['ARS']:.4f}")

# O tarih civarındaki barları göster (+-20 bar)
print(f"\n{'Index':<6} {'DateTime':<20} {'Close':<10} {'Gap':<10} {'Vade?':<6}")
print("-" * 120)

calc_map = {dt: i for i, dt in enumerate(df_raw['DateTime'].tolist())}

for i in range(max(0, target_idx - 20), min(len(df_ind), target_idx + 20)):
    row = df_ind.iloc[i]
    dt = row['DateTime']
    
    if dt in calc_map:
        idx = calc_map[dt]
        close = df_raw.iloc[idx]['Kapanis']
        
        # Gap kontrolü (önceki bar ile fiyat farkı)
        gap = ""
        vade_marker = ""
        
        if idx > 0:
            prev_close = df_raw.iloc[idx-1]['Kapanis']
            price_gap = abs(close - prev_close)
            gap_pct = (price_gap / prev_close) * 100
            
            # %0.5'ten büyük gap varsa işaretle
            if gap_pct > 0.5:
                gap = f"{price_gap:.2f} ({gap_pct:.2f}%)"
                
            # Vade geçişi kontrolü (18:09 sonrası → 19:00+ geçiş)
            prev_dt = df_raw.iloc[idx-1]['DateTime']
            curr_hour = dt.hour
            prev_hour = prev_dt.hour
            curr_day = dt.day
            prev_day = prev_dt.day
            
            # Çift ay sonu + saat geçişi
            is_even_month = dt.month % 2 == 0
            hour_jump = (prev_hour >= 18 and curr_hour >= 19) or (curr_day != prev_day)
            
            if is_even_month and hour_jump and gap_pct > 0.3:
                vade_marker = "VADE?"
            elif hour_jump and gap_pct > 0.3:
                vade_marker = "GAP"
                
        marker = ">>>" if i == target_idx else ""
        print(f"{i:<6} {str(dt):<20} {close:<10.2f} {gap:<10} {vade_marker:<6} {marker}")

print("=" * 120)

# Ocak ayında vade geçişi olmamalı (tek ay), ama Aralık sonunda olmalı
# 31 Aralık 2025 civarına bakalım
print("\nARALIK 2025 VADE SONU KONTROLÜ:")
print("-" * 120)

dec_vade = df_raw[(df_raw['DateTime'] >= '2025-12-30') & (df_raw['DateTime'] <= '2026-01-02')]
print(f"\n{'DateTime':<20} {'Close':<10} {'Gap':<15}")
print("-" * 50)

for idx, row in dec_vade.iterrows():
    dt = row['DateTime']
    close = row['Kapanis']
    
    if idx > 0:
        prev_close = df_raw.iloc[idx-1]['Kapanis']
        price_gap = abs(close - prev_close)
        gap_pct = (price_gap / prev_close) * 100
        gap_str = f"{price_gap:.2f} ({gap_pct:.2f}%)" if gap_pct > 0.3 else ""
        print(f"{str(dt):<20} {close:<10.2f} {gap_str:<15}")

print("=" * 120)
