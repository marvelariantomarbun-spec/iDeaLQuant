# -*- coding: utf-8 -*-
"""Aynı bar'da çıkış + giriş durumlarını analiz et"""

import pandas as pd
from datetime import datetime, timedelta

# IdealData
ideal = pd.read_csv('d:/Projects/IdealQuant/data/ideal_signals_2_Nolu_Strateji_200000Bar.csv', sep=';', encoding='utf-8-sig')
ideal.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
ideal['Time'] = pd.to_datetime(ideal['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
ideal['KapanisTime'] = pd.to_datetime(ideal['KapanisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
ideal['Direction'] = ideal['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')

# Ardisik islemleri bul - ayni dakikada kapanan ve baslayan
print('=== AYNI BAR KAPANIŞ + GİRİŞ DURUMU ===')
count = 0
for i in range(1, len(ideal)):
    prev_close = ideal.loc[i-1, 'KapanisTime']
    curr_open = ideal.loc[i, 'Time']
    if prev_close == curr_open:
        count += 1
        if count <= 20:
            prev_dir = ideal.loc[i-1, 'Direction']
            curr_dir = ideal.loc[i, 'Direction']
            print(f"  {i}: Kapanış {prev_close} = Açılış {curr_open} | {prev_dir} -> {curr_dir}")

print(f"\nToplam: {count} adet aynı bar'da çıkış+giriş var")

# Uyumsuz sinyalleri kontrol et
problem_times = [
    '2025-01-15 16:30:00',
    '2025-02-25 22:38:00',
    '2025-03-10 21:08:00',
    '2025-03-17 13:28:00',
]

print("\n=== UYUMSUZ SİNYALLER ANALİZİ ===")
for t in problem_times:
    target = pd.to_datetime(t)
    # Bu zamanda açılan işlem
    match = ideal[ideal['Time'] == target]
    if len(match) > 0:
        idx = match.index[0]
        print(f"\n{t}:")
        print(f"  Açılan işlem: {match.iloc[0]['Direction']} @ {match.iloc[0]['AcilisFyt']}")
        # Önceki işlem
        if idx > 0:
            prev = ideal.loc[idx-1]
            print(f"  Önceki kapanış: {prev['KapanisTime']} | {prev['Direction']}")
            if prev['KapanisTime'] == target:
                print(f"  >>> AYNI BAR'DA ÇIKIŞ + GİRİŞ!")
