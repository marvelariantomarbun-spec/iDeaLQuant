#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""İlk uyumsuzluğu bul"""

import pandas as pd

# IdealData sinyallerini yukle
ideal = pd.read_csv('data/ideal_signals_2_Nolu_Strateji_200000Bar.csv', sep=';')
ideal.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
ideal['AcilisTarihi'] = pd.to_datetime(ideal['AcilisTarihi'], dayfirst=True)

# Python sinyallerini yukle
py = pd.read_csv('tests/python_signals.csv', sep=';')
py['DateTime'] = pd.to_datetime(py['Tarih'] + ' ' + py['Saat'], dayfirst=True)

# IdealData'dan giriş sinyallerini çıkar
ideal_entries = []
for _, row in ideal.iterrows():
    signal = 'A' if row['Yon'].strip() == 'Alış' else 'S'
    ideal_entries.append({
        'DateTime': row['AcilisTarihi'],
        'Signal': signal,
        'No': row['No']
    })
ideal_entries_df = pd.DataFrame(ideal_entries)

# Python'dan giriş sinyallerini çıkar
py_entries = py[py['Sinyal'].isin(['A', 'S'])][['DateTime', 'Sinyal']].copy()
py_entries.columns = ['DateTime', 'Signal']

print(f'IdealData giriş sayısı: {len(ideal_entries_df)}')
print(f'Python giriş sayısı: {len(py_entries)}')

# İlk 50 sinyali karşılaştır
print('\n=== İLK 50 SİNYAL KARŞILAŞTIRMASI ===')
print(f'{"No":5s} | {"IdealData Tarih":22s} | {"ID Sinyal":8s} | {"Python Tarih":22s} | {"Py Sinyal":8s} | {"Eşleşme":8s}')
print('-' * 100)

tolerance = pd.Timedelta(minutes=2)
for i in range(min(50, len(ideal_entries_df), len(py_entries))):
    id_row = ideal_entries_df.iloc[i]
    py_row = py_entries.iloc[i]
    
    time_diff = abs(id_row['DateTime'] - py_row['DateTime'])
    time_match = time_diff <= tolerance
    signal_match = id_row['Signal'] == py_row['Signal']
    
    match_status = '✓' if (time_match and signal_match) else '✗'
    
    print(f'{i+1:5d} | {str(id_row["DateTime"]):22s} | {id_row["Signal"]:8s} | {str(py_row["DateTime"]):22s} | {py_row["Signal"]:8s} | {match_status:8s}')
    
    if not (time_match and signal_match):
        print(f'      ^^^ İLK UYUMSUZLUK: Fark={time_diff}, Sinyal={id_row["Signal"]} vs {py_row["Signal"]}')
        break
