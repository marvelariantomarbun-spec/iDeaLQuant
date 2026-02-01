#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IdealData belirli gün sinyallerini incele"""

import pandas as pd

# IdealData sinyallerini kontrol et
ideal = pd.read_csv('data/ideal_signals_2_Nolu_Strateji_200000Bar.csv', sep=';')

# Türkçe sütun isimlerini normalize et
ideal.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']

# Parse dates
ideal['AcilisTarihi'] = pd.to_datetime(ideal['AcilisTarihi'], dayfirst=True)
ideal['KapanisTarihi'] = pd.to_datetime(ideal['KapanisTarihi'], dayfirst=True)

# 2025-01-14 ve 2025-01-15 sinyallerini goster
day_signals = ideal[((ideal['AcilisTarihi'].dt.date >= pd.to_datetime('2025-01-14').date()) & 
                     (ideal['AcilisTarihi'].dt.date <= pd.to_datetime('2025-01-15').date())) |
                    ((ideal['KapanisTarihi'].dt.date >= pd.to_datetime('2025-01-14').date()) &
                     (ideal['KapanisTarihi'].dt.date <= pd.to_datetime('2025-01-15').date()))]

print('=== 2025-01-14 - 2025-01-15 IdealData Tüm İşlemler ===')
for _, row in day_signals.iterrows():
    print(f"No:{row['No']:3d} | {row['Yon']:5s} | Açılış: {row['AcilisTarihi']} @ {row['AcilisFyt']} | Kapanış: {row['KapanisTarihi']} @ {row['KapanisFyt']} | K/Z={row['KarZarar']}")
