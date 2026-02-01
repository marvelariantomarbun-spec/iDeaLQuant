#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python sinyallerini incele"""

import pandas as pd

# Python sinyallerini yukle
py = pd.read_csv('tests/python_signals.csv', sep=';')
py['DateTime'] = pd.to_datetime(py['Tarih'] + ' ' + py['Saat'], dayfirst=True)

# 2025-01-15 ve oncesi sinyalleri (son 30)
day = py[(py['DateTime'] >= '2025-01-14 00:00:00') & (py['DateTime'] <= '2025-01-15 23:59:59')]
print('=== Python 2025-01-14 - 2025-01-15 Sinyalleri ===')
for _, row in day.iterrows():
    print(f"{row['DateTime']} | {row['Sinyal']} | C={row['Kapanis']:.2f} | Pos={row['Pozisyon']}")
