# Debug script for bar matching
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'd:/Projects/IdealQuant/src')
from indicators.core import SMA

# IdealData export
ideal = pd.read_csv('d:/Projects/IdealQuant/data/ideal_ind.csv', sep=';', decimal='.')
print('IdealData ilk 5 satir:')
print(ideal.head())

# Orijinal CSV
raw = pd.read_csv('d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
print('\nOrijinal CSV son 5 satir:')
print(raw.tail())

# Bar 189950'deki degerler
bar_start = int(ideal['BarNo'].iloc[0])
print(f'\nBar {bar_start} verileri:')
print(f"  IdealData Kapanis: {ideal['Kapanis'].iloc[0]}")
print(f"  CSV Kapanis: {raw['Kapanis'].iloc[bar_start]}")

# SMA karsilastirma
close_full = raw['Kapanis'].values.astype(float)
py_sma_full = np.array(SMA(close_full, 20))
print(f"\nSMA20 karsilastirma (bar {bar_start}):")
print(f"  IdealData SMA20: {ideal['SMA20'].iloc[0]}")
print(f"  Python SMA20 (full hesap): {py_sma_full[bar_start]}")
