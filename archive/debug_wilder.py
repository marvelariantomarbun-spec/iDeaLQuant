# -*- coding: utf-8 -*-
"""RSI başlangıç değeri analizi - Wilder smoothing karşılaştırması"""
import sys
sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
import numpy as np

# Veri oku
raw = pd.read_csv('d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
close = raw['Kapanis'].values.astype(np.float64)  # Double precision
high = raw['Yuksek'].values.astype(np.float64)
low = raw['Dusuk'].values.astype(np.float64)

# IdealData export
ideal = pd.read_csv('d:/Projects/IdealQuant/data/ideal_ind.csv', sep=';', decimal='.')
bar_start = int(ideal['BarNo'].iloc[0])

print("=" * 80)
print("WILDER SMOOTHING HESAPLAMA DETAYLARI")
print("=" * 80)

# RSI hesaplama - double precision ile
period = 14
n = len(close)
avg_gain = 0.0
avg_loss = 0.0

# İlk periyot için ortalama
for i in range(1, period + 1):
    change = close[i] - close[i - 1]
    if change > 0:
        avg_gain += change
    else:
        avg_loss += abs(change)

avg_gain /= period
avg_loss /= period

print(f"İlk Avg Gain (bar {period}): {avg_gain:.10f}")
print(f"İlk Avg Loss (bar {period}): {avg_loss:.10f}")

# Sonraki değerler
rsi_values = [50.0] * n
rsi_values[period] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss if avg_loss != 0 else 0)))

for i in range(period + 1, n):
    change = close[i] - close[i - 1]
    if change > 0:
        current_gain = change
        current_loss = 0.0
    else:
        current_gain = 0.0
        current_loss = abs(change)
    
    avg_gain = (avg_gain * (period - 1) + current_gain) / period
    avg_loss = (avg_loss * (period - 1) + current_loss) / period
    
    if avg_loss == 0:
        rsi_values[i] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

# Bar 189950 için karşılaştırma
print(f"\nBar {bar_start} RSI karşılaştırması:")
print(f"  IdealData: {ideal['RSI14'].iloc[0]:.4f}")
print(f"  Python (float64): {rsi_values[bar_start]:.4f}")
print(f"  Fark: {abs(ideal['RSI14'].iloc[0] - rsi_values[bar_start]):.10f}")

# ATR için aynı analiz
tr = [0.0] * n
tr[0] = high[0] - low[0]
for i in range(1, n):
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i - 1])
    lc = abs(low[i] - close[i - 1])
    tr[i] = max(hl, hc, lc)

atr_values = [0.0] * n
atr_values[period - 1] = sum(tr[:period]) / period

for i in range(period, n):
    atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

print(f"\nBar {bar_start} ATR karşılaştırması:")
print(f"  IdealData: {ideal['ATR14'].iloc[0]:.4f}")
print(f"  Python (float64): {atr_values[bar_start]:.4f}")
print(f"  Fark: {abs(ideal['ATR14'].iloc[0] - atr_values[bar_start]):.10f}")

# Yuvarlama ile test
print("\n" + "=" * 80)
print("YUVARLAMA İLE KARŞILAŞTIRMA")
print("=" * 80)

for i in range(5):
    bar = bar_start + i
    ideal_rsi = ideal['RSI14'].iloc[i]
    py_rsi = round(rsi_values[bar], 2)
    ideal_atr = ideal['ATR14'].iloc[i]
    py_atr = round(atr_values[bar], 2)
    
    print(f"Bar {bar}: RSI IdealData={ideal_rsi:.2f} Python={py_rsi:.2f} | ATR IdealData={ideal_atr:.2f} Python={py_atr:.2f}")
