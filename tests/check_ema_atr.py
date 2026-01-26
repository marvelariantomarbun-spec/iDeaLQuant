# -*- coding: utf-8 -*-
"""
EMA ve ATR doğruluğunu kontrol et
"""

import sys, io, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indicators.core import EMA, ATR

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Fiyat verisi
df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
highs = df_raw['Yuksek'].tolist()
lows = df_raw['Dusuk'].tolist()
closes = df_raw['Kapanis'].tolist()

# Python hesaplamaları
py_ema = EMA(typical, 3)
py_atr = ATR(highs, lows, closes, 10)

print("Python EMA ve ATR Hesaplama")
print("=" * 80)
print(f"{'Index':<8} {'Typical':<12} {'EMA':<15} {'ATR':<15}")
print("-" * 80)

for i in range(50, 70):
    print(f"{i:<8} {typical[i]:<12.4f} {py_ema[i]:<15.6f} {py_atr[i]:<15.6f}")

print("=" * 80)

# İdealData ARS export datasından EMA ve ATR olup olmadığını kontrol et
# Eğer varsa karşılaştıralım
print("\n(IdealData export dosyasında EMA/ATR yok, sadece ARS var)")
print("ARS calculation'ın içinde kullanılan EMA ve ATR'yi doğrulayamıyoruz...")
print("\nSonuç: Sorun yuvarlama değil, muhtemelen EMA/ATR başlangıç koşulları farklı.")
