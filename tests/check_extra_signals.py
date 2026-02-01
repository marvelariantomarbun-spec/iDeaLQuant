#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python'un IdealData'da olmayan fazla sinyallerini incele"""

import sys
import os
sys.path.insert(0, 'd:/Projects/IdealQuant/src')
sys.path.insert(0, 'd:/Projects/IdealQuant')
import pandas as pd
from datetime import timedelta
from strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2, Signal

# IdealData sinyalleri
ideal = pd.read_csv('data/ideal_signals_2_Nolu_Strateji_200000Bar.csv', sep=';')
ideal.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
ideal['Time'] = pd.to_datetime(ideal['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
ideal['Direction'] = ideal['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')

# Python stratejisini çalıştır
df = pd.read_csv('data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254', low_memory=False)
df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')

config = StrategyConfigV2(
    ars_ema_period=3, ars_atr_period=10, ars_atr_mult=0.5,
    ars_min_band=0.002, ars_max_band=0.015,
    momentum_period=5, breakout_period=10,
    mfi_period=14, mfi_hhv_period=14, mfi_llv_period=14,
    volume_hhv_period=14, volume_llv_period=14,
    kar_al_pct=3.0, iz_stop_pct=1.5, vade_tipi='ENDEKS'
)

opens = df['Acilis'].values.tolist()
highs = df['Yuksek'].values.tolist()
lows = df['Dusuk'].values.tolist()
closes = df['Kapanis'].values.tolist()
typical = ((df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3).tolist()
times = df['DateTime'].tolist()
volumes = df['Lot'].values.tolist()

strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, volumes, config)

# Python sinyallerini topla
python_signals = []
current_pos = 'FLAT'
entry_price = 0.0
extreme_price = 0.0

for i in range(len(closes)):
    if current_pos == 'LONG':
        extreme_price = max(extreme_price, highs[i])
    elif current_pos == 'SHORT':
        extreme_price = min(extreme_price if extreme_price > 0 else highs[i], lows[i])
    
    sig = strategy.get_signal(i, current_pos, entry_price, extreme_price)
    
    if sig in [Signal.LONG, Signal.SHORT]:
        python_signals.append({
            'time': times[i],
            'direction': 'LONG' if sig == Signal.LONG else 'SHORT',
            'price': closes[i]
        })
        if sig == Signal.LONG:
            current_pos = 'LONG'
            entry_price = closes[i]
            extreme_price = highs[i]
        else:
            current_pos = 'SHORT'
            entry_price = closes[i]
            extreme_price = lows[i]
    elif sig == Signal.FLAT:
        current_pos = 'FLAT'
        entry_price = 0.0
        extreme_price = 0.0

py_df = pd.DataFrame(python_signals)

# Python sinyallerinden IdealData'da olmayanları bul
tolerance = timedelta(minutes=1)
extra_signals = []

for _, py_row in py_df.iterrows():
    py_time = py_row['time']
    py_dir = py_row['direction']
    
    # IdealData'da eşleşme ara
    mask = (ideal['Time'] >= py_time - tolerance) & (ideal['Time'] <= py_time + tolerance) & \
           (ideal['Direction'] == py_dir)
    
    if len(ideal[mask]) == 0:
        extra_signals.append({
            'time': py_time,
            'direction': py_dir,
            'price': py_row['price']
        })

print(f"Python toplam giriş: {len(py_df)}")
print(f"IdealData toplam giriş: {len(ideal)}")
print(f"Python fazla sinyal: {len(extra_signals)}")

# İlk 30 fazla sinyali göster
print("\n=== İLK 30 FAZLA SİNYAL ===")
for i, s in enumerate(extra_signals[:30]):
    print(f"{i+1:3d}. {s['time']} | {s['direction']:5s} | {s['price']:.2f}")

# Günlük dağılım
extra_df = pd.DataFrame(extra_signals)
if len(extra_df) > 0:
    extra_df['date'] = pd.to_datetime(extra_df['time']).dt.date
    daily = extra_df.groupby('date').size()
    print(f"\n=== GÜNLÜK DAĞILIM (En fazla 20 gün) ===")
    top_days = daily.sort_values(ascending=False).head(20)
    for date, count in top_days.items():
        print(f"{date}: {count} fazla sinyal")
