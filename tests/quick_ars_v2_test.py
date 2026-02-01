# -*- coding: utf-8 -*-
"""ARS Trend v2 Hızlı Sinyal Testi"""

import sys
import os
sys.path.insert(0, 'd:/Projects/IdealQuant/src')
sys.path.insert(0, 'd:/Projects/IdealQuant')

import pandas as pd
from datetime import datetime

# Fiyat verisini yukle
csv_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', low_memory=False)
df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')

print(f"Veri boyutu: {len(df)} bar")
print(f"Tarih araligi: {df['DateTime'].iloc[0]} - {df['DateTime'].iloc[-1]}")

from src.strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2, Signal

opens = df['Acilis'].values.tolist()
highs = df['Yuksek'].values.tolist()
lows = df['Dusuk'].values.tolist()
closes = df['Kapanis'].values.tolist()
typical = ((df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3).tolist()
times = df['DateTime'].tolist()
volumes = df['Lot'].values.tolist()

config = StrategyConfigV2(
    ars_ema_period=3, ars_atr_period=10, ars_atr_mult=0.5,
    ars_min_band=0.002, ars_max_band=0.015,
    momentum_period=5, breakout_period=10,
    mfi_period=14, mfi_hhv_period=14, mfi_llv_period=14,
    volume_hhv_period=14, volume_llv_period=14,
    kar_al_pct=3.0, iz_stop_pct=1.5, vade_tipi='ENDEKS'
)

print("Strateji olusturuluyor...")
strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, volumes, config)

signals = []
current_pos = 'FLAT'
entry_price = 0.0
extreme_price = 0.0

for i in range(len(closes)):
    if current_pos == 'LONG':
        extreme_price = max(extreme_price, highs[i])
    elif current_pos == 'SHORT':
        extreme_price = min(extreme_price if extreme_price > 0 else highs[i], lows[i])
    
    sig = strategy.get_signal(i, current_pos, entry_price, extreme_price)
    
    if sig != Signal.NONE:
        if sig == Signal.LONG:
            signals.append({'bar': i, 'time': times[i], 'signal': 'LONG', 'price': closes[i]})
            current_pos = 'LONG'
            entry_price = closes[i]
            extreme_price = highs[i]
        elif sig == Signal.SHORT:
            signals.append({'bar': i, 'time': times[i], 'signal': 'SHORT', 'price': closes[i]})
            current_pos = 'SHORT'
            entry_price = closes[i]
            extreme_price = lows[i]
        elif sig == Signal.FLAT:
            signals.append({'bar': i, 'time': times[i], 'signal': 'FLAT', 'price': closes[i]})
            current_pos = 'FLAT'
            entry_price = 0.0
            extreme_price = 0.0

print(f"\n{'='*60}")
print(f"SONUC: Toplam {len(signals)} sinyal uretildi")
print(f"{'='*60}")

long_count = sum(1 for s in signals if s['signal'] == 'LONG')
short_count = sum(1 for s in signals if s['signal'] == 'SHORT')
flat_count = sum(1 for s in signals if s['signal'] == 'FLAT')
print(f"LONG: {long_count}, SHORT: {short_count}, FLAT: {flat_count}")

if signals:
    print("\nIlk 5 sinyal:")
    for s in signals[:5]:
        print(f"  Bar {s['bar']:6d} | {s['time']} | {s['signal']:5s} | {s['price']:.2f}")
    
    print("\nSon 5 sinyal:")
    for s in signals[-5:]:
        print(f"  Bar {s['bar']:6d} | {s['time']} | {s['signal']:5s} | {s['price']:.2f}")
else:
    print("\n!!! HIC SINYAL URETILMEDI !!!")
