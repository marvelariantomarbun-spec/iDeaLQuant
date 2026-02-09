# -*- coding: utf-8 -*-
"""
Deep diagnostic to compare exact signal generation
"""
import pandas as pd
from datetime import datetime

# Load data
data_file = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
df = pd.read_csv(data_file, encoding='latin-1', sep=';')
df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
print(f"Loaded {len(df)} rows")

params = {
    "adx_period": 12,
    "adx_threshold": 50.0,
    "ars_k": 0.01,
    "ars_mesafe_threshold": 0.5,
    "ars_period": 11,
    "bb_avg_period": 100,
    "bb_period": 48,
    "bb_std": 1.5,
    "bb_width_multiplier": 0.9,
    "exit_score": 3,
    "filter_score_threshold": 1,
    "macdv_long": 32,
    "macdv_short": 19,
    "macdv_signal": 15,
    "macdv_threshold": 15.001,
    "min_score": 3,
    "netlot_period": 8,
    "netlot_threshold": 35.0,
    "yatay_adx_threshold": 30.0,
    "yatay_ars_bars": 5
}

# Prepare data
opens = df['Acilis'].tolist()
highs = df['Yuksek'].tolist()
lows = df['Dusuk'].tolist()
closes = df['Kapanis'].tolist()
typical = [(h+l+c)/3 for h,l,c in zip(highs, lows, closes)]

dates = []
for _, row in df.iterrows():
    date_str = f"{row['Tarih']} {row['Saat']}"
    try:
        dt = datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
    except:
        try:
            dt = datetime.strptime(date_str, "%d.%m.%Y %H:%M")
        except:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    dates.append(dt)

from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig, Signal

cfg = ScoreConfig(
    min_score=params['min_score'],
    exit_score=params['exit_score'],
    ars_period=params['ars_period'],
    ars_k=params['ars_k'],
    adx_period=params['adx_period'],
    adx_threshold=params['adx_threshold'],
    macdv_short=params['macdv_short'],
    macdv_long=params['macdv_long'],
    macdv_signal=params['macdv_signal'],
    macdv_threshold=params['macdv_threshold'],
    netlot_period=params['netlot_period'],
    netlot_threshold=params['netlot_threshold'],
    ars_mesafe_threshold=params['ars_mesafe_threshold'],
    bb_period=params['bb_period'],
    bb_std=params['bb_std'],
    bb_width_multiplier=params['bb_width_multiplier'],
    bb_avg_period=params['bb_avg_period'],
    yatay_ars_bars=params['yatay_ars_bars'],
    yatay_adx_threshold=params['yatay_adx_threshold'],
    filter_score_threshold=params['filter_score_threshold'],
    vade_tipi="ENDEKS"
)

strategy = ScoreBasedStrategy(opens, highs, lows, closes, typical, cfg, dates=dates)

# Count how often each score combination happens
score_combos = {}
entry_candidates = 0
entry_blocked_by_short = 0
entry_blocked_by_long = 0

for i in range(strategy.warmup_bars, len(closes)):
    l_score = strategy.long_scores[i]
    s_score = strategy.short_scores[i]
    yf = strategy.yatay_filtre[i]
    
    key = (l_score, s_score)
    score_combos[key] = score_combos.get(key, 0) + 1
    
    if yf == 1:
        # Check for LONG entry
        if l_score >= params['min_score']:
            entry_candidates += 1
            if s_score >= 2:  # Blocked because s_score >= 2
                entry_blocked_by_short += 1
        # Check for SHORT entry
        if s_score >= params['min_score']:
            entry_candidates += 1
            if l_score >= 2:  # Blocked because l_score >= 2
                entry_blocked_by_long += 1

print("\n=== Score Combinations (top 20) ===")
for combo, count in sorted(score_combos.items(), key=lambda x: -x[1])[:20]:
    print(f"  LScore={combo[0]}, SScore={combo[1]}: {count:,} bars")

print(f"\n=== Entry Analysis ===")
print(f"Entry candidates (score >= {params['min_score']}): {entry_candidates:,}")
print(f"Blocked by short score >= 2: {entry_blocked_by_short:,}")
print(f"Blocked by long score >= 2: {entry_blocked_by_long:,}")

# Check ADX contribution
adx_above = 0
macdv_triggered = 0
netlot_triggered = 0

for i in range(strategy.warmup_bars, len(closes)):
    if strategy.adx[i] > params['adx_threshold']:
        adx_above += 1
    
    # Check MACDV
    macdv_val = strategy.macdv[i]
    macdv_sig = strategy.macdv_sig[i]
    if macdv_val > (macdv_sig + params['macdv_threshold']) or macdv_val < (macdv_sig - params['macdv_threshold']):
        macdv_triggered += 1
    
    # Check NetLot
    netlot = strategy.netlot_ma[i]
    if abs(netlot) > params['netlot_threshold']:
        netlot_triggered += 1

total_bars = len(closes) - strategy.warmup_bars
print(f"\n=== Indicator Analysis ===")
print(f"Total bars after warmup: {total_bars:,}")
print(f"ADX > {params['adx_threshold']}: {adx_above:,} ({100*adx_above/total_bars:.1f}%)")
print(f"MACDV triggered (|diff| > {params['macdv_threshold']}): {macdv_triggered:,} ({100*macdv_triggered/total_bars:.1f}%)")
print(f"NetLot triggered (|val| > {params['netlot_threshold']}): {netlot_triggered:,} ({100*netlot_triggered/total_bars:.1f}%)")
