# -*- coding: utf-8 -*-
"""
Diagnostic script to compare Python vs IdealData backtest logic
"""
import json
from pathlib import Path
import pandas as pd

# Load data
data_file = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
df = pd.read_csv(data_file, encoding='latin-1', sep=';')
# Rename columns to standard names
df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
print(f"Loaded {len(df)} rows")

# Load the parameters from DB (Hibrit result)
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
    "macdv_long": 32,  # Fixed to integer
    "macdv_short": 19,
    "macdv_signal": 15,
    "macdv_threshold": 15.001,
    "min_score": 3,
    "netlot_period": 8,
    "netlot_threshold": 35.0,
    "yatay_adx_threshold": 30.0,
    "yatay_ars_bars": 5
}

print("\n=== Parameters ===")
for k, v in sorted(params.items()):
    print(f"  {k}: {v}")

# Run backtest with these params
from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig

# Prepare data
opens = df['Acilis'].tolist()
highs = df['Yuksek'].tolist()
lows = df['Dusuk'].tolist()
closes = df['Kapanis'].tolist()
typical = [(h+l+c)/3 for h,l,c in zip(highs, lows, closes)]

# Parse dates
from datetime import datetime
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

# Create config
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

# Create strategy
strategy = ScoreBasedStrategy(opens, highs, lows, closes, typical, cfg, dates=dates)

print(f"\n=== Strategy Initialized ===")
print(f"warmup_bars: {strategy.warmup_bars}")
print(f"vade_cooldown_bar: {strategy.vade_cooldown_bar}")

# Run backtest manually to count trades
from src.strategies.score_based import Signal
position = "FLAT"
entry_price = 0
trades = 0
long_entries = 0
short_entries = 0

for i in range(len(closes)):
    signal = strategy.get_signal(i, position, entry_price, 0)
    
    if signal == Signal.LONG and position == "FLAT":
        position = "LONG"
        entry_price = closes[i]
        trades += 1
        long_entries += 1
    elif signal == Signal.SHORT and position == "FLAT":
        position = "SHORT"
        entry_price = closes[i]
        trades += 1
        short_entries += 1
    elif signal == Signal.FLAT and position != "FLAT":
        position = "FLAT"
        entry_price = 0

print(f"\n=== Backtest Results ===")
print(f"Total Trades: {trades}")
print(f"Long Entries: {long_entries}")
print(f"Short Entries: {short_entries}")

# Count how many bars pass the yatay filtre
yatay_pass = sum(strategy.yatay_filtre)
print(f"\nYatay Filtre Pass: {yatay_pass} bars")

# Count how many bars have high scores
high_long_score = sum(1 for s in strategy.long_scores if s >= params['min_score'])
high_short_score = sum(1 for s in strategy.short_scores if s >= params['min_score'])
print(f"Long Score >= {params['min_score']}: {high_long_score} bars")
print(f"Short Score >= {params['min_score']}: {high_short_score} bars")

# Sample some scores for debugging
print(f"\n=== Sample Scores (first 100 bars after warmup) ===")
start_idx = strategy.warmup_bars
for i in range(start_idx, min(start_idx + 20, len(closes))):
    l_score = strategy.long_scores[i]
    s_score = strategy.short_scores[i]
    yf = strategy.yatay_filtre[i]
    print(f"  Bar {i}: LScore={l_score}, SScore={s_score}, YatayFiltre={yf}")
