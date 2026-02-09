# -*- coding: utf-8 -*-
"""
Debug script using IdealData Binary Parser directly
"""
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from src.data.ideal_parser import read_ideal_data

# Correct file path found by search
# Note the single quote escaping
file_path = "d:\\iDeal\\ChartData\\VIP\\01\\VIP'VIP-X030-T.01"

print(f"Reading file: {file_path}")

try:
    df = read_ideal_data(file_path)
    print(f"Loaded {len(df)} rows")
    print(f"First date: {df.iloc[0]['DateTime']}")
    print(f"Last date: {df.iloc[-1]['DateTime']}")
    
    # Filter date range (01.01.2024 - 31.12.2025) like the user app filter
    df = df[(df['DateTime'] >= '2024-01-01') & (df['DateTime'] <= '2025-12-31')]
    print(f"Filtered rows (2024-2025): {len(df)}")
    print(f"Filtered start: {df.iloc[0]['DateTime']}")
    print(f"Filtered end: {df.iloc[-1]['DateTime']}")
    
except Exception as e:
    print(f"Error reading file: {e}")
    # Try another path if failed
    file_path2 = "d:\\iDeal\\ChartData\\VIP\\01\\X030-T.01"
    try:
        df = read_ideal_data(file_path2)
        print(f"Loaded {len(df)} rows from alternative path")
    except Exception as e2:
        print(f"Error reading alternative path: {e2}")
        sys.exit(1)

# Parameters from Hybrid result (ID 33)
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

# Run backtest logic
from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig, Signal

# Prepare data lists
opens = df['Open'].tolist()
highs = df['High'].tolist()
lows = df['Low'].tolist()
closes = df['Close'].tolist()
# Typical price
typical = [(h+l+c)/3 for h,l,c in zip(highs, lows, closes)]
# Dates
dates = df['DateTime'].tolist()

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

# Simulate C# logic
def simulate_csharp_logic():
    trades = 0
    son_yon = ""
    start_idx = strategy.warmup_bars
    
    trade_list = []
    
    for i in range(start_idx, len(closes)):
        l_score = strategy.long_scores[i]
        s_score = strategy.short_scores[i]
        yf = strategy.yatay_filtre[i]
        
        c_above_ars = closes[i] > strategy.ars[i]
        c_below_ars = closes[i] < strategy.ars[i]
        
        sinyal = ""
        
        # Exit
        if son_yon == "A":
            if c_below_ars or s_score >= params['exit_score']:
                sinyal = "F"
        elif son_yon == "S":
            if c_above_ars or l_score >= params['exit_score']:
                sinyal = "F"
        
        # Entry (re-entry allowed immediately after flat)
        if sinyal == "" and son_yon != "A" and son_yon != "S":
            if yf == 1:
                if l_score >= params['min_score'] and s_score < 2:
                    sinyal = "A"
                elif s_score >= params['min_score'] and l_score < 2:
                    sinyal = "S"
            # Special case: Check re-entry in same bar if exited?
            # IdealData logic executes sequentially. If exited, son_yon becomes F.
            # Next check is entry.
            # But in C# loop example:
            # if (Sinyal == "" ...) -> If Sinyal was set to "F" above, this block is SKIPPED!
            # So NO immediate re-entry in the same bar after exit.
        
        # Position update
        if sinyal != "" and son_yon != sinyal:
            if sinyal in ("A", "S"):
                trades += 1
                trade_list.append((dates[i], sinyal, closes[i]))
            son_yon = sinyal
            
    return trades, trade_list

trades, trade_list = simulate_csharp_logic()
print(f"\n=== Result ===")
print(f"Total Trades: {trades}")
if len(trade_list) > 0:
    print(f"First Trade: {trade_list[0]}")
    print(f"Last Trade: {trade_list[-1]}")
