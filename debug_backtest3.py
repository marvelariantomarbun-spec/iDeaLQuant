# -*- coding: utf-8 -*-
"""
Compare full trade sequence Python vs what C# would do
"""
import pandas as pd
from datetime import datetime, time

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

# Simulate C# logic WITHOUT Python's state management
# Just use indicators and check conditions

def simulate_csharp_logic():
    """Simulate exactly what C# export does"""
    trades = 0
    son_yon = ""  # "", "A", "S", "F"
    
    warmup_bars = max(50, max(params['ars_period'], max(params['adx_period'], max(params['macdv_short'], params['macdv_long']))) + 10)
    
    for i in range(warmup_bars, len(closes)):
        # Get scores from strategy (same calculation as Python)
        l_score = strategy.long_scores[i]
        s_score = strategy.short_scores[i]
        yf = strategy.yatay_filtre[i]
        
        # C# ARS check for exit
        c_above_ars = closes[i] > strategy.ars[i]
        c_below_ars = closes[i] < strategy.ars[i]
        
        sinyal = ""
        
        # Exit logic (C# style)
        if son_yon == "A":  # Long position
            if c_below_ars or s_score >= params['exit_score']:
                sinyal = "F"
        elif son_yon == "S":  # Short position
            if c_above_ars or l_score >= params['exit_score']:
                sinyal = "F"
        
        # Entry logic (C# style)
        if sinyal == "" and son_yon != "A" and son_yon != "S":
            if yf == 1:
                if l_score >= params['min_score'] and s_score < 2:
                    sinyal = "A"
                elif s_score >= params['min_score'] and l_score < 2:
                    sinyal = "S"
        
        # Position update
        if sinyal != "" and son_yon != sinyal:
            if sinyal in ("A", "S"):
                trades += 1
            son_yon = sinyal
    
    return trades

# Simulate Python logic (what our backtest does)
def simulate_python_logic():
    """Simulate what Python backtest does"""
    trades = 0
    position = "FLAT"
    
    for i in range(len(closes)):
        signal = strategy.get_signal(i, position, 0, 0)
        
        if signal == Signal.LONG and position == "FLAT":
            position = "LONG"
            trades += 1
        elif signal == Signal.SHORT and position == "FLAT":
            position = "SHORT"
            trades += 1
        elif signal == Signal.FLAT and position != "FLAT":
            position = "FLAT"
    
    return trades

print("\n=== Trade Count Comparison ===")
csharp_trades = simulate_csharp_logic()
python_trades = simulate_python_logic()
print(f"C# Logic Simulation: {csharp_trades} trades")
print(f"Python Logic: {python_trades} trades")
print(f"Difference: {csharp_trades - python_trades}")

# If still different, trace first few trades
if csharp_trades != python_trades:
    print("\n=== Tracing First 10 Trades ===")
    
    # C# trace
    print("\nC# Logic First 10 Trades:")
    csharp_trade_list = []
    son_yon = ""
    warmup_bars = max(50, max(params['ars_period'], max(params['adx_period'], max(params['macdv_short'], params['macdv_long']))) + 10)
    
    for i in range(warmup_bars, len(closes)):
        l_score = strategy.long_scores[i]
        s_score = strategy.short_scores[i]
        yf = strategy.yatay_filtre[i]
        c_above_ars = closes[i] > strategy.ars[i]
        c_below_ars = closes[i] < strategy.ars[i]
        
        sinyal = ""
        
        if son_yon == "A":
            if c_below_ars or s_score >= params['exit_score']:
                sinyal = "F"
        elif son_yon == "S":
            if c_above_ars or l_score >= params['exit_score']:
                sinyal = "F"
        
        if sinyal == "" and son_yon != "A" and son_yon != "S":
            if yf == 1:
                if l_score >= params['min_score'] and s_score < 2:
                    sinyal = "A"
                elif s_score >= params['min_score'] and l_score < 2:
                    sinyal = "S"
        
        if sinyal != "" and son_yon != sinyal:
            if sinyal in ("A", "S"):
                csharp_trade_list.append((i, dates[i], sinyal, l_score, s_score, yf))
                if len(csharp_trade_list) >= 10:
                    break
            son_yon = sinyal
    
    for t in csharp_trade_list:
        print(f"  Bar {t[0]}: {t[1]} {t[2]} LS={t[3]} SS={t[4]} YF={t[5]}")
    
    # Python trace
    print("\nPython Logic First 10 Trades:")
    python_trade_list = []
    position = "FLAT"
    
    for i in range(len(closes)):
        signal = strategy.get_signal(i, position, 0, 0)
        
        if signal == Signal.LONG and position == "FLAT":
            l_score = strategy.long_scores[i]
            s_score = strategy.short_scores[i]
            yf = strategy.yatay_filtre[i]
            python_trade_list.append((i, dates[i], "LONG", l_score, s_score, yf))
            position = "LONG"
            if len(python_trade_list) >= 10:
                break
        elif signal == Signal.SHORT and position == "FLAT":
            l_score = strategy.long_scores[i]
            s_score = strategy.short_scores[i]
            yf = strategy.yatay_filtre[i]
            python_trade_list.append((i, dates[i], "SHORT", l_score, s_score, yf))
            position = "SHORT"
            if len(python_trade_list) >= 10:
                break
        elif signal == Signal.FLAT and position != "FLAT":
            position = "FLAT"
    
    for t in python_trade_list:
        print(f"  Bar {t[0]}: {t[1]} {t[2]} LS={t[3]} SS={t[4]} YF={t[5]}")
