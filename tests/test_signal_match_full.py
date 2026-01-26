# -*- coding: utf-8 -*-
"""
Full Signal Comparison Test
Matches Python executed trades against IdealData trade export.
"""

import sys, io, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2, Signal

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 1. Load Market Data
# -----------------------------------------------------------------------------
print("Loading market data...")
data_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
df_raw = pd.read_csv(data_path, sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
df_raw = df_raw.sort_values('DateTime').reset_index(drop=True)

# Data lists for strategy
opens = df_raw['Acilis'].values.tolist()
highs = df_raw['Yuksek'].values.tolist()
lows = df_raw['Dusuk'].values.tolist()
closes = df_raw['Kapanis'].values.tolist()
typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
times = df_raw['DateTime'].tolist()

print(f"Market data loaded: {len(df_raw)} bars")
print(f"Range: {times[0]} to {times[-1]}")

# 2. Load IdealData Trades (Reference)
# -----------------------------------------------------------------------------
print("\nLoading IdealData trades...")
ref_path = "d:/Projects/IdealQuant/data/ideal_signals_ars_v2.csv"

# Try reading with different encodings if needed
try:
    df_trades = pd.read_csv(ref_path, sep=';', encoding='utf-8')
except:
    df_trades = pd.read_csv(ref_path, sep=';', encoding='cp1254')

print(f"Columns found: {df_trades.columns.tolist()}")

# Map columns by index to avoid encoding issues with Turkish characters
# 0:No, 1:Yön, 2:Lot, 3:Açılış Tarihi, 4:Açılış Fyt, 5:Kapanış Tarihi ...
# We need col 1 (Yön), col 3 (Açılış Tarihi), col 5 (Kapanış Tarihi)

col_yon = df_trades.columns[1]
col_acilis_tarih = df_trades.columns[3]
col_kapanis_tarih = df_trades.columns[5]

print(f"Using columns: Direction='{col_yon}', OpenTime='{col_acilis_tarih}', CloseTime='{col_kapanis_tarih}'")

df_trades['OpenTime'] = pd.to_datetime(df_trades[col_acilis_tarih], format='%d.%m.%Y %H:%M', errors='coerce')
df_trades['CloseTime'] = pd.to_datetime(df_trades[col_kapanis_tarih], format='%d.%m.%Y %H:%M', errors='coerce')
df_trades['Direction'] = df_trades[col_yon].map({'Alış': 'LONG', 'Satış': 'SHORT'})

# Filter valid trades
df_trades = df_trades.dropna(subset=['OpenTime', 'Direction'])
print(f"IdealData trades loaded: {len(df_trades)} trades")
if len(df_trades) > 0:
    print(f"First trade: {df_trades.iloc[0]['OpenTime']} {df_trades.iloc[0]['Direction']}")
    print(f"Last trade: {df_trades.iloc[-1]['OpenTime']} {df_trades.iloc[-1]['Direction']}")

# 3. Running Python Backtest
# -----------------------------------------------------------------------------
print("\nRunning Python backtest strategy...")

# Config matching IdealData
config = StrategyConfigV2(
    ars_ema_period = 3,
    ars_atr_period = 10,
    ars_atr_mult = 0.5,
    ars_min_band = 0.002,
    ars_max_band = 0.015,
    momentum_period = 5,
    breakout_period = 10,
    rsi_period = 14,
    kar_al_pct = 3.0,
    iz_stop_pct = 1.5,
    vade_tipi = "ENDEKS"
)

strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, config)

# Simulation State
current_position = "FLAT" # LONG, SHORT, FLAT
entry_price = 0.0
entry_time = None
extreme_price = 0.0
position_size = 0

python_trades = []

# Find start index (a bit before first trade to warmup)
first_trade_date = df_trades.iloc[0]['OpenTime']
start_idx = 0
for i, t in enumerate(times):
    if t >= first_trade_date - timedelta(days=5):
        start_idx = i
        break

print(f"Starting simulation from bar {start_idx} ({times[start_idx]})")

for i in range(start_idx, len(closes)):
    # 1. Check outputs from previous position (Exit logic is inside get_signal)
    signal = strategy.get_signal(i, current_position, entry_price, extreme_price)
    
    # 2. Update extreme price for trailing stop
    if current_position == "LONG":
        extreme_price = max(extreme_price, highs[i])
    elif current_position == "SHORT":
        extreme_price = min(extreme_price, lows[i])
        
    # 3. Execute Signals
    # Important: IdealData usually executes signals on the CLOSE of the bar (or next open)
    # The signal calculation uses current bar close. So execution is conceptually "at close" or "next open".
    # Looking at IdealData logs, execution seems instant on signal bar close time.
    
    if signal == Signal.LONG and current_position != "LONG":
        # Close Short if exists
        if current_position == "SHORT":
            python_trades.append({
                'Direction': 'SHORT',
                'OpenTime': entry_time,
                'CloseTime': times[i],
                'EntryPrice': entry_price,
                'ExitPrice': closes[i],
                'Result': 'SwitchToLong'
            })
            
        # Open Long
        current_position = "LONG"
        entry_price = closes[i]
        entry_time = times[i]
        extreme_price = closes[i] # Init extreme
        
    elif signal == Signal.SHORT and current_position != "SHORT":
        # Close Long if exists
        if current_position == "LONG":
            python_trades.append({
                'Direction': 'LONG',
                'OpenTime': entry_time,
                'CloseTime': times[i],
                'EntryPrice': entry_price,
                'ExitPrice': closes[i],
                'Result': 'SwitchToShort'
            })
            
        # Open Short
        current_position = "SHORT"
        entry_price = closes[i]
        entry_time = times[i]
        extreme_price = closes[i] # Init extreme
        
    elif signal == Signal.FLAT and current_position != "FLAT":
        # Close executing position
        python_trades.append({
            'Direction': current_position,
            'OpenTime': entry_time,
            'CloseTime': times[i],
            'EntryPrice': entry_price,
            'ExitPrice': closes[i],
            'Result': 'Flat'
        })
        current_position = "FLAT"
        entry_price = 0
        entry_time = None

print(f"Python trades generated: {len(python_trades)}")

# 4. Compare Signals
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

# Convert to DataFrame for easier matching
df_py = pd.DataFrame(python_trades)
if not df_py.empty:
    df_py['Day'] = df_py['OpenTime'].dt.date
    
    # Matching logic: Find nearest trade in IdealData for each Python trade
    matched = 0
    perfect_match = 0
    
    print(f"{'Py_Index':<8} {'Py_OpenTime':<20} {'Py_Dir':<6} {'ID_OpenTime':<20} {'ID_Dir':<6} {'TimeDiff':<10} {'Match?'}")
    print("-" * 100)
    
    for i, py_row in df_py.iterrows():
        # Find IdealData trade with same direction and close time
        # Search window: +/- 60 minutes
        window = timedelta(minutes=60)
        
        candidates = df_trades[
            (df_trades['Direction'] == py_row['Direction']) &
            (df_trades['OpenTime'] >= py_row['OpenTime'] - window) &
            (df_trades['OpenTime'] <= py_row['OpenTime'] + window)
        ]
        
        match_status = "❌"
        id_info = "---"
        id_dir = ""
        diff_str = ""
        
        if not candidates.empty:
            # Pick closest
            candidates['Diff'] = (candidates['OpenTime'] - py_row['OpenTime']).abs()
            best_match = candidates.sort_values('Diff').iloc[0]
            
            diff_mins = best_match['Diff'].total_seconds() / 60
            id_info = str(best_match['OpenTime'])
            id_dir = best_match['Direction']
            diff_str = f"{diff_mins:.1f}m"
            
            if diff_mins == 0:
                match_status = "✅ Perfect"
                perfect_match += 1
                matched += 1
            elif diff_mins <= 5: # Allow small timing diffs
                match_status = "✓ Good"
                matched += 1
            else:
                match_status = "⚠️ Late/Early"
        
        # Print first 20 and then failures
        if i < 20 or match_status.startswith("❌") or match_status.startswith("⚠️"):
             print(f"{i:<8} {str(py_row['OpenTime']):<20} {py_row['Direction']:<6} {id_info:<20} {id_dir:<6} {diff_str:<10} {match_status}")

    print("-" * 100)
    print(f"Total Python Trades: {len(df_py)}")
    print(f"Total Ideal Trades:  {len(df_trades)}")
    print(f"Matched Trades:      {matched} ({matched/len(df_py)*100:.1f}%)")
    print(f"Perfect Matches:     {perfect_match} ({perfect_match/len(df_py)*100:.1f}%)")
    
    if matched / len(df_trades) > 0.95:
        print("\n✅ SUCCESS: Signal matching > 95%")
    else:
        print("\n❌ FAILURE: Signal matching low")

else:
    print("No python trades generated!")
    
