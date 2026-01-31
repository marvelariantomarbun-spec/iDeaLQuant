"""
Debug script to find exact source of indicator differences
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from src.indicators.core import HHV, LLV, RSI, Momentum

# Load data
print("Loading data...")
raw = pd.read_csv('data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

ideal = pd.read_csv('data/ideal_ars_v2_data.csv', sep=';', decimal='.')

print(f"Raw CSV: {len(raw)} bars")
print(f"IdealData: {len(ideal)} bars")

# Match by datetime
raw['Time_Short'] = raw['Saat'].str[:5]
raw['DateTime'] = raw['Tarih'] + ' ' + raw['Time_Short']
ideal['DateTime'] = ideal['Date'] + ' ' + ideal['Time']

# Find matching indices
raw_dt_idx = {dt: i for i, dt in enumerate(raw['DateTime'])}
matches = [(i, raw_dt_idx[dt]) for i, dt in enumerate(ideal['DateTime']) if dt in raw_dt_idx]

print(f"\nMatched bars: {len(matches)}")
print(f"First match: ideal[{matches[0][0]}] -> raw[{matches[0][1]}]")
print(f"Last match: ideal[{matches[-1][0]}] -> raw[{matches[-1][1]}]")

# Calculate Python indicators
print("\nCalculating Python indicators...")
highs = raw['Yuksek'].values.astype(float).tolist()
lows = raw['Dusuk'].values.astype(float).tolist()
closes = raw['Kapanis'].values.astype(float).tolist()

py_hhv = HHV(highs, 20)
py_llv = LLV(lows, 20)
py_momentum = Momentum(closes, 10)
py_rsi = RSI(closes, 14)

# Detailed comparison for first 10 matched bars
print("\n" + "="*80)
print("DETAILED BAR-BY-BAR COMPARISON (First 10 matched bars)")
print("="*80)

for idx in range(min(10, len(matches))):
    i_ideal, i_raw = matches[idx]
    
    print(f"\n--- Bar {idx} (ideal[{i_ideal}] -> raw[{i_raw}]) ---")
    print(f"DateTime: {ideal.iloc[i_ideal]['DateTime']}")
    
    # HHV
    ideal_hhv = ideal.iloc[i_ideal]['HHV']
    python_hhv = py_hhv[i_raw]
    hhv_diff = abs(ideal_hhv - python_hhv)
    print(f"HHV:      Ideal={ideal_hhv:.2f}, Python={python_hhv:.2f}, Diff={hhv_diff:.2f}")
    
    if hhv_diff > 0:
        # Show the window
        start = max(0, i_raw - 19)
        window = highs[start:i_raw+1]
        print(f"          Window[{start}:{i_raw+1}] max={max(window):.2f}")
    
    # LLV
    ideal_llv = ideal.iloc[i_ideal]['LLV']
    python_llv = py_llv[i_raw]
    llv_diff = abs(ideal_llv - python_llv)
    print(f"LLV:      Ideal={ideal_llv:.2f}, Python={python_llv:.2f}, Diff={llv_diff:.2f}")
    
    # Momentum
    ideal_mom = ideal.iloc[i_ideal]['Momentum']
    python_mom = py_momentum[i_raw]
    mom_diff = abs(ideal_mom - python_mom)
    print(f"Momentum: Ideal={ideal_mom:.4f}, Python={python_mom:.4f}, Diff={mom_diff:.4f}")
    
    # RSI
    ideal_rsi = ideal.iloc[i_ideal]['RSI']
    python_rsi = py_rsi[i_raw]
    rsi_diff = abs(ideal_rsi - python_rsi)
    print(f"RSI:      Ideal={ideal_rsi:.4f}, Python={python_rsi:.4f}, Diff={rsi_diff:.4f}")

# Find bars with largest differences
print("\n" + "="*80)
print("FINDING BARS WITH LARGEST DIFFERENCES")
print("="*80)

hhv_diffs = []
for i_ideal, i_raw in matches:
    ideal_val = ideal.iloc[i_ideal]['HHV']
    python_val = py_hhv[i_raw]
    hhv_diffs.append((i_ideal, i_raw, ideal_val, python_val, abs(ideal_val - python_val)))

# Sort by diff
hhv_diffs.sort(key=lambda x: x[4], reverse=True)
print("\nTop 5 HHV differences:")
for item in hhv_diffs[:5]:
    print(f"  ideal[{item[0]}] raw[{item[1]}]: Ideal={item[2]:.2f}, Python={item[3]:.2f}, Diff={item[4]:.2f}")
