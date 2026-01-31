"""
Check for missing bars / gaps in CSV data that might cause HHV differences
"""
import pandas as pd
import numpy as np

raw = pd.read_csv('data/VIP_X030T_1dk_.csv', sep=';', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

ideal = pd.read_csv('data/ideal_ars_v2_data.csv', sep=';', decimal='.')

# Check the problematic area around bar 196329
print("="*60)
print("CHECKING AROUND LARGEST DIFFERENCE (raw[196329])")
print("="*60)

print("\nRaw CSV bars 196325-196340:")
for i in range(196325, min(196340, len(raw))):
    row = raw.iloc[i]
    print(f"  {i}: {row['Tarih']} {row['Saat']} High={row['Yuksek']}")

# Find the corresponding ideal bar
raw['Time_Short'] = raw['Saat'].str[:5]
raw['DateTime'] = raw['Tarih'] + ' ' + raw['Time_Short']
ideal['DateTime'] = ideal['Date'] + ' ' + ideal['Time']

# Get datetime for raw[196329]
dt_196329 = raw.iloc[196329]['DateTime']
print(f"\nraw[196329] DateTime: {dt_196329}")

# Find matching ideal bar
ideal_match = ideal[ideal['DateTime'] == dt_196329]
if len(ideal_match) > 0:
    print(f"Matching ideal bar found: index {ideal_match.index[0]}")
    print(f"IdealData values: HHV={ideal_match.iloc[0]['HHV']}, LLV={ideal_match.iloc[0]['LLV']}")

# Check for gaps in raw CSV around that area
print("\n" + "="*60)
print("CHECKING FOR TIME GAPS IN RAW CSV")
print("="*60)

# Parse datetime
raw['DT_full'] = pd.to_datetime(raw['Tarih'] + ' ' + raw['Saat'], format='%d.%m.%Y %H:%M:%S')

# Look for gaps > 1 minute around problematic area
for i in range(196300, 196350):
    if i+1 < len(raw):
        diff = (raw.iloc[i+1]['DT_full'] - raw.iloc[i]['DT_full']).total_seconds()
        if diff > 120:  # More than 2 minutes gap
            print(f"  GAP at {i}: {raw.iloc[i]['DT_full']} -> {raw.iloc[i+1]['DT_full']} ({diff/60:.1f} min)")

# Check ideal vs raw bar counts between matched dates
print("\n" + "="*60)
print("BAR COUNT COMPARISON")
print("="*60)

first_ideal_dt = ideal.iloc[0]['DateTime']
last_ideal_dt = ideal.iloc[-1]['DateTime']
print(f"IdealData range: {first_ideal_dt} to {last_ideal_dt} ({len(ideal)} bars)")

# Count raw bars in same range
first_raw_idx = raw[raw['DateTime'] == first_ideal_dt].index[0] if len(raw[raw['DateTime'] == first_ideal_dt]) > 0 else None
last_raw_idx = raw[raw['DateTime'] == last_ideal_dt].index[0] if len(raw[raw['DateTime'] == last_ideal_dt]) > 0 else None

if first_raw_idx is not None and last_raw_idx is not None:
    raw_bars_in_range = last_raw_idx - first_raw_idx + 1
    print(f"Raw CSV range: {first_raw_idx} to {last_raw_idx} ({raw_bars_in_range} bars)")
    print(f"Difference: {raw_bars_in_range - len(ideal)} extra bars in CSV")
