"""
Deep debug MFI - check window behavior
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

raw = pd.read_csv('data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

ideal = pd.read_csv('data/ideal_ars_v2_data.csv', sep=';', decimal='.')

h = raw['Yuksek'].values.astype(float)
l = raw['Dusuk'].values.astype(float)
c = raw['Kapanis'].values.astype(float)
lot = raw['Lot'].values.astype(float)
n = len(c)
period = 14

tp = (h + l + c) / 3
raw_mf = tp * lot

# Check bar 195004 where drift starts
bar = 195004
print(f"Analyzing BarNo {bar}:")
print(f"IdealData MFI: {ideal[ideal['BarNo']==bar]['MFI'].values[0]}")

# MFI window: bars [bar-13 : bar+1]
window_start = bar - period + 1
window_end = bar + 1
print(f"\nWindow: [{window_start}:{window_end}]")

# Calculate pos/neg flow for this window
pos_sum = 0
neg_sum = 0

print("\nBar-by-bar breakdown:")
for i in range(window_start, window_end):
    direction = ""
    if tp[i] > tp[i-1]:
        pos_sum += raw_mf[i]
        direction = "UP   +"
    elif tp[i] < tp[i-1]:
        neg_sum += raw_mf[i]
        direction = "DOWN -"
    else:
        direction = "FLAT  "
    
    print(f"  Bar {i}: TP={tp[i]:.2f}, Lot={lot[i]:.0f}, RawMF={raw_mf[i]:.0f}, {direction}")

print(f"\nSum Positive: {pos_sum:.0f}")
print(f"Sum Negative: {neg_sum:.0f}")

if neg_sum > 0:
    ratio = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + ratio))
    print(f"MFI calculated: {mfi:.2f}")
else:
    print("MFI: 100 (no negative flow)")

# Check if there's missing data or different indexing
print("\n" + "="*60)
print("Checking for missing minutes around bar 195004:")
print("="*60)
raw['DT'] = pd.to_datetime(raw['Tarih'] + ' ' + raw['Saat'], format='%d.%m.%Y %H:%M:%S')

for i in range(bar-5, bar+5):
    if i+1 < n:
        diff_sec = (raw.iloc[i+1]['DT'] - raw.iloc[i]['DT']).total_seconds()
        gap = " <-- GAP!" if diff_sec > 120 else ""
        print(f"  Bar {i}: {raw.iloc[i]['DT']} (next in {diff_sec/60:.0f} min){gap}")
