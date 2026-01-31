"""
Test MFI with Lot column instead of Hacim
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# Load data
raw = pd.read_csv('data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

ideal = pd.read_csv('data/ideal_ars_v2_data.csv', sep=';', decimal='.')

h = raw['Yuksek'].values.astype(float)
l = raw['Dusuk'].values.astype(float)
c = raw['Kapanis'].values.astype(float)
lot = raw['Lot'].values.astype(float)  # Use Lot instead of Hacim
n = len(c)
period = 14

# Calculate MFI with Lot
tp = (h + l + c) / 3
raw_mf = tp * lot

pos_mf = np.zeros(n)
neg_mf = np.zeros(n)

for i in range(1, n):
    if tp[i] >= tp[i-1]:  # >= olmalı (flat = positive)
        pos_mf[i] = raw_mf[i]
    elif tp[i] < tp[i-1]:
        neg_mf[i] = raw_mf[i]

# Rolling sum
mfi_lot = np.full(n, 50.0)
for i in range(period, n):
    sum_pos = np.sum(pos_mf[i-period+1:i+1])
    sum_neg = np.sum(neg_mf[i-period+1:i+1])
    
    if sum_neg == 0:
        mfi_lot[i] = 100.0
    elif sum_pos == 0:
        mfi_lot[i] = 0.0
    else:
        ratio = sum_pos / sum_neg
        mfi_lot[i] = 100.0 - (100.0 / (1.0 + ratio))

# Compare with IdealData
bar_indices = ideal['BarNo'].values.astype(int)
ideal_mfi = ideal['MFI'].values
py_mfi_aligned = mfi_lot[bar_indices]

diff = np.abs(ideal_mfi - py_mfi_aligned)
max_diff = np.nanmax(diff)
mean_diff = np.nanmean(diff)

print("="*60)
print("MFI with LOT column test")
print("="*60)
print(f"Max diff:  {max_diff:.4f}")
print(f"Mean diff: {mean_diff:.6f}")
print(f"Status:    {'OK' if max_diff < 0.02 else 'FAIL'}")

# Show first 10 bars
print("\nFirst 10 bars comparison:")
for i in range(10):
    bar = bar_indices[i]
    print(f"  BarNo {bar}: Ideal={ideal_mfi[i]:.2f}, Python={py_mfi_aligned[i]:.2f}, Diff={diff[i]:.4f}")
