"""
Test MFI with different FLAT bar handling
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

# Method 1: FLAT = 0 (current)
def mfi_flat_zero():
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    for i in range(1, n):
        if tp[i] > tp[i-1]:
            pos_mf[i] = raw_mf[i]
        elif tp[i] < tp[i-1]:
            neg_mf[i] = raw_mf[i]
    
    result = np.full(n, 50.0)
    for i in range(period, n):
        sum_pos = np.sum(pos_mf[i-period+1:i+1])
        sum_neg = np.sum(neg_mf[i-period+1:i+1])
        if sum_neg == 0:
            result[i] = 100.0
        elif sum_pos == 0:
            result[i] = 0.0
        else:
            result[i] = 100.0 - (100.0 / (1.0 + sum_pos/sum_neg))
    return result

# Method 2: FLAT inherits previous direction
def mfi_flat_inherit():
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    last_dir = 0  # 1=pos, -1=neg, 0=none
    
    for i in range(1, n):
        if tp[i] > tp[i-1]:
            pos_mf[i] = raw_mf[i]
            last_dir = 1
        elif tp[i] < tp[i-1]:
            neg_mf[i] = raw_mf[i]
            last_dir = -1
        else:
            # FLAT - inherit previous
            if last_dir == 1:
                pos_mf[i] = raw_mf[i]
            elif last_dir == -1:
                neg_mf[i] = raw_mf[i]
    
    result = np.full(n, 50.0)
    for i in range(period, n):
        sum_pos = np.sum(pos_mf[i-period+1:i+1])
        sum_neg = np.sum(neg_mf[i-period+1:i+1])
        if sum_neg == 0:
            result[i] = 100.0
        elif sum_pos == 0:
            result[i] = 0.0
        else:
            result[i] = 100.0 - (100.0 / (1.0 + sum_pos/sum_neg))
    return result

# Method 3: FLAT = positive
def mfi_flat_positive():
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    for i in range(1, n):
        if tp[i] >= tp[i-1]:  # Include FLAT as positive
            pos_mf[i] = raw_mf[i]
        else:
            neg_mf[i] = raw_mf[i]
    
    result = np.full(n, 50.0)
    for i in range(period, n):
        sum_pos = np.sum(pos_mf[i-period+1:i+1])
        sum_neg = np.sum(neg_mf[i-period+1:i+1])
        if sum_neg == 0:
            result[i] = 100.0
        elif sum_pos == 0:
            result[i] = 0.0
        else:
            result[i] = 100.0 - (100.0 / (1.0 + sum_pos/sum_neg))
    return result

# Test all
bar_indices = ideal['BarNo'].values.astype(int)
ideal_mfi = ideal['MFI'].values

methods = [
    ('FLAT=0 (current)', mfi_flat_zero()),
    ('FLAT=inherit', mfi_flat_inherit()),
    ('FLAT=positive', mfi_flat_positive()),
]

print("="*60)
print("MFI FLAT HANDLING TEST")
print("="*60)

for name, mfi in methods:
    aligned = mfi[bar_indices]
    diff = np.abs(ideal_mfi - aligned)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    print(f"{name:20}: Max={max_diff:8.4f}, Mean={mean_diff:8.4f}")
