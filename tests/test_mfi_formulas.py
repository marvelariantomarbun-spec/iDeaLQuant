"""
Test different MFI formulas to match IdealData
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
v = raw['Hacim'].values.astype(float)
n = len(c)
period = 14

# Calculate typical price
tp = (h + l + c) / 3

# Current implementation (SUM based)
def mfi_sum_based():
    raw_mf = tp * v
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
            ratio = sum_pos / sum_neg
            result[i] = 100.0 - (100.0 / (1.0 + ratio))
    return result

# Alternative: RMA smoothed (like RSI)
def mfi_rma_based():
    raw_mf = tp * v
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    
    for i in range(1, n):
        if tp[i] > tp[i-1]:
            pos_mf[i] = raw_mf[i]
        elif tp[i] < tp[i-1]:
            neg_mf[i] = raw_mf[i]
    
    # RMA smoothing
    avg_pos = np.zeros(n)
    avg_neg = np.zeros(n)
    
    avg_pos[period] = np.mean(pos_mf[1:period+1])
    avg_neg[period] = np.mean(neg_mf[1:period+1])
    
    for i in range(period+1, n):
        avg_pos[i] = (avg_pos[i-1] * (period-1) + pos_mf[i]) / period
        avg_neg[i] = (avg_neg[i-1] * (period-1) + neg_mf[i]) / period
    
    result = np.full(n, 50.0)
    for i in range(period, n):
        if avg_neg[i] == 0:
            result[i] = 100.0
        elif avg_pos[i] == 0:
            result[i] = 0.0
        else:
            ratio = avg_pos[i] / avg_neg[i]
            result[i] = 100.0 - (100.0 / (1.0 + ratio))
    return result

# Alternative: Using Lot instead of Hacim
def mfi_with_lot():
    lot = raw['Lot'].values.astype(float)
    raw_mf = tp * lot
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
            ratio = sum_pos / sum_neg
            result[i] = 100.0 - (100.0 / (1.0 + ratio))
    return result

# Test all formulas
bar_indices = ideal['BarNo'].values.astype(int)
ideal_mfi = ideal['MFI'].values

formulas = [
    ('SUM (current)', mfi_sum_based()),
    ('RMA', mfi_rma_based()),
    ('With Lot', mfi_with_lot()),
]

print("="*70)
print("MFI FORMULA COMPARISON")
print("="*70)

for name, mfi_vals in formulas:
    aligned = mfi_vals[bar_indices]
    diff = np.abs(ideal_mfi - aligned)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    print(f"{name:15}: Max={max_diff:8.4f}, Mean={mean_diff:8.4f}")
