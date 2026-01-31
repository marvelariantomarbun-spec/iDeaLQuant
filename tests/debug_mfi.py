"""
Debug MFI difference between Python and IdealData
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from src.indicators.core import MoneyFlowIndex

# Load data
raw = pd.read_csv('data/VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

ideal = pd.read_csv('data/ideal_ars_v2_data.csv', sep=';', decimal='.')

# Check specific bar where we have max diff
bar_no = 195515
ideal_row = ideal[ideal['BarNo'] == bar_no]
print(f"Checking BarNo {bar_no}:")
print(f"IdealData MFI: {ideal_row['MFI'].values[0]}")

h = raw['Yuksek'].values.astype(float).tolist()
l = raw['Dusuk'].values.astype(float).tolist()
c = raw['Kapanis'].values.astype(float).tolist()
v = raw['Hacim'].values.astype(float).tolist()

py_mfi = MoneyFlowIndex(h, l, c, v, 14)
print(f"Python MFI: {py_mfi[bar_no]:.4f}")

# Check volume data
print(f"\nVolume at bar {bar_no}: {v[bar_no]}")
print(f"Volume range [{bar_no-5}:{bar_no+5}]: {v[bar_no-5:bar_no+5]}")

# Check for 0 volumes in MFI window
window_volumes = v[bar_no-13:bar_no+1]
print(f"\nMFI window volumes: {window_volumes}")
print(f"Any zero volumes: {0 in window_volumes}")

# Check typical price movement
tp = [(h[i] + l[i] + c[i]) / 3 for i in range(len(c))]
print(f"\nTypical price at {bar_no}: {tp[bar_no]:.2f}")
print(f"Typical price at {bar_no-1}: {tp[bar_no-1]:.2f}")
print(f"TP direction: {'UP' if tp[bar_no] > tp[bar_no-1] else 'DOWN' if tp[bar_no] < tp[bar_no-1] else 'FLAT'}")

# Compare first 5 bars
print("\n" + "="*60)
print("FIRST 5 BARS COMPARISON:")
print("="*60)
for i in range(5):
    bar = ideal['BarNo'].iloc[i]
    ideal_mfi = ideal['MFI'].iloc[i]
    py_val = py_mfi[bar]
    diff = abs(ideal_mfi - py_val)
    print(f"BarNo {bar}: Ideal={ideal_mfi:.2f}, Python={py_val:.2f}, Diff={diff:.2f}")
