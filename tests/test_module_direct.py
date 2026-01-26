# -*- coding: utf-8 -*-
"""
Doğrudan ARS_Dynamic testi - modül import kontrolü
"""

import sys, io, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Cache temizle
if 'indicators.core' in sys.modules:
    del sys.modules['indicators.core']
    
from indicators.core import ARS_Dynamic, EMA, ATR
import math

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 80)
print("MODÜLogtest - math.floor kullanılıyor mu?")
print("=" * 80)

# Test data
typical = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0] * 20
highs = [h + 2 for h in typical]
lows = [l - 2 for l in typical]
closes = typical.copy()

ars = ARS_Dynamic(typical, highs, lows, closes, ema_period=3, atr_period=10, atr_mult=0.5)

print(f"İlk 20 ARS değeri:")
for i in range(20):
    print(f"{i}: {ars[i]:.6f}")

print("\n" + "=" * 80)
print("Yuvarlama testi:")
print("=" * 80)

# Manuel yuvarlama testi
test_val = 14180.724330
test_step = 0.951088

# Eski yöntem (banker's rounding)
old_round = round(test_val / test_step) * test_step
print(f"Eski yöntem (banker's rounding): {old_round:.6f}")

# Yeni yöntem (standard rounding)
new_round = math.floor(test_val / test_step + 0.5) * test_step
print(f"Yeni yöntem (standard rounding): {new_round:.6f}")

# IdealData beklenen
ideal_val = 14183.580000
print(f"IdealData değeri: {ideal_val:.6f}")

print(f"\nFark (eski): {abs(old_round - ideal_val):.6f}")
print(f"Fark (yeni): {abs(new_round - ideal_val):.6f}")

print("\n" + "=" * 80)
