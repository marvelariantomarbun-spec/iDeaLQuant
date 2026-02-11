# -*- coding: utf-8 -*-
"""Test timeframe scaling"""
import sys, os
sys.path.insert(0, '.')
from src.ui.widgets.optimizer_panel import STRATEGY1_PARAM_GROUPS, STRATEGY2_PARAM_GROUPS, scale_param_groups

print('=== STRATEJI 1: adx_period ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY1_PARAM_GROUPS, period)
    p = s['ADX']['params']['adx_period']
    print(f"  {period:2d}dk: min={p['min']:3d}  max={p['max']:3d}  default={p['default']:3d}  step={p['step']}")

print()
print('=== STRATEJI 1: ars_period ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY1_PARAM_GROUPS, period)
    p = s['ARS']['params']['ars_period']
    print(f"  {period:2d}dk: min={p['min']:3d}  max={p['max']:3d}  default={p['default']:3d}  step={p['step']}")

print()
print('=== STRATEJI 1: adx_threshold (olceklenmemeli) ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY1_PARAM_GROUPS, period)
    p = s['ADX']['params']['adx_threshold']
    print(f"  {period:2d}dk: min={p['min']}  max={p['max']}  default={p['default']}  step={p['step']}")

print()
print('=== STRATEJI 2: mfi_period ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY2_PARAM_GROUPS, period)
    p = s['Giris_Filtreleri']['params']['mfi_period']
    print(f"  {period:2d}dk: min={p['min']:3d}  max={p['max']:3d}  default={p['default']:3d}  step={p['step']}")

print()
print('=== STRATEJI 2: atr_sl_mult (olceklenmemeli) ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY2_PARAM_GROUPS, period)
    p = s['Cikis_Risk']['params']['atr_sl_mult']
    print(f"  {period:2d}dk: min={p['min']}  max={p['max']}  default={p['default']}  step={p['step']}")

print()
print('=== STRATEJI 1: bb_period ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY1_PARAM_GROUPS, period)
    p = s['Yatay_BB']['params']['bb_period']
    print(f"  {period:2d}dk: min={p['min']:3d}  max={p['max']:3d}  default={p['default']:3d}  step={p['step']}")

print()
print('=== STRATEJI 1: min_score (olceklenmemeli) ===')
for period in [1, 5, 15, 60]:
    s = scale_param_groups(STRATEGY1_PARAM_GROUPS, period)
    p = s['Skor']['params']['min_score']
    print(f"  {period:2d}dk: min={p['min']}  max={p['max']}  default={p['default']}  step={p['step']}")
