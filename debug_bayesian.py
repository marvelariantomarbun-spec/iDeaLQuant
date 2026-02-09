# -*- coding: utf-8 -*-
"""Bayesian exception debug"""
from datetime import datetime
import traceback
import pandas as pd
from src.data.ideal_parser import load_ideal_data
from src.optimization.bayesian_optimizer import IndicatorCache

df = load_ideal_data(r'D:\iDeal\ChartData', 'VIP', 'X030-T', '1')
start = datetime(2024, 1, 1)
end = datetime(2026, 1, 30, 23, 59)
df = df[(df['DateTime'] >= start) & (df['DateTime'] <= end)].copy()
print(f'Veri: {len(df)} bar')

cache = IndicatorCache(df)
print(f'closes: {cache.closes.shape}')
print(f'opens: {cache.opens.shape}')
print(f'typical: {cache.typical.shape}')
print(f'dates: {type(cache.dates)}')

# _evaluate_strategy1 doğrudan çağır
from src.optimization.bayesian_optimizer import BayesianObjective

objective = BayesianObjective(cache, strategy_index=0)

params = {
    'ars_period': 3,
    'ars_k': 0.015,
    'adx_period': 14,
    'adx_threshold': 25.0,
    'macdv_short': 12,
    'macdv_long': 26,
    'macdv_signal': 9,
    'macdv_threshold': 0.0,
    'netlot_period': 5,
    'netlot_threshold': 20.0,
    'ars_mesafe_threshold': 0.05,
    'bb_period': 20,
    'bb_std': 2.0,
    'bb_width_multiplier': 1.0,
    'bb_avg_period': 50,
    'yatay_ars_bars': 10,
    'yatay_adx_threshold': 20.0,
    'filter_score_threshold': 2,
    'min_score': 3,
    'exit_score': 3
}

print("\n_evaluate_strategy1 çağrılıyor...")
try:
    result = objective._evaluate_strategy1(params)
    print(f"Sonuç: {result}")
except Exception as e:
    print(f"HATA: {e}")
    traceback.print_exc()
