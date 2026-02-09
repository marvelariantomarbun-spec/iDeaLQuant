# -*- coding: utf-8 -*-
"""Test Bayesian ve Genetic optimizer - 01.01.2024 - 30.01.2026"""
import pandas as pd
from datetime import datetime

# Veri yükle
print("=" * 60)
print("VIP X030-T 1dk veri yükleniyor...")
print("=" * 60)

from src.data.ideal_parser import load_ideal_data

df = load_ideal_data(r'D:\iDeal\ChartData', 'VIP', 'X030-T', '1')
print(f"Toplam bar: {len(df)}")
print(f"Kolonlar: {df.columns.tolist()}")

# Tarih filtresi
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 1, 30, 23, 59)

df_filtered = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)].copy()
print(f"Filtrelenen bar (01.01.2024 - 30.01.2026): {len(df_filtered)}")

if len(df_filtered) == 0:
    print("HATA: Filtrelenmiş veri yok!")
    exit(1)

# ===== GENETIK TEST =====
print("\n" + "=" * 60)
print("GENETIK OPTIMIZER TESTİ")
print("=" * 60)

from src.optimization.genetic_optimizer import GeneticOptimizer, GeneticConfig

config = GeneticConfig(
    population_size=30,
    generations=10,
    elite_ratio=0.1,
    crossover_rate=0.8,
    mutation_rate=0.15
)

print("Genetik optimizer başlatılıyor...")
optimizer = GeneticOptimizer(df_filtered, config, strategy_index=0)

print("Evrim başlıyor (10 jenerasyon)...")
results = optimizer.run(verbose=True)

if results:
    print("\n--- GENETIK SONUÇ ---")
    if isinstance(results, list):
        best = results[0]
    else:
        best = results
    print(f"Net Kar: {best.get('net_profit', 0):.0f}")
    print(f"İşlem: {best.get('trades', 0)}")
    print(f"PF: {best.get('pf', 0):.2f}")
    print(f"Max DD: {best.get('max_dd', 0):.0f}")
else:
    print("Genetik sonuç üretemedi!")

# ===== BAYESIAN TEST =====
print("\n" + "=" * 60)
print("BAYESIAN OPTIMIZER TESTİ")
print("=" * 60)

from src.optimization.bayesian_optimizer import BayesianOptimizer

print("Bayesian optimizer başlatılıyor (50 deneme)...")
optimizer2 = BayesianOptimizer(df_filtered, n_trials=50, strategy_index=0)

print("Arama başlıyor...")
results2 = optimizer2.run(verbose=False)

if results2:
    print("\n--- BAYESIAN SONUÇ ---")
    if isinstance(results2, list):
        best2 = results2[0]
    else:
        best2 = results2
    print(f"Net Kar: {best2.get('net_profit', 0):.0f}")
    print(f"İşlem: {best2.get('trades', 0)}")
    print(f"PF: {best2.get('pf', 0):.2f}")
    print(f"Max DD: {best2.get('max_dd', 0):.0f}")
else:
    print("Bayesian sonuç üretemedi!")

print("\n" + "=" * 60)
print("TEST TAMAMLANDI")
print("=" * 60)
