# -*- coding: utf-8 -*-
"""
Walk-Forward Analysis Module
----------------------------
Strateji parametrelerinin zaman içindeki kararlılığını test eder.
Veriyi pencerelere böler (Train/Test) ve kaydırarak ilerler.

Train: Optimize et (Genetik Algoritma kullanarak)
Test: İleriye dönük test et (Out-of-sample)
"""

import sys
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  <-- Removed dependency
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Proje kök dizini
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.optimization.genetic_optimizer import GeneticOptimizer, GeneticConfig, FitnessEvaluator
from src.optimization.strategy2_optimizer import load_data, IndicatorCache, fast_backtest_strategy2

@dataclass
class WalkForwardConfig:
    train_window_months: int = 6   # Eğitim penceresi (ay)
    test_window_months: int = 1    # Test penceresi (ay)
    step_months: int = 1           # Kaydırma adımı (ay)
    genetic_config: Optional[GeneticConfig] = None  # Optimizer ayarları

class WalkForwardAnalysis:
    def __init__(self, df: pd.DataFrame, config: WalkForwardConfig, strategy_index: int = 1):
        self.df = df
        self.config = config
        self.strategy_index = strategy_index
        self.results = []
        
        # Tarih indeksini ayarla (eğer yoksa)
        if 'Tarih' in df.columns:
            # Tarih formatı dd.mm.yyyy varsayılıyor
            self.df['Date'] = pd.to_datetime(self.df['Tarih'], format='%d.%m.%Y', dayfirst=True)
        
    def run(self):
        """Walk-Forward analizini çalıştır"""
        start_date = self.df['Date'].min()
        end_date = self.df['Date'].max()
        
        print(f"Veri Aralığı: {start_date.date()} - {end_date.date()}")
        print(f"Train: {self.config.train_window_months} ay, Test: {self.config.test_window_months} ay")
        
        current_date = start_date
        
        train_months = self.config.train_window_months
        test_months = self.config.test_window_months
        
        window_count = 0
        
        while True:
            # Pencere tarihlerini belirle
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months)
            
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > end_date:
                break
                
            # Veriyi böl
            train_mask = (self.df['Date'] >= train_start) & (self.df['Date'] < train_end)
            test_mask = (self.df['Date'] >= test_start) & (self.df['Date'] < test_end)
            
            df_train = self.df.loc[train_mask].copy().reset_index(drop=True)
            df_test = self.df.loc[test_mask].copy().reset_index(drop=True)
            
            if len(df_train) < 500 or len(df_test) < 100:
                print(f"Yetersiz veri (Train:{len(df_train)}, Test:{len(df_test)}), atlanıyor...")
                current_date += pd.DateOffset(months=self.config.step_months)
                continue

            window_count += 1
            print(f"\n--- Pencere {window_count} ---")
            print(f"Train: {train_start.date()} -> {train_end.date()} ({len(df_train)} bars)")
            print(f"Test : {test_start.date()} -> {test_end.date()} ({len(df_test)} bars)")
            
            # 1. OPTİMİZASYON (Train)
            print("  [STEP 1] Optimize ediliyor (Genetik)...")
            # Hızlı optimizasyon için nesil sayısını düşük tutabiliriz
            gen_config = self.config.genetic_config or GeneticConfig(
                population_size=30, generations=10, early_stop_generations=3
            )
            
            optimizer = GeneticOptimizer(df_train, gen_config, strategy_index=self.strategy_index)
            opt_result = optimizer.run(verbose=False)
            
            best_params = opt_result['best_params']
            train_score = opt_result['best_fitness']
            
            print(f"  Best Train Fitness: {train_score:.2f}")
            print(f"  Params: {best_params}")
            
            # 2. VALIDASYON (Test)
            print("  Test ediliyor (OOS)...")
            evaluator = FitnessEvaluator(df_test, strategy_index=self.strategy_index)
            test_result = evaluator.evaluate(best_params)
            
            print(f"  Test Net Profit: {test_result['net_profit']:.2f}")
            print(f"  Test PF: {test_result['pf']:.2f}")
            
            self.results.append({
                'window': window_count,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'params': best_params,
                'train_metrics': opt_result['best_result'],
                'test_metrics': test_result
            })
            
            # Bir sonraki adıma kaydır
            current_date += pd.DateOffset(months=self.config.step_months)
            
        self._analyze_results()
        
    def _analyze_results(self):
        """Sonuçları analiz et ve özetle"""
        if not self.results:
            print("Sonuç yok.")
            return

        print("\n=== WALK-FORWARD ANALİZ SONUÇLARI ===")
        total_profit = sum(r['test_metrics']['net_profit'] for r in self.results)
        avg_pf = np.mean([r['test_metrics']['pf'] for r in self.results])
        win_windows = sum(1 for r in self.results if r['test_metrics']['net_profit'] > 0)
        total_windows = len(self.results)
        
        print(f"Toplam Test Karı: {total_profit:,.2f}")
        print(f"Ortalama Test PF: {avg_pf:.2f}")
        print(f"Kazançlı Pencereler: {win_windows}/{total_windows} (%{win_windows/total_windows*100:.1f})")
        
        # Walk-Forward Efficiency Ratio (WFE)
        # WFE = OOS Toplam Kar / IS Toplam Kar
        # %50+ geçerli kabul edilir, %100+ mükemmel
        total_train_profit = sum(r['train_metrics']['net_profit'] for r in self.results)
        if total_train_profit > 0:
            wfe = (total_profit / total_train_profit) * 100
            print(f"Walk-Forward Efficiency: %{wfe:.1f}")
            if wfe >= 50:
                print(f"  → Strateji robust görünüyor (WFE >= %50)")
            else:
                print(f"  ⚠ Overfit riski yüksek (WFE < %50)")
        else:
            print(f"Walk-Forward Efficiency: N/A (Train karı <= 0)")
            wfe = 0.0

        # Genel özet satırı ekle
        print(f"\nStability Ratio: {win_windows}/{total_windows} = %{win_windows/total_windows*100:.1f}")

        # CSV Kaydet
        summary_data = []
        for r in self.results:
            row = {
                'Window': r['window'],
                'Test_Start': r['test_start'].date(),
                'Test_End': r['test_end'].date(),
                'Train_NP': r['train_metrics']['net_profit'],
                'Test_NP': r['test_metrics']['net_profit'],
                'Test_PF': r['test_metrics']['pf'],
                'Test_DD': r['test_metrics']['max_dd'],
                'WFE': wfe,
                **r['params']
            }
            summary_data.append(row)
            
        df_res = pd.DataFrame(summary_data)
        
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(results_dir, "walk_forward_results.csv")
        df_res.to_csv(output_path, index=False)
        print(f"Detaylı sonuçlar kaydedildi: {output_path}")

if __name__ == "__main__":
    # Test çalıştırması
    try:
        df = load_data()
        config = WalkForwardConfig(
            train_window_months=6,
            test_window_months=1,
            step_months=1
        )
        wf = WalkForwardAnalysis(df, config)
        wf.run()
    except KeyboardInterrupt:
        print("İptal.")
