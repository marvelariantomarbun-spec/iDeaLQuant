# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation Module
-----------------------------
Strateji sonuçlarının şansa bağlılığını ve riskini analiz eder.
İşlem listesi üzerinde yeniden örnekleme (resampling) yaparak olası alternatif senaryoları simüle eder.

Analizler:
1. Trade Resampling: İşlem sırasını karıştırarak olası Max DD ve Kar dağılımı.
2. Randomized Parameters: (Opsiyonel) Parametre hassasiyet analizi.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class MonteCarloSimulator:
    def __init__(self, trades: List[float], initial_capital: float = 10000.0):
        """
        Args:
            trades: İşlem PnL listesi (TL bazlı)
            initial_capital: Başlangıç sermayesi (MaxDD hesaplamak için)
        """
        self.trades = np.array(trades)
        self.initial_capital = initial_capital
        
    def run_simulation(self, num_simulations: int = 1000, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Monte Carlo simülasyonunu çalıştır.
        
        Args:
            num_simulations: Simülasyon sayısı
            confidence_level: Güven aralığı (%95)
            
        Returns:
            Dict: Analiz sonuçları (MaxDD riski, Kar ihtimali vb.)
        """
        if len(self.trades) < 10:
            return {'error': 'Yetersiz işlem sayısı (<10)'}
            
        final_equities = []
        max_drawdowns = []
        ruin_probabilities = [] # Sermayenin %X'ini kaybetme ihtimali
        
        n_trades = len(self.trades)
        
        print(f"Monte Carlo Simülasyonu Başlıyor ({num_simulations} iterasyon)...")
        
        for _ in range(num_simulations):
            # 1. Resampling (Bootstrap)
            # Mevcut işlemlerden rastgele seçim yap (tekrarlı)
            indices = np.random.randint(0, n_trades, n_trades)
            sim_trades = self.trades[indices]
            
            # 2. Equity Curve
            equity_curve = np.concatenate([[self.initial_capital], self.initial_capital + np.cumsum(sim_trades)])
            
            # 3. Metrics
            final_equity = equity_curve[-1]
            final_equities.append(final_equity)
            
            # Max DD
            peak = np.maximum.accumulate(equity_curve)
            dd = peak - equity_curve
            max_dd = np.max(dd)
            max_drawdowns.append(max_dd)
            
        # İstatistikler
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        # Risk of Ruin (Basitleştirilmiş: Başlangıç sermayesinin %20'sini kaybetme olasılığı)
        # Tabii DD TL bazlı hesaplandığı için, %20 = 2000 TL diyelim
        ruin_threshold = self.initial_capital * 0.2
        p_ruin = np.mean(max_drawdowns > ruin_threshold) * 100
        
        # Güven aralıkları (Worst case %5)
        worst_case_equity = np.percentile(final_equities, (1 - confidence_level) * 100)
        worst_case_dd = np.percentile(max_drawdowns, confidence_level * 100)
        median_dd = np.median(max_drawdowns)
        median_profit = np.median(final_equities) - self.initial_capital
        
        result = {
            'simulations': num_simulations,
            'original_profit': np.sum(self.trades),
            'median_profit': median_profit,
            'worst_case_profit': worst_case_equity - self.initial_capital,
            'median_max_dd': median_dd,
            'worst_case_max_dd': worst_case_dd, # %95 güvenle MaxDD bundan kötü olmaz
            'risk_of_ruin_20pct': p_ruin
        }
        
        return result

    def print_report(self, result: Dict[str, Any]):
        """Raporu yazdır"""
        if 'error' in result:
            print(result['error'])
            return
            
        print("\n=== MONTE CARLO SİMÜLASYON RAPORU ===")
        print(f"Simülasyon Sayısı: {result['simulations']}")
        print(f"Orijinal Net Kar : {result['original_profit']:,.2f} TL")
        print("-" * 40)
        print(f"Medyan Net Kar   : {result['median_profit']:,.2f} TL")
        print(f"Kötü Senaryo Kar : {result['worst_case_profit']:,.2f} TL (%5 ihtimal)")
        print("-" * 40)
        print(f"Medyan MaxDD     : {result['median_max_dd']:,.2f} TL")
        print(f"Kötü Senaryo DD  : {result['worst_case_max_dd']:,.2f} TL (%95 ihtimal)")
        print(f"Batış Riski (%20): %{result['risk_of_ruin_20pct']:.1f}")
        print("=======================================")

if __name__ == "__main__":
    # Test verisi
    dummy_trades = np.random.normal(50, 200, 100) # Ort 50 TL kazanç, 200 TL std dev
    mc = MonteCarloSimulator(dummy_trades)
    res = mc.run_simulation()
    mc.print_report(res)
