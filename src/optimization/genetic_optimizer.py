# -*- coding: utf-8 -*-
"""
Genetic Algorithm Optimizer for Strategy 2 (ARS Trend v2)
Hibrit yaklaşım: Grid Search + Genetic Algorithm

Avantajlar:
- Daha az kombinasyon denemesi
- Yerel optimumlara takılmaz
- Yüksek boyutlu parametre uzaylarında etkili
"""

import sys
import os
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random

# Proje kök dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, ATR, Momentum, HHV, LLV, ARS_Dynamic, MoneyFlowIndex

# ==============================================================================
# GENETIC ALGORITHM CONFIG
# ==============================================================================
@dataclass
class GeneticConfig:
    """Genetik Algoritma Konfigürasyonu"""
    population_size: int = 50          # Popülasyon boyutu
    generations: int = 30              # Nesil sayısı
    elite_ratio: float = 0.1           # Elit oran (%10)
    crossover_rate: float = 0.8        # Çaprazlama oranı
    mutation_rate: float = 0.15        # Mutasyon oranı
    tournament_size: int = 5           # Turnuva boyutu
    
    # Erken durdurma
    early_stop_generations: int = 8    # İyileşme olmadan bekleme
    min_improvement: float = 0.01      # Minimum iyileşme oranı


# ==============================================================================
# PARAMETER SPACE
# ==============================================================================

# Strateji 1 Parametre Uzayı (20 parametre)
STRATEGY1_PARAMS = {
    # ARS
    'ars_period': (2, 15, 1, True),
    'ars_k': (0.005, 0.03, 0.005, False),
    # ADX
    'adx_period': (10, 30, 2, True),
    'adx_threshold': (15.0, 35.0, 5.0, False),
    # MACD-V
    'macdv_short': (8, 18, 1, True),
    'macdv_long': (20, 40, 2, True),
    'macdv_signal': (5, 15, 1, True),
    'macdv_threshold': (-50.0, 50.0, 10.0, False),
    # NetLot
    'netlot_period': (3, 10, 1, True),
    'netlot_threshold': (10.0, 50.0, 5.0, False),
    # Yatay Filtre
    'ars_mesafe_threshold': (0.1, 0.5, 0.05, False),
    'bb_period': (15, 30, 5, True),
    'bb_std': (1.5, 3.0, 0.5, False),
    'bb_width_multiplier': (0.5, 1.5, 0.1, False),
    'bb_avg_period': (30, 100, 10, True),
    'yatay_ars_bars': (5, 20, 5, True),
    'yatay_adx_threshold': (15.0, 30.0, 5.0, False),
    'filter_score_threshold': (1, 4, 1, True),
    # Skor
    'min_score': (2, 4, 1, True),
    'exit_score': (2, 4, 1, True),
}

# Strateji 2 Parametre Uzayı (20 parametre)
STRATEGY2_PARAMS = {
    # ARS Dinamik
    'ars_ema_period': (2, 12, 1, True),
    'ars_atr_period': (7, 20, 2, True),
    'ars_atr_mult': (0.3, 1.5, 0.1, False),
    'ars_min_band': (0.001, 0.005, 0.001, False),
    'ars_max_band': (0.010, 0.025, 0.005, False),
    # Giriş Filtreleri
    'momentum_period': (3, 10, 1, True),
    'momentum_threshold': (50.0, 200.0, 25.0, False),
    'breakout_period': (5, 30, 5, True),
    'mfi_period': (10, 21, 2, True),
    'mfi_hhv_period': (10, 21, 2, True),
    'mfi_llv_period': (10, 21, 2, True),
    'volume_hhv_period': (10, 21, 2, True),
    # Çıkış/Risk
    'atr_exit_period': (10, 21, 2, True),
    'atr_sl_mult': (1.0, 4.0, 0.5, False),
    'atr_tp_mult': (3.0, 8.0, 1.0, False),
    'atr_trail_mult': (1.0, 4.0, 0.5, False),
    'exit_confirm_bars': (1, 5, 1, True),
    'exit_confirm_mult': (0.5, 2.0, 0.25, False),
    # İnce Ayar
    'volume_mult': (0.5, 1.5, 0.1, False),
    'volume_llv_period': (10, 21, 2, True),
}


class ParameterSpace:
    """Parametre uzayı tanımı - Her iki strateji için"""
    def __init__(self, strategy_index: int = 1):
        """
        Args:
            strategy_index: 0 = Strateji 1 (Gatekeeper), 1 = Strateji 2 (ARS Trend v2)
        """
        self.strategy_index = strategy_index
        self.params = STRATEGY1_PARAMS if strategy_index == 0 else STRATEGY2_PARAMS
        self.param_names = list(self.params.keys())
        self.n_params = len(self.param_names)
        
    def random_individual(self) -> np.ndarray:
        """Rastgele birey oluştur"""
        genes = []
        for name in self.param_names:
            min_val, max_val, step, is_int = self.params[name]
            if is_int:
                val = random.choice(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                n_steps = int((max_val - min_val) / step) + 1
                val = min_val + random.randint(0, n_steps - 1) * step
            genes.append(val)
        return np.array(genes)
    
    def decode(self, genes: np.ndarray) -> Dict[str, Any]:
        """Genleri parametre sözlüğüne çevir"""
        return {name: genes[i] for i, name in enumerate(self.param_names)}
    
    def mutate(self, genes: np.ndarray) -> np.ndarray:
        """Mutasyon uygula"""
        new_genes = genes.copy()
        for i, name in enumerate(self.param_names):
            if random.random() < 0.3:  # Her gen için %30 şans
                min_val, max_val, step, is_int = self.params[name]
                # Rastgele yeni değer veya ±step
                if random.random() < 0.5:
                    # Küçük mutasyon
                    delta = step * random.choice([-1, 1])
                    new_genes[i] = np.clip(new_genes[i] + delta, min_val, max_val)
                else:
                    # Tamamen yeni değer
                    if is_int:
                        new_genes[i] = random.choice(range(int(min_val), int(max_val) + 1, int(step)))
                    else:
                        n_steps = int((max_val - min_val) / step) + 1
                        new_genes[i] = min_val + random.randint(0, n_steps - 1) * step
        return new_genes
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """İki noktalı çaprazlama"""
        n = len(parent1)
        if n < 3:
            return parent1.copy(), parent2.copy()
        
        # İki nokta seç
        points = sorted(random.sample(range(1, n), 2))
        p1, p2 = points
        
        child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
        child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])
        
        return child1, child2


# ==============================================================================
# FITNESS FUNCTION (Strategy-based Backtest wrapper)
# ==============================================================================
class FitnessEvaluator:
    """Fitness değerlendirici - Her iki strateji için backtest wrapper"""
    
    def __init__(self, df: pd.DataFrame, strategy_index: int = 1):
        """
        Args:
            df: Veri DataFrame'i
            strategy_index: 0 = Strateji 1 (Gatekeeper), 1 = Strateji 2 (ARS Trend v2)
        """
        self.df = df
        self.strategy_index = strategy_index
        self.opens = df['Acilis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.closes = df['Kapanis'].values
        self.typical = df['Tipik'].values
        self.volumes = df['Lot'].values
        self.n = len(self.closes)
        
        # Tarih bilgisi
        if 'DateTime' in df.columns:
            self.dates = df['DateTime'].tolist()
        else:
            self.dates = None
        
        # Cache for indicators
        self._indicator_cache = {}
    
    def _get_cached(self, key, calc_fn):
        if key not in self._indicator_cache:
            self._indicator_cache[key] = calc_fn()
        return self._indicator_cache[key]
    
    def evaluate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Birey fitness'ını hesapla - strateji bazlı"""
        try:
            if self.strategy_index == 0:
                return self._evaluate_strategy1(params)
            else:
                return self._evaluate_strategy2(params)
        except Exception as e:
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'fitness': -999999}
    
    def _evaluate_strategy1(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 1 (Gatekeeper) için fitness hesapla"""
        from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig
        
        # ScoreConfig oluştur
        config = ScoreConfig(
            ars_period=int(params.get('ars_period', 3)),
            ars_k=float(params.get('ars_k', 0.01)),
            adx_period=int(params.get('adx_period', 17)),
            adx_threshold=float(params.get('adx_threshold', 25.0)),
            macdv_short=int(params.get('macdv_short', 13)),
            macdv_long=int(params.get('macdv_long', 28)),
            macdv_signal=int(params.get('macdv_signal', 8)),
            macdv_threshold=float(params.get('macdv_threshold', 0.0)),
            netlot_period=int(params.get('netlot_period', 5)),
            netlot_threshold=float(params.get('netlot_threshold', 20.0)),
            ars_mesafe_threshold=float(params.get('ars_mesafe_threshold', 0.25)),
            bb_period=int(params.get('bb_period', 20)),
            bb_std=float(params.get('bb_std', 2.0)),
            bb_width_multiplier=float(params.get('bb_width_multiplier', 0.8)),
            bb_avg_period=int(params.get('bb_avg_period', 50)),
            yatay_ars_bars=int(params.get('yatay_ars_bars', 10)),
            yatay_adx_threshold=float(params.get('yatay_adx_threshold', 20.0)),
            filter_score_threshold=int(params.get('filter_score_threshold', 2)),
            min_score=int(params.get('min_score', 3)),
            exit_score=int(params.get('exit_score', 3)),
        )
        
        # Strateji çalıştır
        strategy = ScoreBasedStrategy(
            opens=self.opens.tolist(),
            highs=self.highs.tolist(),
            lows=self.lows.tolist(),
            closes=self.closes.tolist(),
            volumes=self.volumes.tolist(),
            dates=self.dates,
            config=config
        )
        
        result = strategy.run_backtest()
        
        # Fitness hesapla
        net_profit = result.get('net_profit', 0)
        pf = result.get('profit_factor', 0)
        max_dd = result.get('max_drawdown', 0)
        trades = result.get('total_trades', 0)
        
        fitness = net_profit
        if pf > 1.5:
            fitness *= (1 + (pf - 1) * 0.1)
        if max_dd > 0:
            fitness *= (1 - min(0.5, max_dd / 10000))
        if trades < 10:
            fitness *= 0.5
        
        return {
            'net_profit': net_profit,
            'trades': trades,
            'pf': pf,
            'max_dd': max_dd,
            'fitness': fitness
        }
    
    def _evaluate_strategy2(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 2 (ARS Trend v2) için fitness hesapla - inline backtest"""
        try:
            # Parametreleri çıkar (yeni isimlerle)
            ars_ema = int(params.get('ars_ema_period', 3))
            ars_atr_p = int(params.get('ars_atr_period', 10))
            ars_atr_m = float(params.get('ars_atr_mult', 0.5))
            mom_p = int(params.get('momentum_period', 5))
            brk_p = int(params.get('breakout_period', 10))
            mfi_p = int(params.get('mfi_period', 14))
            mfi_hhv_p = int(params.get('mfi_hhv_period', 14))
            mfi_llv_p = int(params.get('mfi_llv_period', 14))
            vol_hhv_p = int(params.get('volume_hhv_period', 14))
            atr_exit_p = int(params.get('atr_exit_period', 14))
            atr_sl_mult = float(params.get('atr_sl_mult', 2.0))
            atr_tp_mult = float(params.get('atr_tp_mult', 5.0))
            atr_trail_mult = float(params.get('atr_trail_mult', 2.0))
            exit_confirm_bars = int(params.get('exit_confirm_bars', 2))
            exit_confirm_mult = float(params.get('exit_confirm_mult', 1.0))
            volume_mult = float(params.get('volume_mult', 0.8))
            
            # İndikatörleri hesapla (cached)
            ars = self._get_cached(
                f'ars_{ars_ema}_{ars_atr_p}_{ars_atr_m:.2f}',
                lambda: np.array(ARS_Dynamic(
                    self.typical.tolist(), self.highs.tolist(), 
                    self.lows.tolist(), self.closes.tolist(),
                    ema_period=ars_ema, atr_period=ars_atr_p, atr_mult=ars_atr_m,
                    min_k=0.002, max_k=0.015
                ))
            )
            
            # ATR (çıkış için)
            from src.indicators.core import ATR as ATR_fn, EMA
            atr = self._get_cached(f'atr_{atr_exit_p}', lambda: np.array(ATR_fn(
                self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), atr_exit_p)))
            
            # EMA (dinamikK hesaplaması için)
            ars_ema_arr = self._get_cached(f'ema_{ars_ema}', lambda: np.array(EMA(self.typical.tolist(), ars_ema)))
            
            # dinamikK hesapla
            n = self.n
            dinamikK = np.zeros(n)
            for i in range(n):
                if ars_ema_arr[i] > 0:
                    dinamikK[i] = (atr[i] / ars_ema_arr[i]) * ars_atr_m
                    dinamikK[i] = max(0.002, min(0.015, dinamikK[i]))
                else:
                    dinamikK[i] = 0.002
            
            mom = self._get_cached(f'mom_{mom_p}', lambda: np.array(Momentum(self.closes.tolist(), mom_p)))
            
            # Tek periyot için HHV/LLV (basitleştirildi)
            hhv = self._get_cached(f'hhv_{brk_p}', lambda: np.array(HHV(self.highs.tolist(), brk_p)))
            llv = self._get_cached(f'llv_{brk_p}', lambda: np.array(LLV(self.lows.tolist(), brk_p)))
            
            mfi = self._get_cached(f'mfi_{mfi_p}', lambda: np.array(MoneyFlowIndex(
                self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), self.volumes.tolist(), mfi_p
            )))
            mfi_hhv = self._get_cached(f'mfi_hhv_{mfi_p}_{mfi_hhv_p}', lambda: np.array(HHV(mfi.tolist(), mfi_hhv_p)))
            mfi_llv = self._get_cached(f'mfi_llv_{mfi_p}_{mfi_llv_p}', lambda: np.array(LLV(mfi.tolist(), mfi_llv_p)))
            vol_hhv = self._get_cached(f'vol_hhv_{vol_hhv_p}', lambda: np.array(HHV(self.volumes.tolist(), vol_hhv_p)))
            
            # Backtest
            result = self._run_backtest_s2(
                ars, atr, dinamikK, mom, hhv, llv,
                mfi, mfi_hhv, mfi_llv, vol_hhv,
                atr_sl_mult, atr_tp_mult, atr_trail_mult,
                exit_confirm_bars, exit_confirm_mult, volume_mult, brk_p
            )
            
            return result
            
        except Exception as e:
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'fitness': -999999}
    
    def _run_backtest_s2(self, ars, atr, dinamikK, mom, hhv, llv,
                      mfi, mfi_hhv, mfi_llv, vol_hhv,
                      atr_sl, atr_tp, atr_trail,
                      exit_confirm_bars, exit_confirm_mult, volume_mult, brk_p) -> Dict[str, float]:
        """Strateji 2 için hızlı backtest"""
        n = self.n
        closes = self.closes
        highs = self.highs
        lows = self.lows
        volumes = self.volumes
        
        # Trend yönü
        trend = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if closes[i] > ars[i]:
                trend[i] = 1
            elif closes[i] < ars[i]:
                trend[i] = -1
        
        pos = 0
        entry_price = 0.0
        extreme_price = 0.0
        entry_atr = 0.0
        bars_against_trend = 0
        
        gross_profit = 0.0
        gross_loss = 0.0
        trades = 0
        max_dd = 0.0
        peak_equity = 0.0
        current_equity = 0.0
        
        current_trend = 0
        warmup = max(brk_p, 60)
        
        for i in range(warmup, n):
            if trend[i] != 0:
                current_trend = trend[i]
            
            current_dinamikK = dinamikK[i]
            
            # ========== EXIT LOGIC ==========
            if pos == 1:
                if closes[i] > extreme_price:
                    extreme_price = closes[i]
                
                exit_signal = False
                
                # 1. Çift Teyitli Trend Dönüşü
                if current_trend == -1:
                    bars_against_trend += 1
                    distance_threshold = ars[i] * (1 - current_dinamikK * exit_confirm_mult)
                    distance_ok = closes[i] < distance_threshold
                    if bars_against_trend >= exit_confirm_bars and distance_ok:
                        exit_signal = True
                else:
                    bars_against_trend = 0
                
                # 2. Take Profit (ATR bazlı)
                if closes[i] >= entry_price + entry_atr * atr_tp:
                    exit_signal = True
                
                # 3. Stop Loss (ATR bazlı)
                if closes[i] <= entry_price - entry_atr * atr_sl:
                    exit_signal = True
                
                # 4. Trailing Stop
                if closes[i] < extreme_price - entry_atr * atr_trail:
                    exit_signal = True
                
                if exit_signal:
                    pnl = closes[i] - entry_price
                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    current_equity += pnl
                    pos = 0
                    bars_against_trend = 0
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    dd = peak_equity - current_equity
                    if dd > max_dd:
                        max_dd = dd
                        
            elif pos == -1:
                if closes[i] < extreme_price:
                    extreme_price = closes[i]
                
                exit_signal = False
                
                # 1. Çift Teyitli Trend Dönüşü
                if current_trend == 1:
                    bars_against_trend += 1
                    distance_threshold = ars[i] * (1 + current_dinamikK * exit_confirm_mult)
                    distance_ok = closes[i] > distance_threshold
                    if bars_against_trend >= exit_confirm_bars and distance_ok:
                        exit_signal = True
                else:
                    bars_against_trend = 0
                
                # 2. Take Profit (ATR bazlı)
                if closes[i] <= entry_price - entry_atr * atr_tp:
                    exit_signal = True
                
                # 3. Stop Loss (ATR bazlı)
                if closes[i] >= entry_price + entry_atr * atr_sl:
                    exit_signal = True
                
                # 4. Trailing Stop
                if closes[i] > extreme_price + entry_atr * atr_trail:
                    exit_signal = True
                
                if exit_signal:
                    pnl = entry_price - closes[i]
                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    current_equity += pnl
                    pos = 0
                    bars_against_trend = 0
                    
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    dd = peak_equity - current_equity
                    if dd > max_dd:
                        max_dd = dd
            
            # ========== ENTRY LOGIC (simplified single period) ==========
            if pos == 0:
                if current_trend == 1:
                    price_ok = closes[i] > hhv[i-1] or highs[i] > hhv[i-1]
                    mom_ok = mom[i] > 100
                    mfi_ok = mfi[i] >= mfi_hhv[i-1]
                    vol_ok = volumes[i] >= vol_hhv[i-1] * volume_mult
                    
                    if price_ok and mom_ok and mfi_ok and vol_ok:
                        pos = 1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        entry_atr = atr[i] if atr[i] > 0 else 1.0
                        bars_against_trend = 0
                        trades += 1
                        
                elif current_trend == -1:
                    price_ok = closes[i] < llv[i-1] or lows[i] < llv[i-1]
                    mom_ok = mom[i] < 100
                    mfi_ok = mfi[i] <= mfi_llv[i-1]
                    vol_ok = volumes[i] >= vol_hhv[i-1] * volume_mult
                    
                    if price_ok and mom_ok and mfi_ok and vol_ok:
                        pos = -1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        entry_atr = atr[i] if atr[i] > 0 else 1.0
                        bars_against_trend = 0
                        trades += 1
        
        net_profit = gross_profit - gross_loss
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
        
        # Fitness hesapla (çok faktörlü)
        fitness = net_profit
        if pf > 1.5:
            fitness *= (1 + (pf - 1) * 0.1)
        if max_dd > 0:
            fitness *= (1 - min(0.5, max_dd / 10000))
        if trades < 10:
            fitness *= 0.5
        
        return {
            'net_profit': net_profit,
            'trades': trades,
            'pf': pf,
            'max_dd': max_dd,
            'fitness': fitness
        }


# ==============================================================================
# GENETIC ALGORITHM ENGINE
# ==============================================================================
class GeneticOptimizer:
    """Genetik Algoritma Optimizasyon Motoru - Her iki strateji için"""
    
    def __init__(self, df: pd.DataFrame, config: Optional[GeneticConfig] = None, strategy_index: int = 1):
        """
        Args:
            df: Veri DataFrame'i
            config: Genetik algoritma konfigürasyonu
            strategy_index: 0 = Strateji 1, 1 = Strateji 2
        """
        self.df = df
        self.config = config or GeneticConfig()
        self.strategy_index = strategy_index
        self.param_space = ParameterSpace(strategy_index)
        self.evaluator = FitnessEvaluator(df, strategy_index)
        
        self.population: List[np.ndarray] = []
        self.fitness_scores: List[float] = []
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = -float('inf')
        self.best_params: Optional[Dict] = None
        
        self.generation_history: List[Dict] = []
        
    def initialize_population(self):
        """İlk popülasyonu oluştur"""
        self.population = [
            self.param_space.random_individual() 
            for _ in range(self.config.population_size)
        ]
        
    def evaluate_population(self):
        """Tüm popülasyonu değerlendir"""
        self.fitness_scores = []
        
        for individual in self.population:
            params = self.param_space.decode(individual)
            result = self.evaluator.evaluate(params)
            self.fitness_scores.append(result['fitness'])
            
            # En iyi bireyi güncelle
            if result['fitness'] > self.best_fitness:
                self.best_fitness = result['fitness']
                self.best_individual = individual.copy()
                self.best_params = params.copy()
                self.best_result = result
                
    def tournament_selection(self) -> np.ndarray:
        """Turnuva seçimi"""
        indices = random.sample(range(len(self.population)), self.config.tournament_size)
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx].copy()
    
    def evolve(self):
        """Bir nesil evrimleştir"""
        new_population = []
        
        # Elitizm - en iyi bireyleri koru
        n_elite = max(1, int(self.config.population_size * self.config.elite_ratio))
        elite_indices = np.argsort(self.fitness_scores)[-n_elite:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Yeni bireyler oluştur
        while len(new_population) < self.config.population_size:
            # Seçim
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Çaprazlama
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.param_space.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutasyon
            if random.random() < self.config.mutation_rate:
                child1 = self.param_space.mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self.param_space.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)
        
        self.population = new_population[:self.config.population_size]
        
    def run(self, verbose: bool = True) -> Dict:
        """Optimizasyonu çalıştır"""
        start_time = time()
        
        if verbose:
            print(f"Genetik Algoritma Başlıyor...")
            print(f"  Popülasyon: {self.config.population_size}")
            print(f"  Nesil: {self.config.generations}")
            print(f"  Parametre: {self.param_space.n_params}")
        
        # İlk popülasyon
        self.initialize_population()
        self.evaluate_population()
        
        no_improve_count = 0
        prev_best = self.best_fitness
        
        for gen in range(self.config.generations):
            # Evrim
            self.evolve()
            self.evaluate_population()
            
            # İstatistikler
            mean_fitness = np.mean(self.fitness_scores)
            max_fitness = max(self.fitness_scores)
            
            self.generation_history.append({
                'generation': gen + 1,
                'mean_fitness': mean_fitness,
                'max_fitness': max_fitness,
                'best_fitness': self.best_fitness
            })
            
            if verbose and (gen + 1) % 5 == 0:
                print(f"  Nesil {gen+1:3d}: Best={self.best_fitness:,.0f} Mean={mean_fitness:,.0f}")
            
            # Erken durdurma kontrolü
            improvement = (self.best_fitness - prev_best) / max(abs(prev_best), 1)
            if improvement < self.config.min_improvement:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_best = self.best_fitness
            
            if no_improve_count >= self.config.early_stop_generations:
                if verbose:
                    print(f"  Erken Durdurma (Nesil {gen+1})")
                break
        
        elapsed = time() - start_time
        
        result = {
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'best_result': self.best_result,
            'generations_run': len(self.generation_history),
            'elapsed_time': elapsed,
            'history': self.generation_history
        }
        
        if verbose:
            print(f"\nSonuç:")
            print(f"  Süre: {elapsed:.1f}sn")
            print(f"  Best Fitness: {self.best_fitness:,.0f}")
            print(f"  Net Kar: {self.best_result['net_profit']:,.0f}")
            print(f"  PF: {self.best_result['pf']:.2f}")
            print(f"  İşlem: {self.best_result['trades']}")
            print(f"  MaxDD: {self.best_result['max_dd']:,.0f}")
            print(f"\nEn İyi Parametreler:")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")
        
        return result


# ==============================================================================
# MAIN
# ==============================================================================
def load_data() -> pd.DataFrame:
    """Veri yükle"""
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    print("Veri Yükleniyor...")
    
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
    df.dropna(inplace=True)
    
    print(f"Veri Hazır: {len(df)} Bar")
    return df


def run_genetic_optimization():
    """Ana fonksiyon"""
    df = load_data()
    
    config = GeneticConfig(
        population_size=50,
        generations=30,
        elite_ratio=0.1,
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=5,
        early_stop_generations=8
    )
    
    optimizer = GeneticOptimizer(df, config)
    result = optimizer.run(verbose=True)
    
    # Sonuçları kaydet
    result_df = pd.DataFrame([{
        **result['best_params'],
        **result['best_result']
    }])
    
    os.makedirs("d:/Projects/IdealQuant/results", exist_ok=True)
    result_df.to_csv("d:/Projects/IdealQuant/results/genetic_optimizer_result.csv", index=False)
    print("\nSonuç kaydedildi: results/genetic_optimizer_result.csv")
    
    return result


if __name__ == "__main__":
    try:
        run_genetic_optimization()
    except KeyboardInterrupt:
        print("\nİptal edildi.")
