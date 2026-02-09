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
from typing import List, Tuple, Dict, Any, Optional, Callable
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
    'macdv_threshold': (0.0, 5.0, 0.5, False),  # MACDV sinyal farkı eşiği (gerçekçi aralık)
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

# ARS Pulse Strategy (Strict Gatekeeper) - Strategy 3
STRATEGY3_PARAMS = {
    'ema_period': (1, 1000, 1, True),
    'k_value': (0.1, 10.0, 0.1, False),
    'macdv_k': (8, 21, 1, True),
    'macdv_u': (20, 45, 1, True),
    'macdv_sig': (5, 15, 1, True),
    'netlot_period': (3, 10, 1, True),
    'adx_th': (15, 45, 5, True),
    'netlot_th': (5, 45, 5, True),
}


class ParameterSpace:
    """Parametre uzayı tanımı - Her iki strateji için"""
    def __init__(self, strategy_index: int = 1, narrowed_ranges: dict = None):
        """
        Args:
            strategy_index: 0=Gatekeeper, 1=ARS Trend v2, 2=ARS Pulse
            narrowed_ranges: Cascade modunda dar araliklar {param_name: (min, max)}
        """
        self.strategy_index = strategy_index
        # Orijinal parametreleri kopyala
        if strategy_index == 0:
            base_params = STRATEGY1_PARAMS
        elif strategy_index == 1:
            base_params = STRATEGY2_PARAMS
        else:
            base_params = STRATEGY3_PARAMS
            
        self.params = {k: list(v) for k, v in base_params.items()}  # Mutable copy
        
        # Cascade: Dar aralik varsa uygula
        if narrowed_ranges:
            self._apply_narrowed_ranges(narrowed_ranges)
        
        self.param_names = list(self.params.keys())
        self.n_params = len(self.param_names)
    
    def _apply_narrowed_ranges(self, narrowed_ranges: dict):
        """Cascade modunda dar araliklari uygula"""
        for param_name, (new_min, new_max) in narrowed_ranges.items():
            if param_name in self.params:
                original = self.params[param_name]
                # original: [min, max, step, is_int]
                orig_min, orig_max, step, is_int = original
                
                # Yeni araligi orijinal sinirlar icinde tut
                final_min = max(new_min, orig_min)
                final_max = min(new_max, orig_max)
                
                # Gecerlilik kontrolu
                if final_min <= final_max:
                    self.params[param_name] = [final_min, final_max, step, is_int]
                    print(f"  [CASCADE] {param_name}: [{orig_min:.4g}-{orig_max:.4g}] => [{final_min:.4g}-{final_max:.4g}]")
        
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
                    new_val = np.clip(new_genes[i] + delta, min_val, max_val)
                    # Period parametreleri için integer zorunluluğu
                    new_genes[i] = int(round(new_val)) if is_int else new_val
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
    
    def __init__(self, df: pd.DataFrame, strategy_index: int = 1, commission: float = 0.0, slippage: float = 0.0):
        """
        Args:
            df: Veri DataFrame'i
            strategy_index: 0 = Strateji 1 (Gatekeeper), 1 = Strateji 2 (ARS Trend v2)
            commission: İşlem başı komisyon
            slippage: İşlem başı kayma
        """
        self.df = df
        self.strategy_index = strategy_index
        self.commission = commission
        self.slippage = slippage
        
        # Hem İngilizce hem Türkçe kolon isimlerini destekle
        open_col = 'Acilis' if 'Acilis' in df.columns else 'Open'
        high_col = 'Yuksek' if 'Yuksek' in df.columns else 'High'
        low_col = 'Dusuk' if 'Dusuk' in df.columns else 'Low'
        close_col = 'Kapanis' if 'Kapanis' in df.columns else 'Close'
        vol_col = 'Lot' if 'Lot' in df.columns else 'Volume'
        
        self.opens = df[open_col].to_numpy().flatten()
        self.highs = df[high_col].to_numpy().flatten()
        self.lows = df[low_col].to_numpy().flatten()
        self.closes = df[close_col].to_numpy().flatten()
        self.typical = df['Tipik'].values.flatten() if 'Tipik' in df.columns else ((df[high_col] + df[low_col] + df[close_col]) / 3).values.flatten()
        self.volumes = df[vol_col].values.flatten()
        self.lots = df[vol_col].values.flatten()
        self.n = len(self.closes)
        
        # Tarih bilgisi
        if 'DateTime' in df.columns:
            self.dates = df['DateTime'].tolist()
            self.times = df['DateTime'].tolist()
        else:
            self.dates = None
            self.times = None
        
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
            elif self.strategy_index == 1:
                return self._evaluate_strategy2(params)
            else:
                return self._evaluate_strategy3(params)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: Genetic Eval Failed: {str(e)}")
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'fitness': -999999}

    def _evaluate_strategy3(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 3 (ARS Pulse) için fitness hesapla"""
        from src.strategies.ars_pulse_strategy import ARSPulseStrategy
        from src.optimization.hybrid_group_optimizer import fast_backtest
        from src.optimization.fitness import quick_fitness
        
        # Data preparation for ARSPulseStrategy
        closes = self.closes
        highs = self.highs
        lows = self.lows
        opens = self.opens
        df = pd.DataFrame({
            'Kapanis': closes,
            'Yuksek': highs,
            'Dusuk': lows,
            'Acilis': opens
        })
        
        # Run Strategy
        strat = ARSPulseStrategy(**params)
        signals, _ = strat.run(df)
        
        # Convert signals to entry/exit for fast_backtest
        # (This is a simplification, ARSPulseStrategy.run already handles the state machine)
        # But fast_backtest expects raw signals + exits.
        # Actually, ARSPulseStrategy returns position signals (1, -1, 0).
        # We need to adapt it for the core backtest engine.
        
        # Position-to-Trade calculation logic
        np_val, trades, pf, dd, sharpe = fast_backtest(closes, signals, (signals == 0), (signals == 0), self.commission, self.slippage)
        
        fitness = quick_fitness(
            np_val, pf, dd, trades,
            initial_capital=10000.0,
            commission=self.commission,
            slippage=self.slippage
        )
        
        return {
            'net_profit': np_val,
            'trades': trades,
            'pf': pf,
            'max_dd': dd,
            'fitness': fitness
        }
    
    def _evaluate_strategy1(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 1 (Gatekeeper) için fitness hesapla"""
        from src.strategies.score_based import ScoreBasedStrategy
        from src.optimization.hybrid_group_optimizer import fast_backtest
        from src.optimization.fitness import quick_fitness
        
        # Kendi cache'imizi oluştur (from_config_dict için)
        class SimpleCache:
            def __init__(self, evaluator):
                self.opens = evaluator.opens
                self.highs = evaluator.highs
                self.lows = evaluator.lows
                self.closes = evaluator.closes
                self.typical = evaluator.typical
                self.lots = evaluator.volumes
                self.volumes = evaluator.volumes
                self.dates = evaluator.dates
                self.times = evaluator.dates
                self.n = evaluator.n
                self.df = evaluator.df
        
        cache = SimpleCache(self)
        
        # Strateji oluştur ve sinyal üret
        strategy = ScoreBasedStrategy.from_config_dict(cache, params)
        signals, exits_long, exits_short = strategy.generate_all_signals()
        
        # Backtest
        np_val, trades, pf, dd, sharpe = fast_backtest(self.closes, signals, exits_long, exits_short, self.commission, self.slippage)
        
        # Fitness hesapla (fitness.py'deki standart mantık)
        fitness = quick_fitness(
            np_val, pf, dd, trades, 
            initial_capital=10000.0,
            commission=self.commission,
            slippage=self.slippage
        )
        
        return {
            'net_profit': np_val,
            'trades': trades,
            'pf': pf,
            'max_dd': dd,
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
                      exit_confirm_bars, exit_confirm_mult, volume_mult, brk_p,
                      commission: float = 0.0, slippage: float = 0.0) -> Dict[str, float]:
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
        
        # Sharpe hesabı için getirileri tut
        trade_returns = []

        
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
                    trade_returns.append(pnl)

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
                    trade_returns.append(pnl)

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
        
        # Maliyetleri düş
        cost_per_trade = commission + slippage
        net_profit = gross_profit - gross_loss - (trades * cost_per_trade)
        pf = (gross_profit / (gross_loss + trades * cost_per_trade)) if (gross_loss + trades * cost_per_trade) > 0 else 999
        
        # Fitness hesapla (fitness.py'deki standart mantık)
        # Sharpe hesapla
        from src.optimization.fitness import quick_fitness, calculate_sharpe
        sharpe = 0.0
        if len(trade_returns) > 1:
            sharpe = calculate_sharpe(np.array(trade_returns))

        fitness = quick_fitness(
            net_profit + (trades * cost_per_trade), # quick_fitness'a brüt karı veriyoruz, o maliyeti düşecek
            pf, max_dd, trades,
            sharpe=sharpe,
            commission=commission,
            slippage=slippage
        )
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
# Global variable for pool workers to avoid data copying (pickling)
_global_evaluator: Optional['FitnessEvaluator'] = None

def _init_pool(df, strategy_index, commission=0.0, slippage=0.0):
    global _global_evaluator
    _global_evaluator = FitnessEvaluator(df, strategy_index, commission, slippage)

def _evaluate_individual(individual_and_param_space):
    individual, param_space = individual_and_param_space
    params = param_space.decode(individual)
    result = _global_evaluator.evaluate(params)
    return individual, result

class GeneticOptimizer:
    """Genetik Algoritma Optimizasyon Motoru - Her iki strateji için"""
    
    def __init__(self, df: pd.DataFrame, config: Optional[GeneticConfig] = None, 
                 strategy_index: int = 1, n_parallel: int = 4,
                 commission: float = 0.0, slippage: float = 0.0,
                 is_cancelled_callback: Optional[Callable[[], bool]] = None,
                 narrowed_ranges: dict = None):
        """
        Args:
            df: Veri DataFrame'i
            config: Genetik algoritma konfigürasyonu
            strategy_index: 0 = Strateji 1, 1 = Strateji 2
            n_parallel: Paralel işlem sayısı
            narrowed_ranges: Cascade modu için dar parametre aralıkları
        """
        self.df = df
        self.config = config or GeneticConfig()
        self.strategy_index = strategy_index
        self.n_parallel = n_parallel
        self.commission = commission
        self.slippage = slippage
        self.param_space = ParameterSpace(strategy_index, narrowed_ranges)  # Cascade destegi
        self.evaluator = FitnessEvaluator(df, strategy_index, commission, slippage)
        self.is_cancelled_callback = is_cancelled_callback
        
        self.population: List[np.ndarray] = []
        self.fitness_scores: List[float] = []
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = -float('inf')
        self.best_params: Optional[Dict] = None
        self.best_result: Optional[Dict] = None
        
        self.generation_history: List[Dict] = []
        self.on_generation_complete = None # Callback function(gen, max_gen, best_fitness)
        
    def initialize_population(self):
        """İlk popülasyonu oluştur"""
        self.population = [
            self.param_space.random_individual() 
            for _ in range(self.config.population_size)
        ]
        
    def evaluate_population(self, pool: Optional[Pool] = None):
        """Tüm popülasyonu değerlendir"""
        self.fitness_scores = [0] * len(self.population)
        
        if self.n_parallel > 1:
            tasks = [(ind, self.param_space) for ind in self.population]
            
            if pool:
                results = pool.map(_evaluate_individual, tasks)
            else:
                with Pool(processes=self.n_parallel, initializer=_init_pool, 
                         initargs=(self.df, self.strategy_index, self.commission, self.slippage)) as p:
                    results = p.map(_evaluate_individual, tasks)
                
            for i, (individual, result) in enumerate(results):
                self.fitness_scores[i] = result['fitness']
                if result['fitness'] > self.best_fitness:
                    self.best_fitness = result['fitness']
                    self.best_individual = individual.copy()
                    self.best_params = self.param_space.decode(individual)
                    self.best_result = result.copy()
        else:
            # Single-threaded
            for i, individual in enumerate(self.population):
                params = self.param_space.decode(individual)
                result = self.evaluator.evaluate(params)
                self.fitness_scores[i] = result['fitness']
                
                if result['fitness'] > self.best_fitness:
                    self.best_fitness = result['fitness']
                    self.best_individual = individual.copy()
                    self.best_params = params.copy()
                    self.best_result = result.copy()
                
    def tournament_selection(self) -> np.ndarray:
        """Turnuva seçimi"""
        indices = random.sample(range(len(self.population)), self.config.tournament_size)
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx].copy()
    
    def evolve(self):
        """Bir nesil evrimleştir"""
        new_population = []
        
        # Elitizm
        n_elite = max(1, int(self.config.population_size * self.config.elite_ratio))
        elite_indices = np.argsort(self.fitness_scores)[-n_elite:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Yeni bireyler oluştur
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.param_space.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
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
            print(f"Genetik Algoritma Basliyor...")
            print(f"  Populasyon: {self.config.population_size}")
            print(f"  Nesil: {self.config.generations}")
            print(f"  Paralel: {self.n_parallel}")
        
        # Pool'u bir kez oluştur ve tüm run boyunca kullan
        pool = None
        if self.n_parallel > 1:
            pool = Pool(processes=self.n_parallel, initializer=_init_pool, initargs=(self.df, self.strategy_index))
            
        try:
            # İlk popülasyon
            self.initialize_population()
            self.evaluate_population(pool=pool)
            
            no_improve_count = 0
            prev_best = self.best_fitness
            
            for gen in range(self.config.generations):
                # İptal kontrolü
                if self.is_cancelled_callback and self.is_cancelled_callback():
                    if verbose: print("Optimizasyon kullanici tarafindan durduruldu.")
                    break

                # Evrim
                self.evolve()
                self.evaluate_population(pool=pool)
                
                # Progress callback
                if self.on_generation_complete:
                    self.on_generation_complete(gen + 1, self.config.generations, self.best_fitness)
                
                if verbose and (gen + 1) % 5 == 0:
                    print(f"  Nesil {gen+1:3d}: Best={self.best_fitness:,.0f}")
                
                # Erken durdurma kontrolü
                improvement = (self.best_fitness - prev_best) / max(abs(prev_best), 1)
                if improvement < self.config.min_improvement:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                prev_best = self.best_fitness
                
                if no_improve_count >= self.config.early_stop_generations:
                    break
        finally:
            if pool:
                pool.close()
                pool.join()
        
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
            print(f"\nSonuc:")
            print(f"  Sure: {elapsed:.1f}sn")
            print(f"  Best Fitness: {self.best_fitness:,.0f}")
            print(f"  Net Kar: {self.best_result['net_profit']:,.0f}")
            print(f"  PF: {self.best_result['pf']:.2f}")
            print(f"  Islem: {self.best_result['trades']}")
            print(f"  MaxDD: {self.best_result['max_dd']:,.0f}")
            print(f"\nEn Iyi Parametreler:")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")
        
        return result


# ==============================================================================
# MAIN
# ==============================================================================
def load_data() -> pd.DataFrame:
    """Veri yükle"""
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    print("Veri yukleniyor...")
    
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
    df.dropna(inplace=True)
    
    print(f"Veri Hazir: {len(df)} Bar")
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
    print("\nSonuc kaydedildi: results/genetic_optimizer_result.csv")
    
    return result


if __name__ == "__main__":
    try:
        run_genetic_optimization()
    except KeyboardInterrupt:
        print("\nIptal edildi.")
