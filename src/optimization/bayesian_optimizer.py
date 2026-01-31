# -*- coding: utf-8 -*-
"""
Bayesian Optimizer for Strategy 2 (ARS Trend v2)
================================================
Optuna kullanarak Bayesian Optimization.
Genetik Algoritma ve Grid Search'e alternatif.

Avantajlar:
- Daha az değerlendirme ile iyi sonuç
- Akıllı arama (Explore vs Exploit)
- Overfitting riski düşük
"""

import sys
import os
import numpy as np
import pandas as pd
from time import time
from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler

# Proje kök dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, ATR, Momentum, HHV, LLV, ARS_Dynamic, MoneyFlowIndex
from src.optimization.fitness import quick_fitness, FitnessConfig


# ==============================================================================
# DATA & CACHE
# ==============================================================================
class IndicatorCache:
    """İndikatör cache - aynı hesaplamayı tekrarlamamak için"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.closes = df['Kapanis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.typical = df['Tipik'].values
        self.volumes = df['Lot'].values
        self.n = len(self.closes)
        self._cache = {}
    
    def get(self, key: str, calc_fn):
        if key not in self._cache:
            self._cache[key] = calc_fn()
        return self._cache[key]


def load_data() -> pd.DataFrame:
    """Veri yükle"""
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    for c in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
    df.dropna(inplace=True)
    return df


# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================
class BayesianObjective:
    """Optuna için objective fonksiyonu"""
    
    def __init__(self, cache: IndicatorCache, fitness_config: Optional[FitnessConfig] = None):
        self.cache = cache
        self.fitness_config = fitness_config or FitnessConfig()
        self.best_params = None
        self.best_result = None
        self.best_fitness = -float('inf')
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna tarafından çağrılır"""
        
        # Parametreleri tanımla
        params = {
            # ARS Dinamik
            'ars_ema': trial.suggest_int('ars_ema', 2, 10),
            'ars_atr_p': trial.suggest_int('ars_atr_p', 8, 20),
            'ars_atr_m': trial.suggest_float('ars_atr_m', 0.3, 1.2, step=0.1),
            
            # Momentum & Breakout
            'momentum_p': trial.suggest_int('momentum_p', 3, 15),
            'breakout_p1': trial.suggest_int('breakout_p1', 8, 20, step=2),
            'breakout_p2': trial.suggest_int('breakout_p2', 20, 40, step=5),
            'breakout_p3': trial.suggest_int('breakout_p3', 40, 80, step=10),
            
            # MFI & Volume
            'mfi_p': trial.suggest_int('mfi_p', 10, 20, step=2),
            'mfi_hhv_p': trial.suggest_int('mfi_hhv_p', 10, 20, step=2),
            'vol_p': trial.suggest_int('vol_p', 10, 20, step=2),
            
            # ATR çıkış
            'atr_exit_p': trial.suggest_int('atr_exit_p', 10, 20, step=2),
            'atr_sl_mult': trial.suggest_float('atr_sl_mult', 1.0, 2.5, step=0.5),
            'atr_tp_mult': trial.suggest_float('atr_tp_mult', 2.0, 5.0, step=0.5),
            'atr_trail_mult': trial.suggest_float('atr_trail_mult', 1.0, 2.0, step=0.5),
            
            # Çift teyitli çıkış
            'exit_confirm_bars': trial.suggest_int('exit_confirm_bars', 1, 3),
            'exit_confirm_mult': trial.suggest_float('exit_confirm_mult', 0.5, 1.5, step=0.5),
        }
        
        # Backtest yap
        result = self._run_backtest(params)
        
        # Fitness hesapla
        fitness = quick_fitness(
            result['net_profit'],
            result['pf'],
            result['max_dd'],
            result['trades'],
            result.get('win_count', 0),
            self.fitness_config.initial_capital
        )
        
        # En iyi sonucu sakla
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params = params.copy()
            self.best_result = result.copy()
        
        return fitness
    
    def _run_backtest(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Backtest çalıştır (Planlanmış Mimari v4.1)"""
        cache = self.cache
        closes = cache.closes
        highs = cache.highs
        lows = cache.lows
        typical = cache.typical
        volumes = cache.volumes
        n = cache.n
        
        # İndikatörleri hesapla (cached)
        ars = cache.get(
            f"ars_{params['ars_ema']}_{params['ars_atr_p']}_{params['ars_atr_m']:.1f}",
            lambda: np.array(ARS_Dynamic(
                typical.tolist(), highs.tolist(), lows.tolist(), closes.tolist(),
                ema_period=params['ars_ema'], atr_period=params['ars_atr_p'],
                atr_mult=params['ars_atr_m'], min_k=0.002, max_k=0.015
            ))
        )
        
        atr = cache.get(f"atr_{params['atr_exit_p']}", lambda: np.array(
            ATR(highs.tolist(), lows.tolist(), closes.tolist(), params['atr_exit_p'])))
        
        ars_ema_arr = cache.get(f"ema_{params['ars_ema']}", lambda: np.array(
            EMA(typical.tolist(), params['ars_ema'])))
        
        # dinamikK
        dinamikK = np.zeros(n)
        for i in range(n):
            if ars_ema_arr[i] > 0:
                dinamikK[i] = (atr[i] / ars_ema_arr[i]) * params['ars_atr_m']
                dinamikK[i] = max(0.002, min(0.015, dinamikK[i]))
        
        mom = cache.get(f"mom_{params['momentum_p']}", lambda: np.array(
            Momentum(closes.tolist(), params['momentum_p'])))
        
        # HHV/LLV (3 periyot)
        hhv1 = cache.get(f"hhv_{params['breakout_p1']}", lambda: np.array(HHV(highs.tolist(), params['breakout_p1'])))
        llv1 = cache.get(f"llv_{params['breakout_p1']}", lambda: np.array(LLV(lows.tolist(), params['breakout_p1'])))
        hhv2 = cache.get(f"hhv_{params['breakout_p2']}", lambda: np.array(HHV(highs.tolist(), params['breakout_p2'])))
        llv2 = cache.get(f"llv_{params['breakout_p2']}", lambda: np.array(LLV(lows.tolist(), params['breakout_p2'])))
        hhv3 = cache.get(f"hhv_{params['breakout_p3']}", lambda: np.array(HHV(highs.tolist(), params['breakout_p3'])))
        llv3 = cache.get(f"llv_{params['breakout_p3']}", lambda: np.array(LLV(lows.tolist(), params['breakout_p3'])))
        
        # MFI
        mfi = cache.get(f"mfi_{params['mfi_p']}", lambda: np.array(MoneyFlowIndex(
            highs.tolist(), lows.tolist(), closes.tolist(), volumes.tolist(), params['mfi_p'])))
        mfi_hhv = cache.get(f"mfi_hhv_{params['mfi_p']}_{params['mfi_hhv_p']}", 
                           lambda: np.array(HHV(mfi.tolist(), params['mfi_hhv_p'])))
        mfi_llv = cache.get(f"mfi_llv_{params['mfi_p']}_{params['mfi_hhv_p']}", 
                           lambda: np.array(LLV(mfi.tolist(), params['mfi_hhv_p'])))
        vol_hhv = cache.get(f"vol_hhv_{params['vol_p']}", lambda: np.array(HHV(volumes.tolist(), params['vol_p'])))
        
        # Çıkış parametreleri
        atr_sl = params['atr_sl_mult']
        atr_tp = params['atr_tp_mult']
        atr_trail = params['atr_trail_mult']
        exit_confirm_bars = params['exit_confirm_bars']
        exit_confirm_mult = params['exit_confirm_mult']
        brk_p2 = params['breakout_p2']
        brk_p3 = params['breakout_p3']
        
        # Trend yönü
        trend = np.zeros(n, dtype=int)
        trend[closes > ars] = 1
        trend[closes < ars] = -1
        
        # Backtest
        pos = 0
        entry_price = 0.0
        extreme_price = 0.0
        entry_atr = 0.0
        bars_against_trend = 0
        
        gross_profit = 0.0
        gross_loss = 0.0
        trades = 0
        win_count = 0
        max_dd = 0.0
        peak_equity = 0.0
        current_equity = 0.0
        current_trend = 0
        
        warmup = max(brk_p3, 60)
        
        for i in range(warmup, n):
            if trend[i] != 0:
                current_trend = trend[i]
            
            current_dinamikK = dinamikK[i]
            
            # EXIT
            if pos == 1:
                if closes[i] > extreme_price:
                    extreme_price = closes[i]
                
                exit_signal = False
                
                if current_trend == -1:
                    bars_against_trend += 1
                    distance_threshold = ars[i] * (1 - current_dinamikK * exit_confirm_mult)
                    if bars_against_trend >= exit_confirm_bars and closes[i] < distance_threshold:
                        exit_signal = True
                else:
                    bars_against_trend = 0
                
                if closes[i] >= entry_price + entry_atr * atr_tp:
                    exit_signal = True
                if closes[i] <= entry_price - entry_atr * atr_sl:
                    exit_signal = True
                if closes[i] < extreme_price - entry_atr * atr_trail:
                    exit_signal = True
                
                if exit_signal:
                    pnl = closes[i] - entry_price
                    if pnl > 0:
                        gross_profit += pnl
                        win_count += 1
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
                
                if current_trend == 1:
                    bars_against_trend += 1
                    distance_threshold = ars[i] * (1 + current_dinamikK * exit_confirm_mult)
                    if bars_against_trend >= exit_confirm_bars and closes[i] > distance_threshold:
                        exit_signal = True
                else:
                    bars_against_trend = 0
                
                if closes[i] <= entry_price - entry_atr * atr_tp:
                    exit_signal = True
                if closes[i] >= entry_price + entry_atr * atr_sl:
                    exit_signal = True
                if closes[i] > extreme_price + entry_atr * atr_trail:
                    exit_signal = True
                
                if exit_signal:
                    pnl = entry_price - closes[i]
                    if pnl > 0:
                        gross_profit += pnl
                        win_count += 1
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
            
            # ENTRY
            if pos == 0:
                if current_trend == 1:
                    price_ok = (closes[i] > hhv1[i-1] or highs[i] > hhv1[i-1]) or \
                               (i > brk_p2 and (closes[i] > hhv2[i-1] or highs[i] > hhv2[i-1])) or \
                               (i > brk_p3 and (closes[i] > hhv3[i-1] or highs[i] > hhv3[i-1]))
                    mom_ok = mom[i] > 100
                    mfi_ok = mfi[i] >= mfi_hhv[i-1]
                    vol_ok = volumes[i] >= vol_hhv[i-1] * 0.8
                    
                    if price_ok and mom_ok and mfi_ok and vol_ok:
                        pos = 1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        entry_atr = atr[i] if atr[i] > 0 else 1.0
                        bars_against_trend = 0
                        trades += 1
                        
                elif current_trend == -1:
                    price_ok = (closes[i] < llv1[i-1] or lows[i] < llv1[i-1]) or \
                               (i > brk_p2 and (closes[i] < llv2[i-1] or lows[i] < llv2[i-1])) or \
                               (i > brk_p3 and (closes[i] < llv3[i-1] or lows[i] < llv3[i-1]))
                    mom_ok = mom[i] < 100
                    mfi_ok = mfi[i] <= mfi_llv[i-1]
                    vol_ok = volumes[i] >= vol_hhv[i-1] * 0.8
                    
                    if price_ok and mom_ok and mfi_ok and vol_ok:
                        pos = -1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        entry_atr = atr[i] if atr[i] > 0 else 1.0
                        bars_against_trend = 0
                        trades += 1
        
        net_profit = gross_profit - gross_loss
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
        
        return {
            'net_profit': net_profit,
            'trades': trades,
            'pf': pf,
            'max_dd': max_dd,
            'win_count': win_count
        }


# ==============================================================================
# BAYESIAN OPTIMIZER
# ==============================================================================
class BayesianOptimizer:
    """Bayesian Optimization motoru (Optuna ile)"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        n_trials: int = 100,
        fitness_config: Optional[FitnessConfig] = None,
        seed: int = 42
    ):
        self.df = df
        self.n_trials = n_trials
        self.fitness_config = fitness_config or FitnessConfig()
        self.seed = seed
        
        self.cache = IndicatorCache(df)
        self.objective = BayesianObjective(self.cache, self.fitness_config)
        self.study = None
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Optimizasyonu çalıştır"""
        start_time = time()
        
        if verbose:
            print("Bayesian Optimizasyon Başlıyor...")
            print(f"  Deneme sayısı: {self.n_trials}")
            print(f"  Parametre sayısı: 16")
        
        # Optuna study oluştur
        sampler = TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='strategy2_bayesian'
        )
        
        # Optimizasyonu çalıştır
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )
        
        elapsed = time() - start_time
        
        result = {
            'best_params': self.objective.best_params,
            'best_fitness': self.objective.best_fitness,
            'best_result': self.objective.best_result,
            'n_trials': self.n_trials,
            'elapsed_time': elapsed,
            'study': self.study
        }
        
        if verbose:
            print(f"\nSonuç:")
            print(f"  Süre: {elapsed:.1f}sn")
            print(f"  Best Fitness: {self.objective.best_fitness:,.0f}")
            if self.objective.best_result:
                print(f"  Net Kar: {self.objective.best_result['net_profit']:,.0f}")
                print(f"  PF: {self.objective.best_result['pf']:.2f}")
                print(f"  İşlem: {self.objective.best_result['trades']}")
                print(f"  MaxDD: {self.objective.best_result['max_dd']:,.0f}")
            print(f"\nEn İyi Parametreler:")
            if self.objective.best_params:
                for k, v in self.objective.best_params.items():
                    print(f"  {k}: {v}")
        
        return result


# ==============================================================================
# MAIN
# ==============================================================================
def run_bayesian_optimization(n_trials: int = 100) -> Dict[str, Any]:
    """Ana fonksiyon"""
    print("Veri yükleniyor...")
    df = load_data()
    print(f"Veri hazır: {len(df)} bar")
    
    optimizer = BayesianOptimizer(df, n_trials=n_trials)
    result = optimizer.run(verbose=True)
    
    # Sonuçları kaydet
    if result['best_params']:
        result_df = pd.DataFrame([{
            **result['best_params'],
            **result['best_result']
        }])
        os.makedirs("d:/Projects/IdealQuant/results", exist_ok=True)
        result_df.to_csv("d:/Projects/IdealQuant/results/bayesian_optimizer_result.csv", index=False)
        print("\nSonuç kaydedildi: results/bayesian_optimizer_result.csv")
    
    return result


if __name__ == "__main__":
    try:
        run_bayesian_optimization(n_trials=100)
    except KeyboardInterrupt:
        print("\nİptal edildi.")
