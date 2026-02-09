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
from typing import Dict, Any, List, Optional, Tuple, Callable
import optuna
from optuna.samplers import TPESampler

# Proje kök dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, ATR, Momentum, HHV, LLV, ARS_Dynamic, MoneyFlowIndex
from src.optimization.fitness import quick_fitness, FitnessConfig, calculate_sharpe


# ==============================================================================
# DATA & CACHE
# ==============================================================================
class IndicatorCache:
    """İndikatör cache - aynı hesaplamayı tekrarlamamak için"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        # Hem İngilizce hem Türkçe kolon isimlerini destekle
        open_col = 'Acilis' if 'Acilis' in df.columns else 'Open'
        high_col = 'Yuksek' if 'Yuksek' in df.columns else 'High'
        low_col = 'Dusuk' if 'Dusuk' in df.columns else 'Low'
        close_col = 'Kapanis' if 'Kapanis' in df.columns else 'Close'
        vol_col = 'Lot' if 'Lot' in df.columns else 'Volume'
        
        self.opens = df[open_col].values.flatten()
        self.closes = df[close_col].values.flatten()
        self.highs = df[high_col].values.flatten()
        self.lows = df[low_col].values.flatten()
        self.typical = df['Tipik'].values.flatten() if 'Tipik' in df.columns else ((df[high_col] + df[low_col] + df[close_col]) / 3).values.flatten()
        self.volumes = df[vol_col].values.flatten()
        self.lots = df[vol_col].values.flatten()
        self.n = len(self.closes)
        
        # Tarih bilgisi (from_config_dict için)
        if 'DateTime' in df.columns:
            self.dates = df['DateTime'].tolist()
            self.times = df['DateTime'].tolist()
        else:
            self.dates = None
            self.times = None
        
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

# Genetik optimizer'dan parametre tanımlarını import et
from src.optimization.genetic_optimizer import STRATEGY1_PARAMS, STRATEGY2_PARAMS


class BayesianObjective:
    """Optuna için objective fonksiyonu - Her iki strateji için"""
    
    def __init__(self, cache: IndicatorCache, fitness_config: Optional[FitnessConfig] = None, 
                 strategy_index: int = 1, commission: float = 0.0, slippage: float = 0.0,
                 narrowed_ranges: dict = None):
        self.cache = cache
        self.fitness_config = fitness_config or FitnessConfig()
        self.strategy_index = strategy_index
        self.commission = commission
        self.slippage = slippage
        
        # Orijinal parametre tanimlarini kopyala
        if strategy_index == 0:
            base_params = STRATEGY1_PARAMS
        elif strategy_index == 1:
            base_params = STRATEGY2_PARAMS
        else:
            from src.optimization.genetic_optimizer import STRATEGY3_PARAMS
            base_params = STRATEGY3_PARAMS
            
        self.param_defs = {k: list(v) for k, v in base_params.items()}  # Mutable copy
        
        # Cascade: Dar aralik varsa uygula
        if narrowed_ranges:
            self._apply_narrowed_ranges(narrowed_ranges)
        
        self.best_params = None
        self.best_result = None
        self.best_fitness = -float('inf')
    
    def _apply_narrowed_ranges(self, narrowed_ranges: dict):
        """Cascade modunda dar araliklari uygula"""
        for param_name, (new_min, new_max) in narrowed_ranges.items():
            if param_name in self.param_defs:
                original = self.param_defs[param_name]
                orig_min, orig_max, step, is_int = original
                
                # Yeni araligi orijinal sinirlar icinde tut
                final_min = max(new_min, orig_min)
                final_max = min(new_max, orig_max)
                
                if final_min <= final_max:
                    self.param_defs[param_name] = [final_min, final_max, step, is_int]
                    print(f"  [CASCADE-BAY] {param_name}: [{orig_min:.4g}-{orig_max:.4g}] => [{final_min:.4g}-{final_max:.4g}]")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna tarafından çağrılır"""
        
        # Strateji bazlı parametre önerileri
        params = {}
        for name, (min_val, max_val, step, is_int) in self.param_defs.items():
            if is_int:
                params[name] = trial.suggest_int(name, int(min_val), int(max_val), step=int(step))
            else:
                params[name] = trial.suggest_float(name, min_val, max_val, step=step)
        
        # Strateji bazlı backtest/evaluate
        if self.strategy_index == 0:
            result = self._evaluate_strategy1(params)
        elif self.strategy_index == 1:
            result = self._evaluate_strategy2(params)
        else:
            result = self._evaluate_strategy3(params)
        
        # Fitness hesapla
        fitness = quick_fitness(
            result['net_profit'],
            result['pf'],
            result['max_dd'],
            result['trades'],
            result.get('win_count', 0),
            self.fitness_config.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
        
        # En iyi sonucu sakla
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params = params.copy()
            self.best_result = result.copy()
        
        return fitness
    
    def _evaluate_strategy3(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 3 (ARS Pulse) için fitness hesapla"""
        try:
            from src.strategies.ars_pulse_strategy import ARSPulseStrategy
            from src.optimization.hybrid_group_optimizer import fast_backtest
            from src.optimization.fitness import quick_fitness
            
            # Data preparation
            df = pd.DataFrame({
                'Kapanis': self.cache.closes,
                'Yuksek': self.cache.highs,
                'Dusuk': self.cache.lows,
                'Acilis': self.cache.opens
            })
            
            # Run Strategy
            strat = ARSPulseStrategy(**params)
            signals, _ = strat.run(df)
            
            # Trading days calculation
            trading_days = 252.0
            if self.cache.dates and len(self.cache.dates) > 1:
                try:
                    trading_days = (self.cache.dates[-1] - self.cache.dates[0]).days
                except: pass
            
            # Backtest
            np_val, trades, pf, dd, sharpe = fast_backtest(self.cache.closes, signals, (signals == 0), (signals == 0), self.commission, self.slippage, trading_days=trading_days)
            
            return {
                'net_profit': np_val,
                'trades': trades,
                'pf': pf,
                'max_dd': dd,
                'fitness': np_val, # Simple fitness
                'win_count': trades // 2
            }
        except Exception as e:
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'win_count': 0}
    
    def _evaluate_strategy1(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 1 için fitness hesapla - ScoreBasedStrategy kullanarak"""
        try:
            from src.strategies.score_based import ScoreBasedStrategy
            from src.optimization.hybrid_group_optimizer import fast_backtest
            
            # Strateji oluştur ve sinyal üret
            strategy = ScoreBasedStrategy.from_config_dict(self.cache, params)
            signals, exits_long, exits_short = strategy.generate_all_signals()
            
            # Trading days calculation
            trading_days = 252.0
            if self.cache.dates and len(self.cache.dates) > 1:
                try:
                    trading_days = (self.cache.dates[-1] - self.cache.dates[0]).days
                except: pass
            
            # Backtest
            np_val, trades, pf, dd, sharpe = fast_backtest(self.cache.closes, signals, exits_long, exits_short, self.commission, self.slippage, trading_days=trading_days)
            
            # Fitness hesapla
            fit = quick_fitness(np_val, pf, dd, trades, sharpe=sharpe, commission=self.commission, slippage=self.slippage)
            
            return {
                'net_profit': np_val,
                'trades': trades,
                'pf': pf,
                'max_dd': dd,
                'fitness': fit,
                'win_count': trades // 2  # Yaklaşık
            }
        except Exception as e:
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'win_count': 0}

    # ... (skipping _evaluate_strategy2 wrapper) ...

    def _run_backtest(self, params: Dict[str, Any], commission: float = 0.0, slippage: float = 0.0) -> Dict[str, float]:
        # ... (initial part of _run_backtest remains same until end) ...
        # ... (skipping to the end where sharpe is calculated) ...
        
        # Fitness hesapla
        from src.optimization.fitness import quick_fitness
        
        sharpe = 0.0
        if len(trade_returns) > 1:
            trading_days = 252.0
            if self.cache.dates and len(self.cache.dates) > 1:
                try:
                   trading_days = (self.cache.dates[-1] - self.cache.dates[0]).days
                except: pass
            
            if trading_days < 1: trading_days = 252.0
            trades_per_year_metric = len(trade_returns) * (252.0 / trading_days)
            
            sharpe = calculate_sharpe(np.array(trade_returns), trades_per_year=trades_per_year_metric)
            
        fit = quick_fitness(net_profit + (trades * cost_per_trade), pf, max_dd, trades, 
                           sharpe=sharpe,
                           commission=commission, slippage=slippage)
        
        return {
            'net_profit': net_profit,
            'trades': trades,
            'pf': pf,
            'max_dd': max_dd,
            'fitness': fit,
            'win_count': 0 # TODO: Gerçek win_count
        }
    
    def _evaluate_strategy2(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 2 için fitness hesapla - inline backtest"""
        # Yeni parametre isimlerini eski isimlere map'le (mevcut _run_backtest uyumluluğu için)
        mapped_params = {
            'ars_ema': params.get('ars_ema_period', 3),
            'ars_atr_p': params.get('ars_atr_period', 10),
            'ars_atr_m': params.get('ars_atr_mult', 0.5),
            'momentum_p': params.get('momentum_period', 5),
            'breakout_p1': params.get('breakout_period', 10),
            'breakout_p2': params.get('breakout_period', 10) + 10,  # Offset
            'breakout_p3': params.get('breakout_period', 10) + 30,  # Offset 
            'mfi_p': params.get('mfi_period', 14),
            'mfi_hhv_p': params.get('mfi_hhv_period', 14),
            'vol_p': params.get('volume_hhv_period', 14),
            'atr_exit_p': params.get('atr_exit_period', 14),
            'atr_sl_mult': params.get('atr_sl_mult', 2.0),
            'atr_tp_mult': params.get('atr_tp_mult', 5.0),
            'atr_trail_mult': params.get('atr_trail_mult', 2.0),
            'exit_confirm_bars': params.get('exit_confirm_bars', 2),
            'exit_confirm_mult': params.get('exit_confirm_mult', 1.0),
        }
        return self._run_backtest(mapped_params, self.commission, self.slippage)
    
    def _run_backtest(self, params: Dict[str, Any], commission: float = 0.0, slippage: float = 0.0) -> Dict[str, float]:
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
        
        # Sharpe hesabı için getirileri tut
        trade_returns = []
        
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
                    trade_returns.append(pnl)
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
                    trade_returns.append(pnl)
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
        
        # Maliyetleri düş
        cost_per_trade = commission + slippage
        net_profit = gross_profit - gross_loss - (trades * cost_per_trade)
        pf = (gross_profit / (gross_loss + trades * cost_per_trade)) if (gross_loss + trades * cost_per_trade) > 0 else 999
        
        # Fitness hesapla
        from src.optimization.fitness import quick_fitness
        
        sharpe = 0.0
        if len(trade_returns) > 1:
            sharpe = calculate_sharpe(np.array(trade_returns))
            
        fit = quick_fitness(net_profit + (trades * cost_per_trade), pf, max_dd, trades, 
                           sharpe=sharpe,
                           commission=commission, slippage=slippage)
        
        return {
            'net_profit': net_profit,
            'trades': trades,
            'pf': pf,
            'max_dd': max_dd,
            'fitness': fit,
            'win_count': 0 # TODO: Gerçek win_count
        }


# ==============================================================================
# BAYESIAN OPTIMIZER
# ==============================================================================
class BayesianOptimizer:
    """Bayesian Optimization motoru (Optuna ile) - Her iki strateji için"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        n_trials: int = 100,
        fitness_config: Optional[FitnessConfig] = None,
        strategy_index: int = 1,
        seed: int = 42,
        n_parallel: int = 4,
        commission: float = 0.0,
        slippage: float = 0.0,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        narrowed_ranges: dict = None
    ):
        """
        Args:
            df: Veri DataFrame'i
            n_trials: Optuna deneme sayısı
            fitness_config: Fitness konfigürasyonu
            strategy_index: 0 = Strateji 1 (Gatekeeper), 1 = Strateji 2 (ARS Trend v2)
            seed: Rastgele seed
            n_parallel: Paralel işlem sayısı
            narrowed_ranges: Cascade modu için dar parametre aralıkları
        """
        self.df = df
        self.n_trials = n_trials
        self.fitness_config = fitness_config or FitnessConfig()
        self.strategy_index = strategy_index
        self.seed = seed
        self.n_parallel = n_parallel
        self.commission = commission
        self.slippage = slippage
        self.is_cancelled_callback = is_cancelled_callback
        
        self.cache = IndicatorCache(df)
        self.objective = BayesianObjective(
            self.cache, self.fitness_config, strategy_index, 
            commission, slippage, narrowed_ranges  # Cascade destegi
        )
        self.study = None
        self.on_trial_complete = None # Callback function(trial_no, max_trials, best_fitness)
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Optimizasyonu çalıştır"""
        start_time = time()
        
        if verbose:
            print("Bayesian Optimizasyon Basliyor...")
            print(f"  Deneme sayisi: {self.n_trials}")
            print(f"  Paralel: {self.n_parallel}")
        
        # Optuna study oluştur
        sampler = TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='strategy_bayesian'
        )
        
        # Optimizasyonu çalıştır
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
        
        # Callback wrapper
        def optuna_callback(study, trial):
            # İptal kontrolü
            if self.is_cancelled_callback and self.is_cancelled_callback():
                study.stop()
                return

            if self.on_trial_complete:
                self.on_trial_complete(len(study.trials), self.n_trials, study.best_value)

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=verbose,
            n_jobs=self.n_parallel,
            callbacks=[optuna_callback]
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
                print(f"  Islem: {self.objective.best_result['trades']}")
                print(f"  MaxDD: {self.objective.best_result['max_dd']:,.0f}")
            print(f"\nEn Iyi Parametreler:")
            if self.objective.best_params:
                for k, v in self.objective.best_params.items():
                    print(f"  {k}: {v}")
        
        return result


# ==============================================================================
# MAIN
# ==============================================================================
def run_bayesian_optimization(n_trials: int = 100) -> Dict[str, Any]:
    """Ana fonksiyon"""
    print("Veri yukleniyor...")
    df = load_data()
    print(f"Veri hazir: {len(df)} bar")
    
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
        print("\nSonuc kaydedildi: results/bayesian_optimizer_result.csv")
    
    return result


if __name__ == "__main__":
    try:
        run_bayesian_optimization(n_trials=100)
    except KeyboardInterrupt:
        print("\nIptal edildi.")
