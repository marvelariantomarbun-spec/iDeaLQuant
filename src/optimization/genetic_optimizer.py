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
            raise ValueError(f"Geçersiz strategy_index: {strategy_index}. Sadece 0 (Gatekeeper) ve 1 (ARS Trend v2) desteklenir.")
            
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
                raise ValueError(f"Geçersiz strategy_index: {self.strategy_index}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: Genetic Eval Failed: {str(e)}")
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'fitness': -999999}


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
        
        # Trading days calculation
        trading_days = 252.0
        if self.dates and len(self.dates) > 1:
            try:
                trading_days = (self.dates[-1] - self.dates[0]).days
            except:
                pass
        
        # Backtest
        np_val, trades, pf, dd, sharpe = fast_backtest(self.closes, signals, exits_long, exits_short, self.commission, self.slippage, trading_days=trading_days)
        
        # Fitness hesapla - Maliyetler np_val icinde dusuruldu, commission/slippage=0.0 gonderilmeli
        fitness = quick_fitness(
            np_val, pf, dd, trades, sharpe=sharpe,
            initial_capital=10000.0,
            commission=0.0,
            slippage=0.0
        )
        
        return {
            'net_profit': np_val,
            'trades': trades,
            'pf': pf,
            'max_dd': dd,
            'sharpe': sharpe,
            'fitness': fitness
        }
    
    def _evaluate_strategy2(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 2 (ARS Trend v2) için fitness hesapla"""
        try:
            from src.strategies.ars_trend_v2 import ARSTrendStrategyV2
            from src.optimization.hybrid_group_optimizer import fast_backtest
            from src.optimization.fitness import quick_fitness
            
            # Simple wrapper for cache
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
            
            # Gercek strateji sinifini kullan (Seans saati, tatil filtreleri vb. icin)
            strategy = ARSTrendStrategyV2.from_config_dict(cache, params)
            signals, exits_long, exits_short = strategy.generate_all_signals()
            
            # Trading days calculation
            trading_days = 252.0
            if self.dates and len(self.dates) > 1:
                try:
                    trading_days = (self.dates[-1] - self.dates[0]).days
                except: pass
            
            # Backtest (Hibrit ile ayni fonksiyonu kullan)
            np_val, trades, pf, dd, sharpe = fast_backtest(
                self.closes, signals, exits_long, exits_short, 
                self.commission, self.slippage, trading_days=trading_days
            )
            
            # Fitness - Maliyetler np_val icinde, burada 0.0 gonderilmeli
            fitness = quick_fitness(
                np_val, pf, dd, trades, sharpe=sharpe,
                commission=0.0, slippage=0.0
            )
            
            return {
                'net_profit': np_val,
                'trades': trades,
                'pf': pf,
                'max_dd': dd,
                'sharpe': sharpe,
                'fitness': fitness
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'fitness': -999999}



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
            pool = Pool(processes=self.n_parallel, initializer=_init_pool, initargs=(self.df, self.strategy_index, self.commission, self.slippage))
            
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
