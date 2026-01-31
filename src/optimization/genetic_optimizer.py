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
class ParameterSpace:
    """Parametre uzayı tanımı (Planlanmış Mimari v4.1)"""
    def __init__(self):
        # Her parametre: (min, max, step, is_int)
        self.params = {
            # ARS Dinamik
            'ars_ema': (2, 10, 1, True),
            'ars_atr_p': (8, 20, 1, True),
            'ars_atr_m': (0.3, 1.2, 0.1, False),
            # Momentum & Breakout (RSI kaldırıldı, 3 farklı breakout periyodu)
            'momentum_p': (3, 15, 1, True),
            'breakout_p1': (8, 20, 2, True),    # Kısa vadeli
            'breakout_p2': (20, 40, 5, True),   # Orta vadeli
            'breakout_p3': (40, 80, 10, True),  # Uzun vadeli
            # MFI & Volume
            'mfi_p': (10, 20, 2, True),
            'mfi_hhv_p': (10, 20, 2, True),
            'vol_p': (10, 20, 2, True),
            # ATR bazlı çıkış
            'atr_exit_p': (10, 20, 2, True),
            'atr_sl_mult': (1.0, 2.5, 0.5, False),
            'atr_tp_mult': (2.0, 5.0, 0.5, False),
            'atr_trail_mult': (1.0, 2.0, 0.5, False),
            # Çift teyitli çıkış (YENİ)
            'exit_confirm_bars': (1, 3, 1, True),
            'exit_confirm_mult': (0.5, 1.5, 0.5, False),
        }
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
# FITNESS FUNCTION (Backtest wrapper)
# ==============================================================================
class FitnessEvaluator:
    """Fitness değerlendirici - Backtest fonksiyonunu sarar"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.opens = df['Acilis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.closes = df['Kapanis'].values
        self.typical = df['Tipik'].values
        self.volumes = df['Lot'].values
        self.n = len(self.closes)
        
        # Cache for indicators
        self._indicator_cache = {}
    
    def _get_cached(self, key, calc_fn):
        if key not in self._indicator_cache:
            self._indicator_cache[key] = calc_fn()
        return self._indicator_cache[key]
    
    def evaluate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Birey fitness'ını hesapla (Planlanmış Mimari v4.1)"""
        try:
            # Parametreleri çıkar
            ars_ema = int(params['ars_ema'])
            ars_atr_p = int(params['ars_atr_p'])
            ars_atr_m = float(params['ars_atr_m'])
            mom_p = int(params['momentum_p'])
            brk_p1 = int(params['breakout_p1'])
            brk_p2 = int(params['breakout_p2'])
            brk_p3 = int(params['breakout_p3'])
            mfi_p = int(params['mfi_p'])
            mfi_hhv_p = int(params['mfi_hhv_p'])
            vol_p = int(params['vol_p'])
            atr_exit_p = int(params['atr_exit_p'])
            atr_sl_mult = float(params['atr_sl_mult'])
            atr_tp_mult = float(params['atr_tp_mult'])
            atr_trail_mult = float(params['atr_trail_mult'])
            exit_confirm_bars = int(params['exit_confirm_bars'])
            exit_confirm_mult = float(params['exit_confirm_mult'])
            
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
            
            # 3 farklı periyot için HHV/LLV
            hhv1 = self._get_cached(f'hhv_{brk_p1}', lambda: np.array(HHV(self.highs.tolist(), brk_p1)))
            llv1 = self._get_cached(f'llv_{brk_p1}', lambda: np.array(LLV(self.lows.tolist(), brk_p1)))
            hhv2 = self._get_cached(f'hhv_{brk_p2}', lambda: np.array(HHV(self.highs.tolist(), brk_p2)))
            llv2 = self._get_cached(f'llv_{brk_p2}', lambda: np.array(LLV(self.lows.tolist(), brk_p2)))
            hhv3 = self._get_cached(f'hhv_{brk_p3}', lambda: np.array(HHV(self.highs.tolist(), brk_p3)))
            llv3 = self._get_cached(f'llv_{brk_p3}', lambda: np.array(LLV(self.lows.tolist(), brk_p3)))
            
            mfi = self._get_cached(f'mfi_{mfi_p}', lambda: np.array(MoneyFlowIndex(
                self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), self.volumes.tolist(), mfi_p
            )))
            mfi_hhv = self._get_cached(f'mfi_hhv_{mfi_p}_{mfi_hhv_p}', lambda: np.array(HHV(mfi.tolist(), mfi_hhv_p)))
            mfi_llv = self._get_cached(f'mfi_llv_{mfi_p}_{mfi_hhv_p}', lambda: np.array(LLV(mfi.tolist(), mfi_hhv_p)))
            vol_hhv = self._get_cached(f'vol_hhv_{vol_p}', lambda: np.array(HHV(self.volumes.tolist(), vol_p)))
            
            # Backtest
            result = self._run_backtest(
                ars, atr, dinamikK, mom, 
                hhv1, llv1, hhv2, llv2, hhv3, llv3,
                mfi, mfi_hhv, mfi_llv, vol_hhv,
                atr_sl_mult, atr_tp_mult, atr_trail_mult,
                exit_confirm_bars, exit_confirm_mult,
                brk_p2, brk_p3
            )
            
            return result
            
        except Exception as e:
            return {'net_profit': -999999, 'trades': 0, 'pf': 0, 'max_dd': 999999, 'fitness': -999999}
    
    def _run_backtest(self, ars, atr, dinamikK, mom, 
                      hhv1, llv1, hhv2, llv2, hhv3, llv3,
                      mfi, mfi_hhv, mfi_llv, vol_hhv,
                      atr_sl, atr_tp, atr_trail,
                      exit_confirm_bars, exit_confirm_mult,
                      brk_p2, brk_p3) -> Dict[str, float]:
        """Hızlı backtest (Planlanmış Mimari v4.1)"""
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
        warmup = max(brk_p3, 60)
        
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
            
            # ========== ENTRY LOGIC ==========
            if pos == 0:
                if current_trend == 1:
                    # Fiyat breakout (3 periyottan en az birinde)
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
    """Genetik Algoritma Optimizasyon Motoru"""
    
    def __init__(self, df: pd.DataFrame, config: Optional[GeneticConfig] = None):
        self.df = df
        self.config = config or GeneticConfig()
        self.param_space = ParameterSpace()
        self.evaluator = FitnessEvaluator(df)
        
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
