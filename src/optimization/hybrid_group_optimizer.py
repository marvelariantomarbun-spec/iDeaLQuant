# -*- coding: utf-8 -*-
"""
Hybrid Group Optimizer v1.0
===========================
Hibrit yaklaşım: Önce grupları bağımsız optimize et, sonra kombine et.

Strateji 1 (Gatekeeper) Grupları:
- Grup 1: ARS (Bağımsız)
- Grup 2: ADX (Bağımsız)
- Grup 3: MACD-V (Bağımsız)
- Grup 4: BB (Bağımsız)
- Grup 5: Hacim (Kademeli)
- Grup 6: Skor (Kademeli)

Akış: Grup 1-4 bağımsız → Kombinasyon → Grup 5-6 kademeli
"""

import sys
import os
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, ATR, ADX, SMA, ARS, NetLot, MACDV
from src.strategies.score_based import ScoreBasedStrategy
from src.strategies.ars_trend_v2 import ARSTrendStrategyV2

# Opsiyonel: Veritabanı entegrasyonu
try:
    from src.core.database import db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    db = None

# ==============================================================================
# GROUP DEFINITIONS
# ==============================================================================
@dataclass
class ParameterGroup:
    """Parametre grubu tanımı"""
    name: str
    params: Dict[str, List[Any]]  # param_name -> [values]
    is_independent: bool = True   # Bağımsız mı, kademeli mi
    default_values: Dict[str, Any] = field(default_factory=dict)  # Diğer gruplar için varsayılanlar

# Strateji 1 Grup Tanımları (20 Parametre)
STRATEGY1_GROUPS = [
    ParameterGroup(
        name="ARS",
        params={
            'ars_period': [3, 4, 5, 8, 10, 12],
            'ars_k': [0.005, 0.008, 0.01, 0.012, 0.015, 0.02],
        },
        is_independent=True,
        default_values={'ars_period': 3, 'ars_k': 0.01}
    ),
    ParameterGroup(
        name="ADX",
        params={
            'adx_period': [14, 17, 21, 25, 30],
            'adx_threshold': [20.0, 25.0, 30.0],
        },
        is_independent=True,
        default_values={'adx_period': 17, 'adx_threshold': 25.0}
    ),
    ParameterGroup(
        name="MACDV",
        params={
            'macdv_short': [10, 13, 15],
            'macdv_long': [24, 28, 32],
            'macdv_signal': [7, 8, 9],
            'macdv_threshold': [0.0, 0.1, 0.2],
        },
        is_independent=True,
        default_values={'macdv_short': 13, 'macdv_long': 28, 'macdv_signal': 8, 'macdv_threshold': 0.0}
    ),
    ParameterGroup(
        name="NetLot",
        params={
            'netlot_period': [3, 5, 8],
            'netlot_threshold': [10, 20, 30, 40],
        },
        is_independent=True,
        default_values={'netlot_period': 5, 'netlot_threshold': 20.0}
    ),
    ParameterGroup(
        name="Yatay_Filtre",
        params={
            'ars_mesafe_threshold': [0.20, 0.25, 0.30],
            'bb_period': [15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5],
            'bb_width_multiplier': [0.6, 0.8, 1.0],
            'bb_avg_period': [30, 50, 70],
            'yatay_ars_bars': [5, 10, 15],
            'yatay_adx_threshold': [15.0, 20.0, 25.0],
            'filter_score_threshold': [1, 2, 3],
        },
        is_independent=True,
        default_values={
            'ars_mesafe_threshold': 0.25, 'bb_period': 20, 'bb_std': 2.0,
            'bb_width_multiplier': 0.8, 'bb_avg_period': 50, 'yatay_ars_bars': 10,
            'yatay_adx_threshold': 20.0, 'filter_score_threshold': 2
        }
    ),
    ParameterGroup(
        name="Skor_Ayarlari",
        params={
            'min_score': [2, 3, 4],
            'exit_score': [2, 3, 4],
        },
        is_independent=False, # Kademeli
        default_values={'min_score': 3, 'exit_score': 3}
    ),
]

# Strateji 2 Grup Tanımları (19 Optimize Edilebilir Parametre)
STRATEGY2_GROUPS = [
    ParameterGroup(
        name="ARS",
        params={
            'ars_ema_period': [2, 3, 5, 8, 10, 12],
            'ars_atr_period': [7, 10, 14, 17, 20],
            'ars_atr_mult': [0.3, 0.5, 0.7, 1.0, 1.2, 1.5],
            'ars_min_band': [0.001, 0.002, 0.003, 0.005],
            'ars_max_band': [0.010, 0.015, 0.020, 0.025],
        },
        is_independent=True,
        default_values={
            'ars_ema_period': 3, 'ars_atr_period': 10, 'ars_atr_mult': 0.5,
            'ars_min_band': 0.002, 'ars_max_band': 0.015
        }
    ),
    ParameterGroup(
        name="Giris_Filtreleri",
        params={
            'momentum_period': [3, 5, 7, 10],
            'momentum_threshold': [50.0, 100.0, 150.0, 200.0],
            'breakout_period': [5, 10, 15, 20, 30],
            'mfi_period': [10, 14, 17, 21],
            'mfi_hhv_period': [10, 14, 17, 21],
            'mfi_llv_period': [10, 14, 17, 21],
            'volume_hhv_period': [10, 14, 17, 21],
        },
        is_independent=True,
        default_values={
            'momentum_period': 5, 'momentum_threshold': 100.0, 'breakout_period': 10,
            'mfi_period': 14, 'mfi_hhv_period': 14, 'mfi_llv_period': 14, 'volume_hhv_period': 14
        }
    ),
    ParameterGroup(
        name="Cikis_Risk",
        params={
            'atr_exit_period': [10, 14, 17, 21],
            'atr_sl_mult': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'atr_tp_mult': [3.0, 4.0, 5.0, 6.0, 8.0],
            'atr_trail_mult': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'exit_confirm_bars': [1, 2, 3, 4, 5],
            'exit_confirm_mult': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        },
        is_independent=False, # Kademeli - Giriş filtrelerine bağlı
        default_values={
            'atr_exit_period': 14, 'atr_sl_mult': 2.0, 'atr_tp_mult': 5.0,
            'atr_trail_mult': 2.0, 'exit_confirm_bars': 2, 'exit_confirm_mult': 1.0
        }
    ),
    ParameterGroup(
        name="Ince_Ayar",
        params={
            'volume_mult': [0.5, 0.6, 0.8, 1.0, 1.2, 1.5],
            'volume_llv_period': [10, 14, 17, 21],
        },
        is_independent=False, # Kademeli
        default_values={'volume_mult': 0.8, 'volume_llv_period': 14}
    ),
]

# ==============================================================================
# DATA & CACHE
# ==============================================================================
g_cache = None

def load_data() -> pd.DataFrame:
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=0, low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    for c in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
    df.dropna(inplace=True)
    # Tarih ve saat kolonlarından datetime oluştur (format: 25.12.2024 17:33:00)
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    # NaT değerleri olan satırları sil
    df = df.dropna(subset=['DateTime']).reset_index(drop=True)
    return df

class IndicatorCache:
    def __init__(self, df):
        self.df = df
        self.opens = df['Acilis'].values
        self.closes = df['Kapanis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.typical = df['Tipik'].values
        self.lots = df['Lot'].values
        self.volumes = df['Lot'].values  # Alias for strategy compatibility
        self.n = len(self.closes)
        
        # Tarih bilgisi (vade/tatil yönetimi için)
        if 'DateTime' in df.columns:
            self.dates = df['DateTime'].tolist()  # For ScoreBasedStrategy
            self.times = df['DateTime'].tolist()  # For ARSTrendStrategyV2
        else:
            self.dates = None
            self.times = None
        
        # Pre-compute some static indicators
        sma20 = pd.Series(self.closes).rolling(20).mean()
        std20 = pd.Series(self.closes).rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        self.bb_width = ((upper - lower) / sma20 * 100).fillna(0).values
        self.bb_width_avg = pd.Series(self.bb_width).rolling(50).mean().values
        
        self._cache = {}
    
    def get_ars(self, p, k):
        key = f'ars_{p}_{k}'
        if key not in self._cache:
            self._cache[key] = np.array(ARS(self.typical.tolist(), int(p), float(k)))
        return self._cache[key]
    
    def get_adx(self, p):
        key = f'adx_{p}'
        if key not in self._cache:
            self._cache[key] = np.array(ADX(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(p)))
        return self._cache[key]
    
    def get_macdv(self, s, l, sig):
        key = f'macdv_{s}_{l}_{sig}'
        if key not in self._cache:
            macd_line, signal_line = MACDV(self.closes.tolist(), self.highs.tolist(), self.lows.tolist(), int(s), int(l), int(sig))
            self._cache[key] = (np.array(macd_line), np.array(signal_line))
        return self._cache[key]
    
    def get_netlot(self, ma_period):
        key = f'netlot_{ma_period}'
        if key not in self._cache:
            netlot = NetLot(self.lots.tolist(), int(ma_period))
            self._cache[key] = np.array(netlot)
        return self._cache[key]

def worker_init():
    global g_cache
    df = load_data()
    g_cache = IndicatorCache(df)


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================
def fast_backtest(closes, signals, exits_long, exits_short) -> Tuple[float, int, float, float]:
    """Hızlı backtest - sinyallere göre işlem simülasyonu"""
    pos = 0
    entry_price = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    for i in range(50, len(closes)):
        # Çıkış kontrolü
        if pos == 1 and exits_long[i]:
            pnl = closes[i] - entry_price
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
            current_equity += pnl
            pos = 0
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = peak_equity - current_equity
            if dd > max_dd:
                max_dd = dd
                
        elif pos == -1 and exits_short[i]:
            pnl = entry_price - closes[i]
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
            current_equity += pnl
            pos = 0
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = peak_equity - current_equity
            if dd > max_dd:
                max_dd = dd
        
        # Giriş kontrolü
        if pos == 0:
            if signals[i] == 1:
                pos = 1
                entry_price = closes[i]
                trades += 1
            elif signals[i] == -1:
                pos = -1
                entry_price = closes[i]
                trades += 1
    
    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    return net_profit, trades, pf, max_dd


# ==============================================================================
# GROUP OPTIMIZER
# ==============================================================================
class HybridGroupOptimizer:
    """Hibrit Grup Optimizasyonu"""
    
    def __init__(self, groups: List[ParameterGroup], 
                 process_id: str = None, strategy_index: int = 0):
        self.groups = groups
        self.independent_groups = [g for g in groups if g.is_independent]
        self.cascaded_groups = [g for g in groups if not g.is_independent]
        
        # Veritabanı entegrasyonu
        self.process_id = process_id
        self.strategy_index = strategy_index
        
        self.group_results: Dict[str, List[Dict]] = {}
        self.combined_results: List[Dict] = []
        self.final_results: List[Dict] = []
        
    def get_default_params(self, exclude_group: str = None) -> Dict[str, Any]:
        """Tüm grupların varsayılan değerlerini al (belirli bir grup hariç)"""
        defaults = {}
        for g in self.groups:
            if g.name != exclude_group:
                defaults.update(g.default_values)
        return defaults
    
    def generate_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Parametre kombinasyonlarını üret"""
        keys = list(params.keys())
        values = list(params.values())
        return [dict(zip(keys, v)) for v in product(*values)]
    
    def run_group_optimization(self, group: ParameterGroup, fixed_params: Dict[str, Any] = None) -> List[Dict]:
        """Tek bir grubu optimize et"""
        print(f"\n=== Grup: {group.name} ===")
        
        # Varsayılan parametreler + sabit parametreler
        base_params = self.get_default_params(exclude_group=group.name)
        if fixed_params:
            base_params.update(fixed_params)
        
        # Bu grubun kombinasyonları
        combos = self.generate_combinations(group.params)
        print(f"Kombinasyon sayısı: {len(combos)}")
        
        results = []
        
        global g_cache
        if g_cache is None:
            df = load_data()
            g_cache = IndicatorCache(df)
        
        for combo in combos:
            # Tüm parametreleri birleştir
            all_params = {**base_params, **combo}
            
            # Backtest yap
            score = self._evaluate_params(all_params)
            if score['net_profit'] > 0:
                results.append({
                    'group': group.name,
                    **combo,
                    **score
                })
        
        # En iyi 10'u tut
        results.sort(key=lambda x: x['net_profit'], reverse=True)
        top_results = results[:10]
        
        print(f"Bulunan: {len(results)}, Top: {len(top_results)}")
        if top_results:
            print(f"En iyi: NP={top_results[0]['net_profit']:.0f}, PF={top_results[0]['pf']:.2f}")
        
        return top_results
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Parametre setini değerlendir (ScoreBased v4.1)
        Strateji sınıfı üzerinden sinyal üretimi - Vade/Tatil yönetimi dahil.
        """
        global g_cache
        
        # Strateji sınıfını kullanarak sinyal üret
        strategy = ScoreBasedStrategy.from_config_dict(g_cache, params)
        signals, exits_long, exits_short = strategy.generate_all_signals()
        
        # Backtest
        np_val, trades, pf, dd = fast_backtest(g_cache.closes, signals, exits_long, exits_short)
        
        return {
            'net_profit': np_val,
            'trades': trades,
            'pf': pf,
            'max_dd': dd
        }

    
    def run_independent_phase(self):
        """Bağımsız grupları optimize et"""
        print("\n" + "="*60)
        print("PHASE 1: BAĞIMSIZ GRUP OPTİMİZASYONU")
        print("="*60)
        
        for group in self.independent_groups:
            results = self.run_group_optimization(group)
            self.group_results[group.name] = results
            
            # Veritabanına kaydet
            if self.process_id and DB_AVAILABLE and db:
                db.save_group_results_batch(
                    process_id=self.process_id,
                    strategy_index=self.strategy_index,
                    group_name=group.name,
                    results=results
                )
                print(f"  -> {group.name} sonuclari DB'ye kaydedildi")
    
    def run_combination_phase(self):
        """Bağımsız grupların en iyilerini kombine et"""
        print("\n" + "="*60)
        print("PHASE 2: KOMBİNASYON")
        print("="*60)
        
        # Her gruptan top 3'ü al
        top_per_group = {}
        for name, results in self.group_results.items():
            top_per_group[name] = results[:3] if len(results) >= 3 else results
        
        # Kombinasyonları oluştur
        group_names = list(top_per_group.keys())
        if not group_names:
            print("Kombinasyon için yeterli sonuç yok.")
            return
        
        # Kartezyen çarpım
        all_combos = list(product(*[top_per_group[n] for n in group_names]))
        print(f"Kombine edilecek: {len(all_combos)} kombinasyon")
        
        global g_cache
        if g_cache is None:
            df = load_data()
            g_cache = IndicatorCache(df)
        
        for combo in all_combos:
            # Tüm parametreleri birleştir
            merged_params = {}
            for group_result in combo:
                for k, v in group_result.items():
                    if k not in ['group', 'net_profit', 'trades', 'pf', 'max_dd']:
                        merged_params[k] = v
            
            # Kademeli grupların varsayılanlarını ekle
            for g in self.cascaded_groups:
                merged_params.update(g.default_values)
            
            # Değerlendir
            score = self._evaluate_params(merged_params)
            if score['net_profit'] > 0:
                self.combined_results.append({
                    **merged_params,
                    **score
                })
        
        self.combined_results.sort(key=lambda x: x['net_profit'], reverse=True)
        print(f"Bulunan: {len(self.combined_results)}")
        if self.combined_results:
            print(f"En iyi: NP={self.combined_results[0]['net_profit']:.0f}")
    
    def run_cascaded_phase(self):
        """Kademeli grupları optimize et (giriş sabit, çıkış optimize)"""
        print("\n" + "="*60)
        print("PHASE 3: KADEMELİ GRUP OPTİMİZASYONU")
        print("="*60)
        
        if not self.combined_results:
            print("Kombinasyon sonucu yok, atlanıyor.")
            return
        
        # En iyi kombinasyonu sabit tut
        best_base = self.combined_results[0].copy()
        # Metrik alanlarını çıkar
        for k in ['net_profit', 'trades', 'pf', 'max_dd']:
            best_base.pop(k, None)
        
        print(f"Sabit Base: {best_base}")
        
        for group in self.cascaded_groups:
            results = self.run_group_optimization(group, fixed_params=best_base)
            
            # Veritabanına kaydet
            if self.process_id and DB_AVAILABLE and db:
                db.save_group_results_batch(
                    process_id=self.process_id,
                    strategy_index=self.strategy_index,
                    group_name=group.name,
                    results=results
                )
                print(f"  -> {group.name} sonuclari DB'ye kaydedildi")
            
            # En iyi sonucu base'e ekle
            if results:
                for k, v in results[0].items():
                    if k not in ['group', 'net_profit', 'trades', 'pf', 'max_dd']:
                        best_base[k] = v
        
        # Final değerlendirme
        final_score = self._evaluate_params(best_base)
        self.final_results = [{**best_base, **final_score}]
        
        print(f"\nFINAL: NP={final_score['net_profit']:.0f}, PF={final_score['pf']:.2f}, DD={final_score['max_dd']:.0f}")
    
    def run_stability_phase(self, best_params: Dict[str, Any]):
        """PHASE 4: STABILITE ANALIZI (Stage 3)
           En iyi parametrenin komşularını tarayarak 'Stability Score' hesaplar.
        """
        print("\n" + "="*60)
        print("PHASE 4: STABILITE ANALIZI")
        print("="*60)
        
        # Sayısal parametreler ve değişim miktarları
        numeric_params = {
            'ars_period': [1], 'ars_k': [0.001],
            'adx_period': [1], 'adx_threshold': [1.0],
            'macdv_short': [1], 'macdv_long': [1], 'macdv_signal': [1],
            'bb_period': [1], 'bb_std': [0.1],
            'netlot_period': [1], 'netlot_threshold': [1.0],
            'ars_ema_period': [1], 'ars_atr_period': [1], 'ars_atr_mult': [0.1],
            'momentum_period': [1], 'breakout_period': [1],
            'atr_exit_period': [1], 'atr_sl_mult': [0.1], 'atr_tp_mult': [0.1], 'atr_trail_mult': [0.1]
        }
        
        current_score = best_params['net_profit']
        neighbor_scores = []
        
        # Sadece best_params içinde olan sayısal parametreleri al
        to_test = {k: v for k, v in numeric_params.items() if k in best_params}
        
        print(f"Test edilecek parametre sayısı: {len(to_test)}")
        
        for p_name, steps in to_test.items():
            for step in steps:
                for direction in [-1, 1]:
                    test_params = best_params.copy()
                    # Metrik temizle
                    for k in ['net_profit', 'trades', 'pf', 'max_dd']: test_params.pop(k, None)
                    
                    original_val = test_params[p_name]
                    test_params[p_name] = original_val + (step * direction)
                    
                    # Sayısal sınırları koru
                    if isinstance(original_val, int): test_params[p_name] = max(1, int(test_params[p_name]))
                    else: test_params[p_name] = max(0.001, test_params[p_name])
                    
                    res = self._evaluate_params(test_params)
                    neighbor_scores.append(res['net_profit'])
        
        stability = (sum(neighbor_scores) / len(neighbor_scores)) / current_score if neighbor_scores else 0
        print(f"Stabilite Skoru: {stability:.2f} (1.00 = Mükemmel, < 0.70 = Riskli)")
        return stability

    def run(self):
        """Tam optimizasyon akışı"""
        start_time = time()
        
        self.run_independent_phase()
        self.run_combination_phase()
        self.run_cascaded_phase()
        
        if self.final_results:
            stability_score = self.run_stability_phase(self.final_results[0])
            self.final_results[0]['stability_score'] = stability_score
            
        elapsed = time() - start_time
        print(f"\nToplam süre: {elapsed:.1f}sn")
        
        return self.final_results


# ==============================================================================
# MAIN
# ==============================================================================
def run_strategy1_hybrid():
    """Strateji 1 için Hibrit Grup Optimizasyonu"""
    print("="*60)
    print("STRATEJI 1 - HİBRİT GRUP OPTİMİZASYONU")
    print("="*60)
    
    optimizer = HybridGroupOptimizer(STRATEGY1_GROUPS)
    results = optimizer.run()
    
    if results:
        # Kaydet
        df = pd.DataFrame(results)
        os.makedirs("d:/Projects/IdealQuant/results", exist_ok=True)
        df.to_csv("d:/Projects/IdealQuant/results/strategy1_hybrid_results.csv", index=False)
        print("\nSonuç kaydedildi: results/strategy1_hybrid_results.csv")
    
    return results


if __name__ == "__main__":
    try:
        run_strategy1_hybrid()
    except KeyboardInterrupt:
        print("\nİptal edildi.")



# ==============================================================================
# STRATEGY 2 HYBRID OPTIMIZER (Uses STRATEGY2_GROUPS defined at top of file)
# ==============================================================================

class Strategy2HybridOptimizer(HybridGroupOptimizer):
    """Strateji 2 için özelleştirilmiş hibrit optimizer"""
    
    def __init__(self):
        super().__init__(STRATEGY2_GROUPS)
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 2 parametrelerini değerlendir (Planlanmış Mimari v4.1 - 21 Parametre)"""
        global g_cache
        
        from src.indicators.core import ARS_Dynamic, Momentum, HHV, LLV, MoneyFlowIndex, ATR, EMA
        
        closes = g_cache.closes
        highs = g_cache.highs
        lows = g_cache.lows
        typical = g_cache.typical
        lots = g_cache.lots
        n = len(closes)
        
        # 1. ARS Dinamik
        ars_ema = int(params.get('ars_ema_period', 3))
        ars_atr_p = int(params.get('ars_atr_period', 10))
        ars_atr_m = float(params.get('ars_atr_mult', 0.5))
        ars_min = float(params.get('ars_min_band', 0.002))
        ars_max = float(params.get('ars_max_band', 0.015))
        
        ars_key = f"ars_dyn_{ars_ema}_{ars_atr_p}_{ars_atr_m}"
        if ars_key not in g_cache._cache:
            g_cache._cache[ars_key] = np.array(ARS_Dynamic(
                typical.tolist(), highs.tolist(), lows.tolist(), closes.tolist(),
                ema_period=ars_ema, atr_period=ars_atr_p, atr_mult=ars_atr_m,
                min_k=ars_min, max_k=ars_max
            ))
        ars = g_cache._cache[ars_key]
        
        # 2. ATR_Exit ve dinamikK
        atr_exit_p = int(params.get('atr_exit_period', 14))
        atr_key = f'atr_{atr_exit_p}'
        if atr_key not in g_cache._cache:
            g_cache._cache[atr_key] = np.array(ATR(highs.tolist(), lows.tolist(), closes.tolist(), atr_exit_p))
        atr_exit = g_cache._cache[atr_key]
        
        # 3. Momentum ve Breakout
        mom_p = int(params.get('momentum_period', 5))
        mom_th = float(params.get('momentum_threshold', 100.0))
        mom_key = f'mom_{mom_p}'
        if mom_key not in g_cache._cache:
            g_cache._cache[mom_key] = np.array(Momentum(closes.tolist(), mom_p))
        mom = g_cache._cache[mom_key]
        
        brk_p = int(params.get('breakout_period', 10))
        hhv_key, llv_key = f'hhv_{brk_p}', f'llv_{brk_p}'
        if hhv_key not in g_cache._cache:
            g_cache._cache[hhv_key] = np.array(HHV(highs.tolist(), brk_p))
            g_cache._cache[llv_key] = np.array(LLV(lows.tolist(), brk_p))
        hhv, llv = g_cache._cache[hhv_key], g_cache._cache[llv_key]
        
        # 4. MFI Breakout
        mfi_p = int(params.get('mfi_period', 14))
        mfi_key = f'mfi_{mfi_p}'
        if mfi_key not in g_cache._cache:
            g_cache._cache[mfi_key] = np.array(MoneyFlowIndex(highs.tolist(), lows.tolist(), closes.tolist(), lots.tolist(), mfi_p))
        mfi = g_cache._cache[mfi_key]
        
        mfi_hhv_p = int(params.get('mfi_hhv_period', 14))
        mfi_llv_p = int(params.get('mfi_llv_period', 14))
        mfi_hhv_key = f'mfi_hhv_{mfi_p}_{mfi_hhv_p}'
        mfi_llv_key = f'mfi_llv_{mfi_p}_{mfi_llv_p}'
        if mfi_hhv_key not in g_cache._cache:
            g_cache._cache[mfi_hhv_key] = np.array(HHV(mfi.tolist(), mfi_hhv_p))
        if mfi_llv_key not in g_cache._cache:
            g_cache._cache[mfi_llv_key] = np.array(LLV(mfi.tolist(), mfi_llv_p))
        mfi_hhv, mfi_llv = g_cache._cache[mfi_hhv_key], g_cache._cache[mfi_llv_key]
        
        # 5. Volume Breakout
        vol_hhv_p = int(params.get('volume_hhv_period', 14))
        vol_llv_p = int(params.get('volume_llv_period', 14))
        vol_mult = float(params.get('volume_mult', 0.8))
        vol_hhv_key = f'vol_hhv_{vol_hhv_p}'
        if vol_hhv_key not in g_cache._cache:
            g_cache._cache[vol_hhv_key] = np.array(HHV(lots.tolist(), vol_hhv_p))
        vol_hhv = g_cache._cache[vol_hhv_key]
        
        # 6. Trade Logic
        pos = 0
        entry_price = 0.0
        extreme_price = 0.0
        
        gross_profit = 0.0
        gross_loss = 0.0
        trades = 0
        max_dd = 0.0
        peak_equity = 0.0
        current_equity = 0.0
        
        bars_against_trend = 0
        
        # ATR Exit Params
        atr_sl_m = float(params.get('atr_sl_mult', 2.0))
        atr_tp_m = float(params.get('atr_tp_mult', 5.0))
        atr_trail_m = float(params.get('atr_trail_mult', 2.0))
        ex_conf_bars = int(params.get('exit_confirm_bars', 2))
        ex_conf_mult = float(params.get('exit_confirm_mult', 1.0))
        
        warmup = max(brk_p, 50)
        
        for i in range(warmup, n):
            # EXIT
            if pos == 1:
                extreme_price = max(extreme_price, highs[i])
                
                exit_signal = False
                
                # A. Double Confirmation Exit
                if closes[i] < ars[i]:
                    bars_against_trend += 1
                else:
                    bars_against_trend = 0
                
                dist_th = atr_exit[i] * ars_atr_m * ex_conf_mult
                distance_ok = (ars[i] - closes[i]) > dist_th
                
                if bars_against_trend >= ex_conf_bars and distance_ok:
                    exit_signal = True
                
                # B. ATR TP/SL/Trail
                if not exit_signal:
                    tp_price = entry_price + (atr_exit[i] * atr_tp_m)
                    sl_price = entry_price - (atr_exit[i] * atr_sl_m)
                    trail_price = extreme_price - (atr_exit[i] * atr_trail_m)
                    actual_stop = max(sl_price, trail_price)
                    
                    if closes[i] >= tp_price or closes[i] < actual_stop:
                        exit_signal = True
                
                if exit_signal:
                    pnl = closes[i] - entry_price
                    if pnl > 0: gross_profit += pnl
                    else: gross_loss += abs(pnl)
                    current_equity += pnl
                    pos = 0
                    trades += 1
                    if current_equity > peak_equity: peak_equity = current_equity
                    max_dd = max(max_dd, peak_equity - current_equity)

            elif pos == -1:
                extreme_price = min(extreme_price, lows[i])
                
                exit_signal = False
                
                # A. Double Confirmation Exit
                if closes[i] > ars[i]:
                    bars_against_trend += 1
                else:
                    bars_against_trend = 0
                
                dist_th = atr_exit[i] * ars_atr_m * ex_conf_mult
                distance_ok = (closes[i] - ars[i]) > dist_th
                
                if bars_against_trend >= ex_conf_bars and distance_ok:
                    exit_signal = True
                
                # B. ATR TP/SL/Trail
                if not exit_signal:
                    tp_price = entry_price - (atr_exit[i] * atr_tp_m)
                    sl_price = entry_price + (atr_exit[i] * atr_sl_m)
                    trail_price = extreme_price + (atr_exit[i] * atr_trail_m)
                    actual_stop = min(sl_price, trail_price)
                    
                    if closes[i] <= tp_price or closes[i] > actual_stop:
                        exit_signal = True
                
                if exit_signal:
                    pnl = entry_price - closes[i]
                    if pnl > 0: gross_profit += pnl
                    else: gross_loss += abs(pnl)
                    current_equity += pnl
                    pos = 0
                    trades += 1
                    if current_equity > peak_equity: peak_equity = current_equity
                    max_dd = max(max_dd, peak_equity - current_equity)

            # ENTRY
            if pos == 0:
                # LONG Entry
                if closes[i] > ars[i]:
                    new_high = highs[i] >= hhv[i-1] and hhv[i] > hhv[i-1]
                    mom_ok = mom[i] > mom_th
                    mfi_ok = mfi[i] >= mfi_hhv[i-1]
                    vol_ok = lots[i] >= vol_hhv[i-1] * vol_mult
                    
                    if new_high and mom_ok and mfi_ok and vol_ok:
                        pos = 1
                        entry_price = closes[i]
                        extreme_price = highs[i]
                        bars_against_trend = 0
                
                # SHORT Entry
                elif closes[i] < ars[i]:
                    new_low = lows[i] <= llv[i-1] and llv[i] < llv[i-1]
                    mom_ok = mom[i] < (200 - mom_th)
                    mfi_ok = mfi[i] <= mfi_llv[i-1]
                    vol_ok = lots[i] >= vol_hhv[i-1] * vol_mult
                    
                    if new_low and mom_ok and mfi_ok and vol_ok:
                        pos = -1
                        entry_price = closes[i]
                        extreme_price = lows[i]
                        bars_against_trend = 0

        net_profit = gross_profit - gross_loss
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
        
        return {
            'net_profit': net_profit,
            'trades': trades,
            'pf': pf,
            'max_dd': max_dd
        }


def run_strategy2_hybrid():
    """Strateji 2 için Hibrit Grup Optimizasyonu"""
    print("="*60)
    print("STRATEJI 2 - HİBRİT GRUP OPTİMİZASYONU")
    print("="*60)
    
    optimizer = Strategy2HybridOptimizer()
    results = optimizer.run()
    
    if results:
        df = pd.DataFrame(results)
        os.makedirs("d:/Projects/IdealQuant/results", exist_ok=True)
        df.to_csv("d:/Projects/IdealQuant/results/strategy2_hybrid_results.csv", index=False)
        print("\nSonuç kaydedildi: results/strategy2_hybrid_results.csv")
    
    return results

