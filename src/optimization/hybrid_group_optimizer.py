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

# Strateji 1 Grup Tanımları
STRATEGY1_GROUPS = [
    ParameterGroup(
        name="ARS",
        params={
            'ars_period': list(range(3, 18, 3)),  # 3, 6, 9, 12, 15
            'ars_k': [0.005, 0.008, 0.01, 0.012, 0.015, 0.02],
        },
        is_independent=True,
        default_values={'ars_period': 9, 'ars_k': 0.01}
    ),
    ParameterGroup(
        name="ADX",
        params={
            'adx_period': list(range(10, 50, 5)),  # 10, 15, 20, ..., 45
            'adx_trend_th': [20, 25, 30],
            'adx_strong_th': [35, 40, 45],
        },
        is_independent=True,
        default_values={'adx_period': 25, 'adx_trend_th': 25, 'adx_strong_th': 40}
    ),
    ParameterGroup(
        name="MACDV",
        params={
            'macdv_short': [8, 10, 12, 14],
            'macdv_long': [20, 24, 26, 28, 30],
            'macdv_signal': [5, 7, 9, 11],
        },
        is_independent=True,
        default_values={'macdv_short': 12, 'macdv_long': 26, 'macdv_signal': 9}
    ),
    ParameterGroup(
        name="BB",
        params={
            'bb_period': [15, 20, 25],
            'bb_stddev': [1.5, 2.0, 2.5],
            'bb_width_mult': [0.6, 0.8, 1.0],
        },
        is_independent=True,
        default_values={'bb_period': 20, 'bb_stddev': 2.0, 'bb_width_mult': 0.8}
    ),
    ParameterGroup(
        name="Hacim",
        params={
            'netlot_th': [5, 10, 15, 20, 25],
            'netlot_ma_period': [10, 20, 30],
        },
        is_independent=False,  # Kademeli
        default_values={'netlot_th': 15, 'netlot_ma_period': 20}
    ),
    ParameterGroup(
        name="Skor",
        params={
            'min_onay_skoru': [2, 3, 4],
            'cikis_hassasiyeti': [2, 3, 4],
            'yatay_min_skor': [2, 3],
            'giris_karsi_max': [1, 2],
        },
        is_independent=False,  # Kademeli
        default_values={'min_onay_skoru': 3, 'cikis_hassasiyeti': 3, 'yatay_min_skor': 2, 'giris_karsi_max': 2}
    ),
]


# ==============================================================================
# DATA & CACHE
# ==============================================================================
g_cache = None

def load_data() -> pd.DataFrame:
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    for c in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
    df.dropna(inplace=True)
    return df

class IndicatorCache:
    def __init__(self, df):
        self.df = df
        self.closes = df['Kapanis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.typical = df['Tipik'].values
        self.lots = df['Lot'].values
        self.n = len(self.closes)
        
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
    
    def __init__(self, groups: List[ParameterGroup]):
        self.groups = groups
        self.independent_groups = [g for g in groups if g.is_independent]
        self.cascaded_groups = [g for g in groups if not g.is_independent]
        
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
        """Parametre setini değerlendir"""
        global g_cache
        closes = g_cache.closes
        n = len(closes)
        
        # İndikatörleri hesapla
        ars = g_cache.get_ars(params.get('ars_period', 9), params.get('ars_k', 0.01))
        adx = g_cache.get_adx(params.get('adx_period', 25))
        macdv_val, macdv_sig = g_cache.get_macdv(
            params.get('macdv_short', 12),
            params.get('macdv_long', 26),
            params.get('macdv_signal', 9)
        )
        netlot = g_cache.get_netlot(params.get('netlot_ma_period', 20))
        
        # Yatay Filtre Skoru
        ars_diff = np.diff(ars, prepend=0) != 0
        ars_degisti = pd.Series(ars_diff).rolling(10).sum().gt(0).values.astype(int)
        ars_mesafe = np.abs(closes - ars) / np.where(ars != 0, ars, 1) * 100
        
        f1 = ars_degisti
        f2 = (ars_mesafe > 0.25).astype(int)
        f3 = (adx > params.get('adx_trend_th', 25)).astype(int)
        f4 = (g_cache.bb_width > g_cache.bb_width_avg * params.get('bb_width_mult', 0.8)).astype(int)
        yatay_filtre = (f1 + f2 + f3 + f4) >= params.get('yatay_min_skor', 2)
        
        # Sinyal Skorları
        ars_long = (closes > ars).astype(int)
        ars_short = (closes < ars).astype(int)
        macdv_long = (macdv_val > macdv_sig).astype(int)
        macdv_short = (macdv_val < macdv_sig).astype(int)
        adx_score = (adx > params.get('adx_strong_th', 40)).astype(int)
        
        nl_th = params.get('netlot_th', 15)
        nl_long = (netlot > nl_th).astype(int)
        nl_short = (netlot < -nl_th).astype(int)
        
        final_l_score = ars_long + macdv_long + nl_long + adx_score
        final_s_score = ars_short + macdv_short + nl_short + adx_score
        
        min_sc = params.get('min_onay_skoru', 3)
        karsi_max = params.get('giris_karsi_max', 2)
        
        # Sinyaller
        signals = np.zeros(n, dtype=int)
        l_cond = yatay_filtre & (final_l_score >= min_sc) & (final_s_score < karsi_max)
        s_cond = yatay_filtre & (final_s_score >= min_sc) & (final_l_score < karsi_max)
        signals[l_cond] = 1
        signals[s_cond] = -1
        
        # Çıkış Koşulları
        ex_sc = params.get('cikis_hassasiyeti', 3)
        exits_long = (closes < ars) | (final_s_score >= ex_sc)
        exits_short = (closes > ars) | (final_l_score >= ex_sc)
        
        # Backtest
        np_val, trades, pf, dd = fast_backtest(closes, signals, exits_long, exits_short)
        
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
            
            # En iyi sonucu base'e ekle
            if results:
                for k, v in results[0].items():
                    if k not in ['group', 'net_profit', 'trades', 'pf', 'max_dd']:
                        best_base[k] = v
        
        # Final değerlendirme
        final_score = self._evaluate_params(best_base)
        self.final_results = [{**best_base, **final_score}]
        
        print(f"\nFINAL: NP={final_score['net_profit']:.0f}, PF={final_score['pf']:.2f}, DD={final_score['max_dd']:.0f}")
    
    def run(self):
        """Tam optimizasyon akışı"""
        start_time = time()
        
        self.run_independent_phase()
        self.run_combination_phase()
        self.run_cascaded_phase()
        
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
# STRATEGY 2 GROUP DEFINITIONS
# ==============================================================================
STRATEGY2_GROUPS = [
    ParameterGroup(
        name="ARS_Dinamik",
        params={
            'ars_ema_period': [3, 5, 8, 10],
            'ars_atr_period': [10, 14, 20],
            'ars_atr_mult': [0.5, 0.8, 1.0, 1.2],
            'ars_min_band': [0.002, 0.003],
            'ars_max_band': [0.012, 0.015, 0.018],
        },
        is_independent=True,
        default_values={'ars_ema_period': 5, 'ars_atr_period': 14, 'ars_atr_mult': 1.0, 
                       'ars_min_band': 0.002, 'ars_max_band': 0.015}
    ),
    ParameterGroup(
        name="Breakout_Mom",
        params={
            'momentum_period': [3, 5, 8, 10],
            'breakout_period_1': [10, 14, 20],      # Kısa vadeli HHV/LLV
            'breakout_period_2': [20, 30, 40],      # Orta vadeli HHV/LLV
            'breakout_period_3': [40, 60, 80],      # Uzun vadeli HHV/LLV (YENİ)
        },
        is_independent=True,
        default_values={'momentum_period': 5, 'breakout_period_1': 14, 
                       'breakout_period_2': 30, 'breakout_period_3': 60}
    ),
    ParameterGroup(
        name="MFI_Hacim",
        params={
            'mfi_period': [10, 14, 20],
            'mfi_hhv_period': [10, 14, 20],
            'mfi_llv_period': [10, 14, 20],
            'vol_hhv_period': [10, 14, 20],
        },
        is_independent=True,
        default_values={'mfi_period': 14, 'mfi_hhv_period': 14, 
                       'mfi_llv_period': 14, 'vol_hhv_period': 14}
    ),
    ParameterGroup(
        name="Cikis_ATR",
        params={
            'atr_exit_period': [10, 14, 20],
            'atr_sl_mult': [1.0, 1.5, 2.0, 2.5],
            'atr_tp_mult': [2.0, 3.0, 4.0, 5.0],
            'atr_trail_mult': [1.0, 1.5, 2.0],
            # Çift Teyitli Trend Dönüşü Parametreleri (YENİ)
            'exit_confirm_bars': [1, 2, 3],         # Kaç bar ARS karşı tarafında kapanmalı
            'exit_confirm_mult': [0.5, 1.0, 1.5],   # ARS mesafe çarpanı (dinamikK × mult)
        },
        is_independent=False,  # Kademeli - giriş sabit, çıkış optimize
        default_values={'atr_exit_period': 14, 'atr_sl_mult': 1.5, 
                       'atr_tp_mult': 3.0, 'atr_trail_mult': 1.5,
                       'exit_confirm_bars': 2, 'exit_confirm_mult': 1.0}
    ),
]


class Strategy2HybridOptimizer(HybridGroupOptimizer):
    """Strateji 2 için özelleştirilmiş hibrit optimizer"""
    
    def __init__(self):
        super().__init__(STRATEGY2_GROUPS)
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Strateji 2 parametrelerini değerlendir (Planlanmış Mimari v4.1)"""
        global g_cache
        
        # Import additional indicators
        from src.indicators.core import ARS_Dynamic, Momentum, HHV, LLV, MoneyFlowIndex
        
        closes = g_cache.closes
        highs = g_cache.highs
        lows = g_cache.lows
        typical = g_cache.typical
        lots = g_cache.lots
        n = len(closes)
        
        # ARS Dinamik - dinamikK'yı da döndürecek şekilde hesapla
        ars_ema = int(params.get('ars_ema_period', 5))
        ars_atr_p = int(params.get('ars_atr_period', 14))
        ars_atr_m = float(params.get('ars_atr_mult', 1.0))
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
        
        # ATR ve dinamikK hesapla (çıkış mesafe teyidi için)
        from src.indicators.core import ATR as ATR_fn, EMA
        atr_exit_p = int(params.get('atr_exit_period', 14))
        atr_key = f'atr_{atr_exit_p}'
        if atr_key not in g_cache._cache:
            g_cache._cache[atr_key] = np.array(ATR_fn(highs.tolist(), lows.tolist(), closes.tolist(), atr_exit_p))
        atr = g_cache._cache[atr_key]
        
        ema_key = f'ema_{ars_ema}'
        if ema_key not in g_cache._cache:
            g_cache._cache[ema_key] = np.array(EMA(typical.tolist(), ars_ema))
        ars_ema_arr = g_cache._cache[ema_key]
        
        # dinamikK: ATR bazlı band genişliği (çıkış mesafe teyidi için)
        dinamikK = np.zeros(n)
        for i in range(n):
            if ars_ema_arr[i] > 0:
                dinamikK[i] = (atr[i] / ars_ema_arr[i]) * ars_atr_m
                dinamikK[i] = max(ars_min, min(ars_max, dinamikK[i]))
            else:
                dinamikK[i] = ars_min
        
        # Momentum
        mom_p = int(params.get('momentum_period', 5))
        mom_key = f'mom_{mom_p}'
        if mom_key not in g_cache._cache:
            g_cache._cache[mom_key] = np.array(Momentum(closes.tolist(), mom_p))
        mom = g_cache._cache[mom_key]
        
        # Çoklu HHV/LLV (3 farklı periyot)
        brk1 = int(params.get('breakout_period_1', 14))
        brk2 = int(params.get('breakout_period_2', 30))
        brk3 = int(params.get('breakout_period_3', 60))
        
        hhv1_key, llv1_key = f'hhv_{brk1}', f'llv_{brk1}'
        hhv2_key, llv2_key = f'hhv_{brk2}', f'llv_{brk2}'
        hhv3_key, llv3_key = f'hhv_{brk3}', f'llv_{brk3}'
        
        if hhv1_key not in g_cache._cache:
            g_cache._cache[hhv1_key] = np.array(HHV(highs.tolist(), brk1))
            g_cache._cache[llv1_key] = np.array(LLV(lows.tolist(), brk1))
        if hhv2_key not in g_cache._cache:
            g_cache._cache[hhv2_key] = np.array(HHV(highs.tolist(), brk2))
            g_cache._cache[llv2_key] = np.array(LLV(lows.tolist(), brk2))
        if hhv3_key not in g_cache._cache:
            g_cache._cache[hhv3_key] = np.array(HHV(highs.tolist(), brk3))
            g_cache._cache[llv3_key] = np.array(LLV(lows.tolist(), brk3))
            
        hhv1, llv1 = g_cache._cache[hhv1_key], g_cache._cache[llv1_key]
        hhv2, llv2 = g_cache._cache[hhv2_key], g_cache._cache[llv2_key]
        hhv3, llv3 = g_cache._cache[hhv3_key], g_cache._cache[llv3_key]
        
        # MFI Breakout (RSI yerine)
        mfi_p = int(params.get('mfi_period', 14))
        mfi_key = f'mfi_{mfi_p}'
        if mfi_key not in g_cache._cache:
            g_cache._cache[mfi_key] = np.array(MoneyFlowIndex(highs.tolist(), lows.tolist(), closes.tolist(), lots.tolist(), mfi_p))
        mfi = g_cache._cache[mfi_key]
        
        # MFI HHV/LLV
        mfi_hhv_p = int(params.get('mfi_hhv_period', 14))
        mfi_llv_p = int(params.get('mfi_llv_period', 14))
        mfi_hhv_key = f'mfi_hhv_{mfi_p}_{mfi_hhv_p}'
        mfi_llv_key = f'mfi_llv_{mfi_p}_{mfi_llv_p}'
        if mfi_hhv_key not in g_cache._cache:
            g_cache._cache[mfi_hhv_key] = np.array(HHV(mfi.tolist(), mfi_hhv_p))
        if mfi_llv_key not in g_cache._cache:
            g_cache._cache[mfi_llv_key] = np.array(LLV(mfi.tolist(), mfi_llv_p))
        mfi_hhv = g_cache._cache[mfi_hhv_key]
        mfi_llv = g_cache._cache[mfi_llv_key]
        
        # Volume HHV
        vol_hhv_p = int(params.get('vol_hhv_period', 14))
        vol_hhv_key = f'vol_hhv_{vol_hhv_p}'
        if vol_hhv_key not in g_cache._cache:
            g_cache._cache[vol_hhv_key] = np.array(HHV(lots.tolist(), vol_hhv_p))
        vol_hhv = g_cache._cache[vol_hhv_key]
        
        # Çıkış parametreleri
        atr_sl = float(params.get('atr_sl_mult', 1.5))
        atr_tp = float(params.get('atr_tp_mult', 3.0))
        atr_trail = float(params.get('atr_trail_mult', 1.5))
        exit_confirm_bars = int(params.get('exit_confirm_bars', 2))
        exit_confirm_mult = float(params.get('exit_confirm_mult', 1.0))
        
        # Trend yönü
        trend = np.zeros(n, dtype=int)
        trend[closes > ars] = 1
        trend[closes < ars] = -1
        
        # Backtest değişkenleri
        pos = 0
        entry_price = 0.0
        extreme_price = 0.0
        entry_atr = 0.0
        bars_against_trend = 0  # Çift teyit için sayaç
        
        gross_profit = 0.0
        gross_loss = 0.0
        trades = 0
        max_dd = 0.0
        peak_equity = 0.0
        current_equity = 0.0
        
        warmup = max(brk3, 60)  # En uzun periyot kadar warmup
        
        for i in range(warmup, n):
            current_trend = trend[i]
            current_dinamikK = dinamikK[i]
            
            # ========== EXIT MANTIGI ==========
            if pos == 1:
                if closes[i] > extreme_price:
                    extreme_price = closes[i]
                
                exit_signal = False
                
                # 1. Çift Teyitli Trend Dönüşü
                if current_trend == -1:
                    bars_against_trend += 1
                    # Mesafe teyidi: Fiyat ARS'tan (dinamikK × mult) kadar uzaklaştı mı?
                    distance_threshold = ars[i] * (1 - current_dinamikK * exit_confirm_mult)
                    distance_ok = closes[i] < distance_threshold
                    # Çoklu bar teyidi
                    if bars_against_trend >= exit_confirm_bars and distance_ok:
                        exit_signal = True
                else:
                    bars_against_trend = 0  # Trend döndü, sayacı sıfırla
                
                # 2. Take Profit (ATR bazlı)
                if closes[i] >= entry_price + entry_atr * atr_tp:
                    exit_signal = True
                
                # 3. Stop Loss (ATR bazlı)
                if closes[i] <= entry_price - entry_atr * atr_sl:
                    exit_signal = True
                
                # 4. Trailing Stop (ATR bazlı)
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
                
                # 4. Trailing Stop (ATR bazlı)
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
            
            # ========== ENTRY MANTIGI ==========
            if pos == 0:
                if current_trend == 1:
                    # Fiyat breakout (3 periyottan en az biriyle)
                    price_ok = (closes[i] > hhv1[i-1] if i > 0 else False) or \
                               (closes[i] > hhv2[i-1] if i > brk2 else False) or \
                               (closes[i] > hhv3[i-1] if i > brk3 else False)
                    # Momentum pozitif
                    mom_ok = mom[i] > 100
                    # MFI Breakout (RSI yerine)
                    mfi_ok = mfi[i] >= mfi_hhv[i-1] if i > 0 else False
                    # Hacim teyidi
                    vol_ok = lots[i] >= vol_hhv[i-1] * 0.8 if i > 0 else False
                    
                    if price_ok and mom_ok and mfi_ok and vol_ok:
                        pos = 1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        entry_atr = atr[i] if atr[i] > 0 else 1.0
                        bars_against_trend = 0
                        trades += 1
                        
                elif current_trend == -1:
                    price_ok = (closes[i] < llv1[i-1] if i > 0 else False) or \
                               (closes[i] < llv2[i-1] if i > brk2 else False) or \
                               (closes[i] < llv3[i-1] if i > brk3 else False)
                    mom_ok = mom[i] < 100
                    mfi_ok = mfi[i] <= mfi_llv[i-1] if i > 0 else False
                    vol_ok = lots[i] >= vol_hhv[i-1] * 0.8 if i > 0 else False
                    
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

