# -*- coding: utf-8 -*-
"""
Hybrid Group Optimizer v1.1
===========================
Hibrit yaklaşım: Önce grupları bağımsız optimize et, sonra kombine et.
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

from src.indicators.core import EMA, ATR, ADX, SMA, ARS, NetLot, MACDV, Momentum, HHV, LLV
from src.indicators.trend import TOMA
from src.indicators.oscillators import TRIX
from src.strategies.score_based import ScoreBasedStrategy
from src.strategies.ars_trend_v2 import ARSTrendStrategyV2
from src.strategies.paradise_strategy import ParadiseStrategy
from src.strategies.paradise_strategy import ParadiseStrategy
from src.optimization.fitness import quick_fitness, calculate_sharpe
from src.strategies.holidays import vade_sonu_is_gunu

# Opsiyonel: Veritabanı entegrasyonu
try:
    from src.core.database import db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    db = None

# ==============================================================================
# SATELLITE-DRONE HELPER FUNCTIONS
# ==============================================================================

# Parametre tipi tanımları: (min_range, max_range, satellite_step, drone_step)
PARAM_TYPE_CONFIG = {
    'period_short': (1, 15, 2, 1),      # ars_period, macdv_signal
    'period_medium': (15, 50, 5, 2),    # adx_period, bb_period
    'period_long': (50, 200, 10, 5),    # bb_avg_period
    'k_factor': (0.001, 0.1, 0.005, 0.001),  # ars_k, ars_atr_mult
    'threshold_int': (1, 10, 1, 1),     # min_score, exit_score
    'threshold_float': (10.0, 50.0, 5.0, 1.0),  # adx_threshold, netlot_threshold
    'multiplier': (0.5, 3.0, 0.5, 0.1),  # bb_std, atr_sl_mult
    'threshold_momentum': (50.0, 200.0, 10.0, 5.0),   # Momentum scale (0-200 arası)
    'momentum_band': (90.0, 110.0, 1.0, 0.5),          # Momentum bant (mom_alt, mom_ust)
    'multiplier_wide': (1.0, 10.0, 1.0, 0.25),         # Geniş çarpanlar (TP mult gibi)
    'period_short_wide': (5, 20, 2, 1),                 # Kısa-orta arası periyotlar
}

# Hangi parametrenin hangi tipte olduğunu belirler
PARAM_TYPES = {
    # Strateji 1
    'ars_period': 'period_short', 'ars_k': 'k_factor',
    'adx_period': 'period_medium', 'adx_threshold': 'threshold_float',
    'macdv_short': 'period_short', 'macdv_long': 'period_medium', 'macdv_signal': 'period_short', 'macdv_threshold': 'multiplier',
    'netlot_period': 'period_short', 'netlot_threshold': 'threshold_float',
    'bb_period': 'period_medium', 'bb_std': 'multiplier', 'bb_avg_period': 'period_long', 'bb_width_multiplier': 'multiplier',
    'ars_mesafe_threshold': 'k_factor', 'yatay_ars_bars': 'period_short', 'yatay_adx_threshold': 'threshold_float',
    'filter_score_threshold': 'threshold_int', 'min_score': 'threshold_int', 'exit_score': 'threshold_int',
    'contrary_score_max': 'threshold_int',
    # Strateji 2
    'ars_ema_period': 'period_short', 'ars_atr_period': 'period_short', 'ars_atr_mult': 'multiplier',
    'ars_min_band': 'k_factor', 'ars_max_band': 'k_factor',
    'momentum_period': 'period_short', 'momentum_threshold': 'threshold_momentum', 'breakout_period': 'period_short_wide',
    'mfi_period': 'period_medium', 'mfi_hhv_period': 'period_medium', 'mfi_llv_period': 'period_medium', 'volume_hhv_period': 'period_medium',
    'atr_exit_period': 'period_medium', 'atr_sl_mult': 'multiplier', 'atr_tp_mult': 'multiplier_wide', 'atr_trail_mult': 'multiplier',
    'exit_confirm_bars': 'threshold_int', 'exit_confirm_mult': 'multiplier', 'volume_mult': 'multiplier', 'volume_llv_period': 'period_medium',
    # Strateji 3 (Paradise)
    'ema_period': 'period_medium', 'dsma_period': 'period_long', 'ma_period': 'period_medium',
    'hh_period': 'period_medium', 'vol_hhv_period': 'period_medium',
    'mom_period': 'period_long', 'mom_alt': 'momentum_band', 'mom_ust': 'momentum_band',
    'atr_period': 'period_medium', 'atr_sl': 'multiplier', 'atr_tp': 'multiplier_wide', 'atr_trail': 'multiplier',
}

def get_step(param_name: str, stage: str = 'satellite', user_step: float = None) -> float:
    """Parametre için Satellite veya Drone adım boyutunu döndürür."""
    param_type = PARAM_TYPES.get(param_name, 'period_medium')
    config = PARAM_TYPE_CONFIG.get(param_type, (1, 100, 5, 1))
    
    # Satellite: En az config'deki kadar veya kullanıcı adımı kadar geniş
    if stage == 'satellite':
        base_step = config[2]
        if user_step is not None:
            return max(base_step, user_step)
        return base_step
    
    # Drone: Config'deki drone adımını kullan, ama kullanıcı adımından küçük olmasın
    # Bu şekilde period_short=1, period_medium=2, period_long=5 farklılıkları korunur
    else:
        drone_base = config[3]  # Parametre tipine göre uygun ince adım
        if user_step is not None:
            # Kullanıcı adımının yarısını taban al, ama config'den küçük olmasın
            return max(drone_base, user_step / 2.0)
        return drone_base





def generate_range(min_val: float, max_val: float, step: float, is_int: bool = False) -> List:
    """Belirli aralık ve adım için değer listesi üretir."""
    if step <= 0: step = 1
    result = []
    current = min_val
    while current <= max_val + (step * 0.01):  # Floating point tolerance
        result.append(int(round(current)) if is_int else round(current, 4))
        current += step
    return result if result else [min_val]

def find_cluster_range(results: List[Dict], param_name: str, original_step: float, original_min: float, original_max: float) -> Tuple[float, float]:
    """
    Kümeleme tabanlı aralık daralması.
    "İyi" sonuçların kümelendiği bölgeyi tespit eder ve 
    o bölgenin min-1adım / max+1adım'ını döndürür.
    """
    if not results: return (original_min, original_max)
    
    # Fitness ortalaması ve standart sapması
    fitnesses = [r.get('fitness', r.get('net_profit', 0)) for r in results]
    if not fitnesses: return (original_min, original_max)
    
    mean_fit = sum(fitnesses) / len(fitnesses)
    variance = sum((f - mean_fit) ** 2 for f in fitnesses) / len(fitnesses)
    std_fit = variance ** 0.5 if variance > 0 else 0
    
    # Eşik: Ortalama + (Std / 2)
    threshold = mean_fit + (std_fit / 2) if std_fit > 0 else mean_fit * 0.8
    
    # "İyi" sonuçları filtrele
    good_results = [r for r, f in zip(results, fitnesses) if f >= threshold]
    if not good_results: good_results = results[:max(1, len(results) // 5)]  # Top %20
    
    # Bu parametre için değerleri topla
    param_values = [r.get(param_name) for r in good_results if param_name in r]
    if not param_values: return (original_min, original_max)
    
    cluster_min = min(param_values)
    cluster_max = max(param_values)
    
    # Bir adım genişlet (boundary'leri kaçırmamak için)
    new_min = max(original_min, cluster_min - original_step)
    new_max = min(original_max, cluster_max + original_step)
    
    return (new_min, new_max)


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
        params={'ars_period': [3, 4, 5, 8, 10, 12], 'ars_k': [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]},
        is_independent=True,
        default_values={'ars_period': 3, 'ars_k': 0.01}
    ),
    ParameterGroup(
        name="ADX",
        params={'adx_period': [14, 17, 21, 25, 30], 'adx_threshold': [20.0, 25.0, 30.0]},
        is_independent=True,
        default_values={'adx_period': 17, 'adx_threshold': 25.0}
    ),
    ParameterGroup(
        name="MACDV",
        params={'macdv_short': [10, 13, 15], 'macdv_long': [24, 28, 32], 'macdv_signal': [7, 8, 9], 'macdv_threshold': [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]},
        is_independent=True,
        default_values={'macdv_short': 13, 'macdv_long': 28, 'macdv_signal': 8, 'macdv_threshold': 0.0}
    ),

    ParameterGroup(
        name="NetLot",
        params={'netlot_period': [3, 5, 8], 'netlot_threshold': [10, 20, 30, 40]},
        is_independent=True,
        default_values={'netlot_period': 5, 'netlot_threshold': 20.0}
    ),
    ParameterGroup(
        name="Yatay_BB",
        params={
            'bb_period': [15, 20, 25], 'bb_std': [1.5, 2.0, 2.5],
            'bb_width_multiplier': [0.6, 0.8, 1.0], 'bb_avg_period': [30, 50, 70],
        },
        is_independent=True,
        default_values={
            'bb_period': 20, 'bb_std': 2.0,
            'bb_width_multiplier': 0.8, 'bb_avg_period': 50
        }
    ),
    ParameterGroup(
        name="Skor_Ayarlari",
        params={'min_score': [2, 3, 4], 'exit_score': [2, 3, 4], 'contrary_score_max': [1, 2, 3]},
        is_independent=True,
        default_values={'min_score': 3, 'exit_score': 3, 'contrary_score_max': 2}
    ),
    ParameterGroup(
        name="Yatay_Onay",
        params={
            'ars_mesafe_threshold': [0.20, 0.25, 0.30], 'yatay_ars_bars': [5, 10, 15],
            'yatay_adx_threshold': [15.0, 20.0, 25.0], 'filter_score_threshold': [1, 2, 3],
        },
        is_independent=False,
        default_values={
            'ars_mesafe_threshold': 0.25, 'yatay_ars_bars': 10,
            'yatay_adx_threshold': 20.0, 'filter_score_threshold': 2
        }
    ),
]

# Strateji 2 Grup Tanımları
STRATEGY2_GROUPS = [
    ParameterGroup(
        name="ARS",
        params={
            'ars_ema_period': [2, 3, 5, 8], 'ars_atr_period': [7, 10, 14], 'ars_atr_mult': [0.5, 0.8, 1.0],
            'ars_min_band': [0.002, 0.003], 'ars_max_band': [0.015, 0.020],
        },
        is_independent=True,
        default_values={'ars_ema_period': 3, 'ars_atr_period': 10, 'ars_atr_mult': 0.5, 'ars_min_band': 0.002, 'ars_max_band': 0.015}
    ),
    ParameterGroup(
        name="Giris_Momentum",
        params={
            'momentum_period': [5, 7, 10],
            'momentum_threshold': [90.0, 100.0, 120.0, 150.0],
            # momentum_base sabit 200.0 - optimize edilmemeli (Short sinyallerini bozar)
            'breakout_period': [8, 10, 15],
        },
        is_independent=True,
        default_values={
            'momentum_period': 5, 'momentum_threshold': 100.0,
            'breakout_period': 10,
        }
    ),
    ParameterGroup(
        name="Giris_MFI_Volume",
        params={
            'mfi_period': [10, 14, 17],
            'mfi_hhv_period': [10, 14, 20],
            'mfi_llv_period': [10, 14, 20],
            'volume_hhv_period': [10, 14, 20],
        },
        is_independent=True,
        default_values={
            'mfi_period': 14, 'mfi_hhv_period': 14,
            'mfi_llv_period': 14, 'volume_hhv_period': 14
        }
    ),
    ParameterGroup(
        name="Cikis_Risk",
        params={
            'atr_exit_period': [14, 17],
            'atr_sl_mult': [1.5, 2.0, 2.5],
            'atr_tp_mult': [3.0, 4.0, 5.0, 6.0],
            'atr_trail_mult': [1.5, 2.0, 3.0],
            'exit_confirm_bars': [2, 3],
            'exit_confirm_mult': [0.75, 1.0, 1.25],
        },
        is_independent=True,
        default_values={
            'atr_exit_period': 14, 'atr_sl_mult': 2.0, 'atr_tp_mult': 5.0,
            'atr_trail_mult': 2.0, 'exit_confirm_bars': 2, 'exit_confirm_mult': 1.0
        }
    ),
    ParameterGroup(
        name="Ince_Ayar",
        params={'volume_mult': [0.6, 0.8, 1.0], 'volume_llv_period': [14, 17]},
        is_independent=False,
        default_values={'volume_mult': 0.8, 'volume_llv_period': 14}
    ),
]

# Strateji 3 (Paradise) Grup Tanimlari
STRATEGY3_GROUPS = [
    ParameterGroup(
        name="Trend",
        params={
            'ema_period': [5, 8, 10, 13, 15, 18, 21, 25, 30, 40, 50, 60, 80],
            'dsma_period': [15, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150],
            'ma_period': [5, 8, 10, 13, 15, 18, 20, 25, 30, 40, 50, 60, 80],
        },
        is_independent=True,
        default_values={'ema_period': 21, 'dsma_period': 50, 'ma_period': 20}
    ),
    ParameterGroup(
        name="Breakout",
        params={
            'hh_period': [5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80],
            'vol_hhv_period': [5, 8, 10, 14, 18, 20, 25, 30, 40, 50],
        },
        is_independent=True,
        default_values={'hh_period': 25, 'vol_hhv_period': 14}
    ),
    ParameterGroup(
        name="Momentum",
        params={
            'mom_period': [10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150],
            'mom_alt': [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 99.5],
            'mom_ust': [100.5, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        },
        is_independent=True,
        default_values={'mom_period': 60, 'mom_alt': 98.0, 'mom_ust': 102.0}
    ),
    ParameterGroup(
        name="Risk",
        params={
            'atr_period': [5, 7, 10, 14, 18, 20, 25, 30],
            'atr_sl': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            'atr_tp': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            'atr_trail': [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        },
        is_independent=False,
        default_values={'atr_period': 14, 'atr_sl': 2.0, 'atr_tp': 4.0, 'atr_trail': 2.5}
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
    df.dropna(inplace=True)
    df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['DateTime']).reset_index(drop=True)
    return df

class IndicatorCache:
    """
    Turbo Indicator Cache - Tüm indikatörleri önbelleğe alır.
    Her işlemci çekirdeği başına bir tane oluşturulur.
    Aynı parametrelerle çağrılan indikatörler cache'den döner (hesaplama yapılmaz).
    """
    def __init__(self, df):
        self.df = df
        open_col, high_col, low_col, close_col, vol_col = 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Lot'
        self.opens = df[open_col].values.flatten()
        self.closes = df[close_col].values.flatten()
        self.highs = df[high_col].values.flatten()
        self.lows = df[low_col].values.flatten()
        self.typical = df['Tipik'].values.flatten()
        self.lots = df[vol_col].values.flatten()
        self.volumes = df[vol_col].values.flatten()
        self.n = len(self.closes)
        self.times = df['DateTime'].tolist()
        self.dates = self.times
        self._cache = {}
    
    def _get(self, key: str, calc_fn):
        """Generic cache getter"""
        if key not in self._cache:
            self._cache[key] = calc_fn()
        return self._cache[key]

    # === ARS ===
    def get_ars(self, period: int, k: float) -> np.ndarray:
        key = f'ars_{period}_{k:.4f}'
        return self._get(key, lambda: np.array(ARS(self.typical.tolist(), int(period), float(k))))

    # === ADX ===
    def get_adx(self, period: int) -> np.ndarray:
        key = f'adx_{period}'
        return self._get(key, lambda: np.array(ADX(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(period))))

    # === MACDV ===
    def get_macdv(self, short: int, long: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
        key = f'macdv_{short}_{long}_{signal}'
        def calc():
            m, s = MACDV(self.closes.tolist(), self.highs.tolist(), self.lows.tolist(), int(short), int(long), int(signal))
            return (np.array(m), np.array(s))
        return self._get(key, calc)

    # === NetLot ===
    def get_netlot(self, period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        key = f'netlot_{period}'
        def calc():
            nl = np.array(NetLot(self.opens.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist()))
            nl_ma = pd.Series(nl).rolling(int(period)).mean().fillna(0).values
            return (nl, nl_ma)
        return self._get(key, calc)

    # === EMA ===
    def get_ema(self, period: int) -> np.ndarray:
        key = f'ema_{period}'
        return self._get(key, lambda: np.array(EMA(self.typical.tolist(), int(period))))

    # === ATR ===
    def get_atr(self, period: int) -> np.ndarray:
        key = f'atr_{period}'
        return self._get(key, lambda: np.array(ATR(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(period))))

    # === SMA ===
    def get_sma(self, period: int) -> np.ndarray:
        key = f'sma_{period}'
        return self._get(key, lambda: pd.Series(self.closes).rolling(int(period)).mean().fillna(0).values)

    # === Bollinger Bands ===
    def get_bb(self, period: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = f'bb_{period}_{std_mult:.1f}'
        def calc():
            sma = pd.Series(self.closes).rolling(int(period)).mean()
            std = pd.Series(self.closes).rolling(int(period)).std()
            upper = (sma + std_mult * std).fillna(0).values
            lower = (sma - std_mult * std).fillna(0).values
            width = np.where(sma != 0, ((upper - lower) / sma) * 100, 0)
            return (upper, lower, width)
        return self._get(key, calc)

    # === BB Width Avg ===
    def get_bb_width_avg(self, bb_period: int, bb_std: float, avg_period: int) -> np.ndarray:
        key = f'bb_width_avg_{bb_period}_{bb_std:.1f}_{avg_period}'
        def calc():
            _, _, width = self.get_bb(bb_period, bb_std)
            return pd.Series(width).rolling(int(avg_period)).mean().fillna(0).values
        return self._get(key, calc)

    # === Paradise Indicator Metodlari ===
    
    # === DSMA (Double SMA) ===
    def get_dsma(self, period: int) -> np.ndarray:
        key = f'dsma_{period}'
        def calc():
            inner = self.get_sma(period)
            return pd.Series(inner).rolling(int(period)).mean().fillna(0).values
        return self._get(key, calc)

    # === Momentum ===
    def get_momentum(self, period: int) -> np.ndarray:
        key = f'mom_{period}'
        return self._get(key, lambda: np.array(Momentum(self.closes.tolist(), int(period))))

    # === HHV ===
    def get_hhv(self, period: int) -> np.ndarray:
        key = f'hhv_{period}'
        return self._get(key, lambda: np.array(HHV(self.highs.tolist(), int(period))))

    # === LLV ===
    def get_llv(self, period: int) -> np.ndarray:
        key = f'llv_{period}'
        return self._get(key, lambda: np.array(LLV(self.lows.tolist(), int(period))))

    # === TRIX ===
    def get_trix(self, period: int) -> np.ndarray:
        key = f'trix_{period}'
        return self._get(key, lambda: np.array(TRIX(self.closes.tolist(), int(period))))

    # === TOMA ===
    def get_toma(self, period: int, opt: float) -> Tuple[np.ndarray, np.ndarray]:
        key = f'toma_{period}_{opt:.2f}'
        def calc():
            toma_val, trend = TOMA(self.closes.tolist(), int(period), float(opt))
            return (np.array(toma_val), np.array(trend))
        return self._get(key, calc)

    # === Volume HHV ===
    def get_vol_hhv(self, period: int) -> np.ndarray:
        key = f'vol_hhv_{period}'
        # HHV fonksiyonu list bekler, volumes array'ini list'e cevir
        return self._get(key, lambda: np.array(HHV(self.volumes.tolist(), int(period))))

    # === Vade Tarihleri ===
    def get_vade_dates(self, vade_tipi: str) -> set:
        key = f'vade_dates_{vade_tipi}'
        def calc():
            vade_dates = set()
            # self.times datetime objeleri listesi olarak varsayilir (IndicatorCache.__init__ df['DateTime'].tolist() yapiyor)
            # Guvenlik icin pd.to_datetime kullanabiliriz ama maliyetli olabilir. 
            # Eger df['DateTime'] zaten datetimelike ise gerek yok.
            # Ancak process safe olmasi icin array'i Series'e cevirip dt accessor kullanmak en iyisi.
            dates = pd.to_datetime(self.times)
            months = dates.to_period('M').unique()
            
            for m in months:
                if vade_tipi == "ENDEKS" and m.month % 2 != 0:
                    continue
                month_date = m.to_timestamp().date()
                vade_gunu = vade_sonu_is_gunu(month_date, vade_tipi)
                vade_dates.add(vade_gunu)
            return vade_dates
        return self._get(key, calc)

    def get_vade_transitions(self, vade_tipi: str) -> set:
        key = f'vade_trans_{vade_tipi}'
        def calc():
            transitions = set()
            dates = pd.to_datetime(self.times)
            # Vectorized approach is faster than loop
            # Pandas shift ile onceki ayi karsilastir
            months = dates.month
            # shift(1) nan getirir, fillna ile ilk ayi koru
            prev_months = pd.Series(months).shift(1).fillna(months[0])
            
            # Ay degisimi olan indeksler
            change_mask = months != prev_months
            change_indices = np.where(change_mask)[0]
            
            for i in change_indices:
                m = months[i]
                if vade_tipi == "ENDEKS" and m % 2 != 0:
                    # Tek ay ise GECIS yap (Cunku vade sonu CIFT aydadir, 
                    # tek ay basinda eski kontrat biter yeni kontrat baslar mi? 
                    # Hayir, ENDEKS kontratlari CIFT aylarda biter.
                    # Ornegin Subat(2) sonu vade biter, Mart(3) basi GECIS olur.
                    # Yani Mart(3) basinda gecis olmali. Mart tek aydir (3 % 2 == 1).
                    # Yani m % 2 == 1 ise transition ekle.
                    transitions.add(i)
                elif vade_tipi == "SPOT":
                    # Her ay gecis
                    transitions.add(i)
            return transitions
        return self._get(key, calc)



# ==============================================================================
# GLOBAL HELPERS
# ==============================================================================
def _init_group_pool(strategy_index):
    global g_cache
    if g_cache is None:
        df = load_data()
        g_cache = IndicatorCache(df)

def _eval_combo_wrapper(params_and_strategy_and_costs):
    params, strategy_index, commission, slippage = params_and_strategy_and_costs
    return _evaluate_params_static(params, strategy_index, commission, slippage)

def _evaluate_params_static(params: Dict[str, Any], strategy_index: int, commission: float = 0.0, slippage: float = 0.0) -> Dict[str, float]:
    global g_cache
    if g_cache is None:
        _init_group_pool(strategy_index)
        
    if strategy_index == 0:
        strategy = ScoreBasedStrategy.from_config_dict(g_cache, params)
    elif strategy_index == 2:
        strategy = ParadiseStrategy.from_config_dict(g_cache, params)
    else:
        strategy = ARSTrendStrategyV2.from_config_dict(g_cache, params)
    
    signals, exits_long, exits_short = strategy.generate_all_signals()
    
    # Trading days calculation
    trading_days = 252.0
    if g_cache.dates and len(g_cache.dates) > 1:
        try:
            # g_cache.dates is a list of datetime or strings? 
            # IndicatorCache converts them to list in __init__
            # Let's assume they are comparable or convertable
            start_date = g_cache.dates[0]
            end_date = g_cache.dates[-1]
            if hasattr(start_date, 'date'):
                delta = end_date - start_date
                trading_days = delta.days
            else:
                 # String format fallback if needed, but IndicatorCache usually has datetime objects if parsed correctly
                 pass
        except:
            pass
            
    np_val, trades, pf, dd, sharpe = fast_backtest(g_cache.closes, signals, exits_long, exits_short, commission, slippage, trading_days=trading_days)
    
    # Fitness hesapla - Maliyetler np_val icinde dusuruldu, çift sayımı önlemek için 0.0 gonderilmeli
    fit = quick_fitness(np_val, pf, dd, trades, sharpe=sharpe, commission=0.0, slippage=0.0)
    
    return {'net_profit': np_val, 'trades': trades, 'pf': pf, 'max_dd': dd, 'sharpe': sharpe, 'fitness': fit}

def fast_backtest(closes, signals, exits_long, exits_short, commission: float = 0.0, slippage: float = 0.0, trading_days: float = 252.0) -> Tuple[float, int, float, float, float]:
    pos, entry_price, gross_profit, gross_loss, trades, max_dd, peak_equity, current_equity = 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0
    
    # Sharpe hesabı için (Welford's algorithm benzeri kümülatif m2)
    # R: Toplam pnl of trade, N: trade count, M2: Sum of squares of differences from the current mean
    # Yıllık Sharpe = (Mean / Std) * Sqrt(Yıllık Trade Sayısı)
    trade_pnls = []

    cost_per_trade = commission + slippage
    n = len(closes)
    
    for i in range(50, n):
        if pos == 1 and exits_long[i]:
            pnl = (closes[i] - entry_price) - cost_per_trade
            trade_pnls.append(pnl)
            if pnl > 0: gross_profit += pnl
            else: gross_loss += abs(pnl)
            current_equity += pnl
            peak_equity = max(peak_equity, current_equity)
            max_dd = max(max_dd, peak_equity - current_equity)
            pos = 0
            trades += 1
        elif pos == -1 and exits_short[i]:
            pnl = (entry_price - closes[i]) - cost_per_trade
            trade_pnls.append(pnl)
            if pnl > 0: gross_profit += pnl
            else: gross_loss += abs(pnl)
            current_equity += pnl
            peak_equity = max(peak_equity, current_equity)
            max_dd = max(max_dd, peak_equity - current_equity)
            pos = 0
            trades += 1

        if pos == 0:
            if signals[i] == 1: pos = 1; entry_price = closes[i]
            elif signals[i] == -1: pos = -1; entry_price = closes[i]

            
    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    # Basit Sharpe (Trade-based)
    sharpe = 0.0
    if len(trade_pnls) > 1:
        # Gerçek yıllık trade sayısını hesapla
        # Eğer trading_days 0 veya çok küçükse default 252 kullan
        if trading_days < 1: trading_days = 252.0
        
        # Yıllık trade frekansı = Toplam Trade / (Toplam Gün / 252)
        # Yani: trades * (252 / trading_days)
        trades_per_year_metric = len(trade_pnls) * (252.0 / trading_days)
        
        sharpe = calculate_sharpe(np.array(trade_pnls), trades_per_year=trades_per_year_metric)

    return net_profit, trades, pf, max_dd, sharpe

def backtest_with_trades(closes, signals, exits_long, exits_short, commission: float = 0.0, slippage: float = 0.0) -> List[float]:
    """Her işlemin PnL listesini döndürür (Monte Carlo için)"""
    pos, entry_price, trades_pnl = 0, 0.0, []
    cost_per_trade = commission + slippage
    for i in range(50, len(closes)):
        if pos == 1 and exits_long[i]:
            pnl = (closes[i] - entry_price) - cost_per_trade
            trades_pnl.append(float(pnl))
            pos = 0
        elif pos == -1 and exits_short[i]:
            pnl = (entry_price - closes[i]) - cost_per_trade
            trades_pnl.append(float(pnl))
            pos = 0
        if pos == 0:
            if signals[i] == 1: pos = 1; entry_price = closes[i]
            elif signals[i] == -1: pos = -1; entry_price = closes[i]
    return trades_pnl

# ==============================================================================
# HIBRID OPTIMIZER CLASS
# ==============================================================================
class HybridGroupOptimizer:
    def __init__(self, groups: List[ParameterGroup], process_id: str = None, strategy_index: int = 0, 
                 is_cancelled_callback=None, on_progress_callback=None, n_parallel: int = 4, 
                 commission: float = 0.0, slippage: float = 0.0):
        self.groups = groups
        self.independent_groups = [g for g in groups if g.is_independent]
        self.cascaded_groups = [g for g in groups if not g.is_independent]
        self.process_id, self.strategy_index, self.n_parallel, self._is_cancelled = process_id, strategy_index, n_parallel, is_cancelled_callback
        self.on_progress = on_progress_callback
        self.commission, self.slippage = commission, slippage
        self.group_results, self.combined_results, self.final_results = {}, [], []
        self.pool = None  # Process pool for termination support

    def get_default_params(self, exclude_group: str = None) -> Dict[str, Any]:
        defaults = {}
        for g in self.groups:
            if g.name != exclude_group: defaults.update(g.default_values)
        return defaults

    def stop(self):
        """Optimizasyonu dışarıdan durdur"""
        if self._is_cancelled:
            # Callback returns True -> handled in loops
            pass
            
        if self.pool:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception as e:
                print(f"Hybrid Pool Terminate Error: {e}")
            finally:
                self.pool = None

    def generate_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys, values = list(params.keys()), list(params.values())
        return [dict(zip(keys, v)) for v in product(*values)]

    def run_group_optimization(self, group: ParameterGroup, fixed_params: Dict[str, Any] = None) -> List[Dict]:
        print(f"\n=== Grup: {group.name} ===")
        base_params = self.get_default_params(exclude_group=group.name)
        if fixed_params: base_params.update(fixed_params)
        combos = self.generate_combinations(group.params)
        results = []
        if self.n_parallel > 1:
            tasks = [({**base_params, **c}, self.strategy_index, self.commission, self.slippage) for c in combos]
            try:
                self.pool = Pool(processes=self.n_parallel, initializer=_init_group_pool, initargs=(self.strategy_index,))
                raw = self.pool.map(_eval_combo_wrapper, tasks)
                self.pool.close()
                self.pool.join()
                self.pool = None
            except Exception as e:
                print(f"Hybrid Pool Execution Error: {e}")
                if self.pool:
                    self.pool.terminate()
                    self.pool = None
                return []

            for i, score in enumerate(raw):
                if score['net_profit'] > 0: results.append({'group': group.name, **combos[i], **score})
        else:
            _init_group_pool(self.strategy_index)
            for combo in combos:
                if self._is_cancelled and self._is_cancelled(): break
                score = _evaluate_params_static({**base_params, **combo}, self.strategy_index, self.commission, self.slippage)
                if score['net_profit'] > 0: results.append({'group': group.name, **combo, **score})
        results.sort(key=lambda x: x['net_profit'], reverse=True)
        top = results[:10]
        print(f"Bulunan: {len(results)}, Top: {len(top)}")
        return top

    def run_independent_phase(self):
        """Eski bağımsız faz - geriye uyumluluk için korunuyor."""
        print("\nPHASE 1: BAĞIMSIZ"); [self.group_results.update({g.name: self.run_group_optimization(g)}) for g in self.independent_groups]

    def run_satellite_drone_phase(self):
        """
        TURBO: Satellite-Drone 2 aşamalı tarama.
        1. Satellite: Geniş adımlarla kaba tarama
        2. Cluster: İyi sonuçların kümelendiği bölgeyi tespit
        3. Drone: Dar aralık, hassas adımlarla ince tarama
        """
        print("\n" + "="*60)
        print("  SATELLITE-DRONE PHASE (TURBO)")
        print("="*60)
        
        total_independent = len(self.independent_groups)
        for i, group in enumerate(self.independent_groups):
            if self._is_cancelled and self._is_cancelled(): break
            
            progress_base = 10 + (i / total_independent) * 70  # Phase 1 is 10-80%
            
            print(f"\n--- Grup: {group.name} ---")
            
            # === SATELLITE SCAN ===
            msg = f"[SAT] {group.name} Satellite Tarama..."
            print(f"  {msg}")
            if self.on_progress: self.on_progress(int(progress_base), msg)
            
            satellite_params = {}
            param_mins, param_maxs = {}, {}
            # ... (rest of param collection)
            for param_name, values in group.params.items():
                param_min, param_max = min(values), max(values)
                param_mins[param_name], param_maxs[param_name] = param_min, param_max
                
                # Kullanıcının adımını hesapla (ilk iki değer arası fark)
                user_step = 1.0
                if len(values) > 1:
                    user_step = abs(values[1] - values[0])
                
                step = get_step(param_name, 'satellite', user_step=user_step)
                is_int = PARAM_TYPES.get(param_name, '').startswith('period') or PARAM_TYPES.get(param_name, '').startswith('threshold_int')
                satellite_params[param_name] = generate_range(param_min, param_max, step, is_int)
            
            satellite_group = ParameterGroup(
                name=group.name + "_SAT",
                params=satellite_params,
                is_independent=True,
                default_values=group.default_values
            )
            satellite_results = self.run_group_optimization(satellite_group)
            
            if not satellite_results:
                print(f"  [!] Satellite sonuc bulunamadi, varsayilan degerler kullanilacak.")
                self.group_results[group.name] = []
                continue
            
            # === CLUSTER ANALYSIS ===
            print("  [CLU] Kumeleme Analizi...")
            drone_params = {}
            
            for param_name, values in group.params.items():
                # Kullanıcının adımını tekrar hesapla
                user_step = 1.0
                if len(values) > 1:
                    user_step = abs(values[1] - values[0])

                sat_step = get_step(param_name, 'satellite', user_step=user_step)
                new_min, new_max = find_cluster_range(
                    satellite_results, param_name, sat_step, 
                    param_mins[param_name], param_maxs[param_name]
                )
                drone_step = get_step(param_name, 'drone', user_step=user_step)
                is_int = PARAM_TYPES.get(param_name, '').startswith('period') or PARAM_TYPES.get(param_name, '').startswith('threshold_int')
                drone_params[param_name] = generate_range(new_min, new_max, drone_step, is_int)
                
                if new_min != param_mins[param_name] or new_max != param_maxs[param_name]:
                    print(f"    {param_name}: [{param_mins[param_name]}-{param_maxs[param_name]}] => [{new_min}-{new_max}]")
            
            # === DRONE SCAN ===
            msg = f"[DRN] {group.name} Drone Tarama..."
            print(f"  {msg}")
            if self.on_progress: self.on_progress(int(progress_base + (0.5 / total_independent) * 70), msg)
            
            drone_group = ParameterGroup(
                name=group.name + "_DRONE",
                params=drone_params,
                is_independent=True,
                default_values=group.default_values
            )
            drone_results = self.run_group_optimization(drone_group)
            
            all_results = satellite_results + drone_results
            all_results.sort(key=lambda x: x.get('fitness', x.get('net_profit', 0)), reverse=True)
            self.group_results[group.name] = all_results[:15]
            print(f"  [OK] {group.name}: {len(all_results)} sonuc => Top 15 secildi")

    def run_iterative_phase(self, max_rounds: int = 3, convergence_threshold: float = 0.05):
        """
        Iterative Coordinate Descent: Her round'da gruplar diger gruplarin 
        EN IYI bulunan degerlerini kullanarak tekrar optimize edilir.
        """
        print("\n" + "="*60)
        print("  ITERATIVE COORDINATE DESCENT")
        print("="*60)
        
        # Baslangic: Tum gruplar default degerlerle
        current_best = {}
        for g in self.groups:
            current_best[g.name] = g.default_values.copy()
        
        total_groups = len(self.independent_groups)
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n--- ROUND {round_num}/{max_rounds} ---")
            round_start_fitness = {}
            round_end_fitness = {}
            
            for i, group in enumerate(self.independent_groups):
                if self._is_cancelled and self._is_cancelled():
                    break
                
                # Progress: Round ve grup bazli
                base_progress = ((round_num - 1) / max_rounds) * 70
                group_progress = (i / total_groups) * (70 / max_rounds)
                progress = int(10 + base_progress + group_progress)
                
                # Diger gruplarin EN IYI degerlerini kullan (default degil!)
                fixed_params = {}
                for other in self.groups:
                    if other.name != group.name:
                        fixed_params.update(current_best[other.name])
                
                msg = f"[R{round_num}] {group.name} optimizing..."
                print(f"  {msg}")
                if self.on_progress:
                    self.on_progress(progress, msg)
                
                # Onceki fitness
                old_results = self.group_results.get(group.name, [])
                round_start_fitness[group.name] = old_results[0].get('fitness', 0) if old_results else 0
                
                # Satellite-Drone calistir
                satellite_params = self._generate_satellite_params(group)
                satellite_group = ParameterGroup(
                    name=group.name + "_SAT",
                    params=satellite_params,
                    is_independent=True,
                    default_values=group.default_values
                )
                satellite_results = self.run_group_optimization(satellite_group, fixed_params)
                
                if satellite_results:
                    # Cluster bul ve Drone calistir
                    drone_params = self._generate_drone_params(group, satellite_results)
                    if drone_params:
                        drone_group = ParameterGroup(
                            name=group.name + "_DRONE",
                            params=drone_params,
                            is_independent=True,
                            default_values=group.default_values
                        )
                        drone_results = self.run_group_optimization(drone_group, fixed_params)
                        all_results = satellite_results + drone_results
                    else:
                        all_results = satellite_results
                    
                    all_results.sort(key=lambda x: x.get('fitness', x.get('net_profit', 0)), reverse=True)
                    self.group_results[group.name] = all_results[:15]
                    
                    # current_best guncelle
                    if all_results:
                        for key in group.params.keys():
                            if key in all_results[0]:
                                current_best[group.name][key] = all_results[0][key]
                        round_end_fitness[group.name] = all_results[0].get('fitness', 0)
            
            # Yakinsama kontrolu
            max_improvement = 0
            for gname in round_start_fitness:
                if round_start_fitness[gname] > 0:
                    improvement = (round_end_fitness.get(gname, 0) - round_start_fitness[gname]) / abs(round_start_fitness[gname])
                    max_improvement = max(max_improvement, improvement)
            
            print(f"  Round {round_num} Max Improvement: {max_improvement:.1%}")
            
            if round_num > 1 and max_improvement < convergence_threshold:
                print(f"  Converged! (Improvement < {convergence_threshold:.0%})")
                break
        
        return current_best
    
    def _generate_satellite_params(self, group: ParameterGroup) -> dict:
        """Grup icin Satellite parametreleri uret."""
        satellite_params = {}
        for param_name, values in group.params.items():
            param_min, param_max = min(values), max(values)
            user_step = abs(values[1] - values[0]) if len(values) > 1 else 1
            step = get_step(param_name, 'satellite', user_step=user_step)
            is_int = param_name.endswith('period') or param_name.endswith('bars')
            satellite_params[param_name] = generate_range(param_min, param_max, step, is_int)
        return satellite_params
    
    def _generate_drone_params(self, group: ParameterGroup, satellite_results: list) -> dict:
        """Satellite sonuclarindan cluster bulup Drone parametreleri uret."""
        if not satellite_results:
            return {}
        
        drone_params = {}
        for param_name, values in group.params.items():
            param_min, param_max = min(values), max(values)
            user_step = abs(values[1] - values[0]) if len(values) > 1 else 1
            sat_step = get_step(param_name, 'satellite', user_step=user_step)
            
            new_min, new_max = find_cluster_range(satellite_results, param_name, sat_step, param_min, param_max)
            drone_step = get_step(param_name, 'drone', user_step=user_step)
            is_int = param_name.endswith('period') or param_name.endswith('bars')
            drone_params[param_name] = generate_range(new_min, new_max, drone_step, is_int)
        
        return drone_params
    
    def run_stability_scoring(self, top_n: int = 3):
        """
        Her parametrenin komsularini test ederek stabilite skoru hesapla.
        Uc degerler elenir.
        """
        print("\n" + "="*60)
        print("  STABILITY SCORING")
        print("="*60)
        
        if self.on_progress:
            self.on_progress(82, "Stability Scoring...")
        
        for group_name, results in self.group_results.items():
            group = next((g for g in self.groups if g.name == group_name), None)
            if not group or not results:
                continue
            
            for result in results[:top_n]:
                stability = self._calculate_stability(result, group)
                result['stability'] = stability
                
                if stability >= 0.6:
                    print(f"  [OK] {group_name}: stability={stability:.0%}")
                else:
                    print(f"  [!] {group_name}: stability={stability:.0%} (edge value warning)")
    
    def _calculate_stability(self, result: dict, group: ParameterGroup) -> float:
        """Komsulari test ederek stabilite orani dondur."""
        original_fitness = result.get('fitness', 0)
        if original_fitness <= 0:
            return 0.0
        
        stable_neighbors = 0
        total_neighbors = 0
        
        # Sabit parametreler (diger gruplardan)
        base_params = result.copy()
        
        for param_name, values in group.params.items():
            if param_name not in result or len(values) < 2:
                continue
            
            current_val = result[param_name]
            step = abs(values[1] - values[0])
            
            for offset in [-step, step]:
                neighbor_val = current_val + offset
                if neighbor_val < min(values) or neighbor_val > max(values):
                    continue
                
                total_neighbors += 1
                
                # Komsuyu test et
                test_params = base_params.copy()
                test_params[param_name] = neighbor_val
                
                try:
                    score = _evaluate_params_static(test_params, self.strategy_index, self.commission, self.slippage)
                    neighbor_fitness = score.get('fitness', 0)
                    
                    # %15 icinde mi?
                    if abs(neighbor_fitness - original_fitness) / abs(original_fitness) < 0.15:
                        stable_neighbors += 1
                except:
                    pass
        
        return stable_neighbors / max(total_neighbors, 1)

    def run_combination_phase(self):
        msg = "PHASE 2: GRUP KOMBINASYONLARI"
        print(f"\n{msg}")
        if self.on_progress: self.on_progress(85, msg)
        
        top_per_group = {n: r[:3] for n, r in self.group_results.items() if r}
        if not top_per_group: return
        
        # Kombinasyonlari olustur
        all_combos = list(product(*top_per_group.values()))
        total_combos = len(all_combos)
        print(f"  Toplam kombinasyon: {total_combos}")
        
        # Paralel islem icin task listesi olustur
        tasks = []
        for combo in all_combos:
            merged = {}
            for res in combo: 
                merged.update({k: v for k, v in res.items() if k not in ['group', 'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 'stability']})
            # Cascaded gruplarin default degerlerini ekle
            for g in self.cascaded_groups:
                merged.update(g.default_values)
            tasks.append((merged, self.strategy_index, self.commission, self.slippage))
        
        # Paralel calistir
        print(f"  {cpu_count()} cekirdek ile paralel calistiriliyor...")
        try:
            self.pool = Pool(processes=cpu_count())
            results = self.pool.starmap(_evaluate_params_static, tasks)
            self.pool.close()
            self.pool.join()
            self.pool = None
        except Exception as e:
            print(f"Hybrid Combination Pool Error: {e}")
            if self.pool:
                self.pool.terminate()
                self.pool = None
            return
        
        # Sonuclari filtrele ve ekle
        for merged_params, score in zip([t[0] for t in tasks], results):
            if score.get('net_profit', 0) > 0:
                self.combined_results.append({**merged_params, **score})
        
        self.combined_results.sort(key=lambda x: x.get('fitness', x.get('net_profit', 0)), reverse=True)
        print(f"  Toplam basarili kombinasyon: {len(self.combined_results)}")


    def run_cascaded_phase(self):
        msg = "PHASE 3: KADEMELI OPTIMIZASYON"
        print(f"\n{msg}")
        if self.on_progress: self.on_progress(92, msg)
        
        if not self.combined_results: return
        best_base = {k: v for k, v in self.combined_results[0].items() if k not in ['net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 'stability']}
        for group in self.cascaded_groups:
            results = self.run_group_optimization(group, fixed_params=best_base)
            if results: [best_base.update({k: v}) for k, v in results[0].items() if k not in ['group', 'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 'stability']]
        score = _evaluate_params_static(best_base, self.strategy_index, self.commission, self.slippage)
        self.final_results = [{**best_base, **score}]

    def run(self, turbo: bool = True, iterative: bool = True, max_rounds: int = 3):
        """
        Optimizasyonu calistir.
        
        Args:
            turbo: Satellite-Drone kullan (her zaman True)
            iterative: Iterative Coordinate Descent kullan (yeni!)
            max_rounds: Iteratif mod icin max round sayisi
        """
        if iterative:
            self.run_iterative_phase(max_rounds)
            self.run_stability_scoring()
        elif turbo:
            self.run_satellite_drone_phase()
        else:
            self.run_independent_phase()
        
        self.run_combination_phase()
        self.run_cascaded_phase()
        
        if self.on_progress:
            self.on_progress(100, "Tamamlandi!")
        
        return self.final_results

    def get_best_results(self, top_n=20):
        return sorted(self.final_results or self.combined_results or [r for res in self.group_results.values() for r in res], key=lambda x: x.get('fitness', x.get('net_profit', 0)), reverse=True)[:top_n]

