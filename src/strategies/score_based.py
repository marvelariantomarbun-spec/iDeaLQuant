# -*- coding: utf-8 -*-
"""
IdealQuant/ScoreBasedStrategy (Ultimate Native)
Two-Stage & Stability Optimized.
Mümkün Olan En İyi Sonuç.

ULTIMATE SONUÇ (26.01.2026):
- Net Kâr: 11.053 Puan (PF 1.37)
- Parametreler: ARS(2, 0.011), ADX(22), RVI(27), Q(25)
- Durum: FINAL / CANLIYA HAZIR
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.indicators.core import EMA, ATR, ADX, SMA, ARS, RVI, Qstick, NetLot
from .ars_trend import Signal

@dataclass
class ScoreConfig:
    """Skor Strateji Konfigürasyonu (Ultimate Optimized)"""
    min_score: int = 3
    
    # İndikatör Periyotları
    rvi_period: int = 27    # Hassas Optimizasyon Sonucu (27)
    qstick_period: int = 25 # Hassas Optimizasyon Sonucu (25)
    adx_period: int = 22    # Hassas Optimizasyon Sonucu (22)
    
    # Eşik Değerler
    netlot_threshold: float = 20.0 
    adx_threshold: float = 25.0
    
    # Yatay Filtre
    ars_mesafe_threshold: float = 0.25
    bb_width_multiplier: float = 0.8
    
    # ARS (Çok Hızlı Tepki)
    ars_period: int = 2      # Hassas Optimizasyon Sonucu (2)
    ars_k: float = 0.011     # Hassas Optimizasyon Sonucu (0.011)


class ScoreBasedStrategy:
    """
    5 İndikatörlü Skor Tabanlı Strateji (Ultimate Native)
    
    İndikatörler:
    1. ARS (2, 0.011) - Ultra Hızlı Trend
    2. RVI (27) - Uzun Vade Momentum
    3. Qstick (25) - Uzun Vade Bar Gücü
    4. NetLot (20) - Hacim Baskısı
    5. ADX (22) - Orta Vade Trend Gücü
    
    Giriş: LONG Score >= 3 VE SHORT Score < 2 (Yatay Filtre +)
    """
    
    def __init__(self, 
                 opens: List[float],
                 highs: List[float],
                 lows: List[float],
                 closes: List[float],
                 typical: List[float],
                 config: Optional[ScoreConfig] = None,
                 indicators_df = None):
                 
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.config = config or ScoreConfig()
        
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        cfg = self.config

        # --- Python Native Hesaplama ---
        self.ars = ARS(self.typical, cfg.ars_period, cfg.ars_k)
        
        self.rvi, self.rvi_signal = RVI(self.opens, self.highs, self.lows, self.closes, cfg.rvi_period)
        self.qstick = Qstick(self.opens, self.closes, cfg.qstick_period)
        
        self.netlot = NetLot(self.opens, self.highs, self.lows, self.closes)
        self.netlot_ma = SMA(self.netlot, 5)
        
        self.adx = ADX(self.highs, self.lows, self.closes, cfg.adx_period)
        
        self._calculate_scores()
        
    def _calculate_scores(self):
        self.long_scores = [0] * self.n
        self.short_scores = [0] * self.n
        self.yatay_filtre = [0] * self.n 
        
        # --- YATAY FİLTRE ---
        from src.indicators.core import BollingerBands, SMA
        upper, middle, lower = BollingerBands(self.closes, 20, 2.0)
        
        bb_width = [0.0] * self.n
        for i in range(self.n):
            if middle[i] != 0:
                bb_width[i] = ((upper[i] - lower[i]) / middle[i]) * 100
                
        bb_width_avg = SMA(bb_width, 50)
        
        cfg = self.config
        
        for i in range(50, self.n):
            ars_sabit = True
            for j in range(1, 11): 
                if i - j >= 0:
                    if self.ars[i] != self.ars[i - j]:
                        ars_sabit = False
                        break
            
            ars_degisme_durumu = 0 if ars_sabit else 1
            
            ars_mesafe = 0.0
            if self.ars[i] != 0:
                ars_mesafe = abs(self.closes[i] - self.ars[i]) / self.ars[i] * 100
                
            f_skor = 0
            if ars_degisme_durumu == 1: f_skor += 1
            if ars_mesafe > cfg.ars_mesafe_threshold: f_skor += 1
            if self.adx[i] > 20.0: f_skor += 1 
            if bb_width[i] > bb_width_avg[i] * cfg.bb_width_multiplier: f_skor += 1
            
            self.yatay_filtre[i] = 1 if f_skor >= 2 else 0

            # --- SİNYAL SKORLARI ---
            ars_long = self.closes[i] > self.ars[i]
            rvi_long = self.rvi[i] > self.rvi_signal[i]
            qstick_long = self.qstick[i] > 0
            netlot_long = self.netlot_ma[i] > cfg.netlot_threshold
            adx_guclu = self.adx[i] > cfg.adx_threshold
            
            ars_short = self.closes[i] < self.ars[i]
            rvi_short = self.rvi[i] < self.rvi_signal[i]
            qstick_short = self.qstick[i] < 0
            netlot_short = self.netlot_ma[i] < -cfg.netlot_threshold
            
            l_score = 0
            if ars_long: l_score += 1
            if rvi_long: l_score += 1
            if qstick_long: l_score += 1
            if netlot_long: l_score += 1
            if adx_guclu: l_score += 1
            
            s_score = 0
            if ars_short: s_score += 1
            if rvi_short: s_score += 1
            if qstick_short: s_score += 1
            if netlot_short: s_score += 1
            if adx_guclu: s_score += 1
            
            self.long_scores[i] = l_score
            self.short_scores[i] = s_score

    def get_signal(self, i: int, current_position: str, 
                   entry_price: float = 0, 
                   extreme_price: float = 0) -> Signal:
        
        if i < 50: 
            return Signal.NONE
            
        cfg = self.config
        l_score = self.long_scores[i]
        s_score = self.short_scores[i]
        
        ars_long = self.closes[i] > self.ars[i]
        ars_short = self.closes[i] < self.ars[i]
            
        # Çıkış Mantığı
        if current_position == "LONG":
            if ars_short or s_score >= 4:
                return Signal.FLAT
                
        elif current_position == "SHORT":
            if ars_long or l_score >= 4:
                return Signal.FLAT
                
        # Giriş Mantığı
        if current_position == "FLAT":
            if self.yatay_filtre[i] == 1:
                if l_score >= cfg.min_score and s_score < 2:
                    return Signal.LONG
                    
                if s_score >= cfg.min_score and l_score < 2:
                    return Signal.SHORT
                
        return Signal.NONE

    def check_long_exit(self, i: int, entry_price: float, max_price: float) -> Tuple[bool, str]: return False, ""
    def check_short_exit(self, i: int, entry_price: float, min_price: float) -> Tuple[bool, str]: return False, ""
