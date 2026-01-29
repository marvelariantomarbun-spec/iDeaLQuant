# -*- coding: utf-8 -*-
"""
IdealQuant - Strateji 1 (Yatay Filtre + Skor Tabanlı Sinyal)
IdealData '1_Nolu_Strateji.txt' portu.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from src.indicators.core import (
    EMA, SMA, ADX, BollingerBands, QQEF, 
    RVI, Qstick, ARS_Dynamic
)
from .ars_trend import Signal

@dataclass
class Strategy1Config:
    # ARS
    ars_k: float = 0.0123
    ars_ema_period: int = 3
    
    # Yatay Filtre
    yatay_esik: int = 10
    ars_mesafe_threshold: float = 0.25
    adx_yatay_threshold: float = 20.0
    bb_multiplier: float = 0.8
    
    # İndikatörler
    qqef_period: int = 14
    qqef_smooth: int = 5
    rvi_period: int = 10
    qstick_period: int = 8
    adx_period: int = 14
    adx_esik: float = 25.0
    netlot_esik: float = 20.0
    
    # Skorlama
    min_onay_skoru: int = 5
    exit_skoru: int = 4

class Strategy1:
    def __init__(self, 
                 opens: List[float], 
                 highs: List[float], 
                 lows: List[float], 
                 closes: List[float], 
                 typical: List[float], 
                 times: Optional[List[any]] = None,
                 config: Optional[Strategy1Config] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        
        self.opens = np.array(opens, dtype=np.float64)
        self.highs = np.array(highs, dtype=np.float64)
        self.lows = np.array(lows, dtype=np.float64)
        self.closes = np.array(closes, dtype=np.float64)
        self.typical = np.array(typical, dtype=np.float64)
        self.times = times
        
        self.config = config or Strategy1Config()
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        self.n = len(closes)
        self._calculate_indicators()

    def _calculate_indicators(self):
        cfg = self.config
        
        # 1. ARS
        # IdealData portu: ARS_EMA = MA(Typical, Exp, 3)
        # ARS_Dynamic fonksiyonu core.py'da zaten var ve validasyonu yapıldı.
        ars_ema = EMA(self.typical.tolist(), cfg.ars_ema_period)
        # ARS_Dynamic(data, period, k) -> data olarak EMA kullanılıyor iDeal'de
        # Ancak core.py'daki ARS_Dynamic kendi içinde EMA hesaplıyor olabilir. 
        # iDeal koduna sadık kalalım: iDeal'de ARS Manuel loop ile hesaplanmış.
        self.ars = self._calculate_ars_manual(ars_ema, cfg.ars_k)
        
        # 2. Yatay Filtre Bileşenleri
        self.ars_degisme_durumu = self._calculate_ars_degisme(self.ars, cfg.yatay_esik)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.ars_mesafe = np.abs(self.closes - self.ars) / self.ars * 100
        self.ars_mesafe = np.nan_to_num(self.ars_mesafe)
        
        self.adx = ADX(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), cfg.adx_period)
        
        # Bollinger
        bb_up, bb_mid, bb_down = BollingerBands(self.closes.tolist(), 20, 2.0)
        self.bb_width = np.zeros(self.n)
        for i in range(self.n):
            if bb_mid[i] != 0:
                self.bb_width[i] = ((bb_up[i] - bb_down[i]) / bb_mid[i]) * 100
        self.bb_width_avg = SMA(self.bb_width.tolist(), 50)
        
        # 3. Diğer İndikatörler
        # NetLot (iDeal mantığı: (C-O)/(H-L) * 100)
        self.netlot = np.zeros(self.n)
        for i in range(self.n):
            diff = self.highs[i] - self.lows[i]
            if diff > 0:
                self.netlot[i] = (self.closes[i] - self.opens[i]) / diff * 100
        self.netlot_ma = SMA(self.netlot.tolist(), 5)
        
        self.qqef, self.qqes = QQEF(self.typical.tolist(), cfg.qqef_period, cfg.qqef_smooth)
        
        self.rvi, self.rvi_signal = RVI(self.opens.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), cfg.rvi_period)
        
        self.qstick = Qstick(self.opens.tolist(), self.closes.tolist(), cfg.qstick_period)
        
        # 4. Yatay Filtre Birleştirme
        self.yatay_filtre = np.zeros(self.n)
        for i in range(50, self.n):
            skor = 0
            if self.ars_degisme_durumu[i] == 1: skor += 1
            if self.ars_mesafe[i] > cfg.ars_mesafe_threshold: skor += 1
            if self.adx[i] > cfg.adx_yatay_threshold: skor += 1
            if self.bb_width[i] > self.bb_width_avg[i] * cfg.bb_multiplier: skor += 1
            self.yatay_filtre[i] = 1 if skor >= 2 else 0

    def _calculate_ars_manual(self, ars_ema, k):
        ars = np.zeros(self.n)
        for i in range(1, self.n):
            alt = ars_ema[i] * (1 - k)
            ust = ars_ema[i] * (1 + k)
            if alt > ars[i-1]:
                ars[i] = alt
            elif ust < ars[i-1]:
                ars[i] = ust
            else:
                ars[i] = ars[i-1]
        return ars

    def _calculate_ars_degisme(self, ars, esik):
        degisme = np.zeros(self.n)
        for i in range(esik, self.n):
            ars_ayni = True
            for j in range(1, esik + 1):
                if ars[i] != ars[i-j]:
                    ars_ayni = False
                    break
            degisme[i] = 0 if ars_ayni else 1
        return degisme

    def get_signal(self, i: int, current_position: str) -> Signal:
        if i < 50:
            return Signal.NONE
            
        cfg = self.config
        
        # Skorlama
        long_score = 0
        short_score = 0
        
        if self.closes[i] > self.ars[i]: long_score += 1
        elif self.closes[i] < self.ars[i]: short_score += 1
        
        if self.qqef[i] > self.qqes[i] and self.qqef[i] > 50: long_score += 1
        elif self.qqef[i] < self.qqes[i] and self.qqef[i] < 50: short_score += 1
        
        if self.rvi[i] > self.rvi_signal[i]: long_score += 1
        elif self.rvi[i] < self.rvi_signal[i]: short_score += 1
        
        if self.qstick[i] > 0: long_score += 1
        elif self.qstick[i] < 0: short_score += 1
        
        if self.netlot_ma[i] > cfg.netlot_esik: long_score += 1
        elif self.netlot_ma[i] < -cfg.netlot_esik: short_score += 1
        
        if self.adx[i] > cfg.adx_esik:
            long_score += 1
            short_score += 1
            
        # ÇIKIŞ MANTIĞI
        if current_position == "LONG":
            if self.closes[i] < self.ars[i] or short_score >= cfg.exit_skoru:
                return Signal.FLAT
        elif current_position == "SHORT":
            if self.closes[i] > self.ars[i] or long_score >= cfg.exit_skoru:
                return Signal.FLAT
                
        # GİRİŞ MANTIĞI
        if current_position == "FLAT":
            if self.yatay_filtre[i] == 1:
                if long_score >= cfg.min_onay_skoru and short_score < 2:
                    return Signal.LONG
                elif short_score >= cfg.min_onay_skoru and long_score < 2:
                    return Signal.SHORT
                    
        return Signal.NONE
