# -*- coding: utf-8 -*-
"""
IdealQuant - ARS Trend Takip Stratejisi
IdealData ARS_Trend_v2 stratejisinin Python portu
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from indicators.core import EMA, ATR, RSI, Momentum, HHV, LLV


class Signal(Enum):
    """Trading sinyalleri"""
    LONG = "A"      # Alış (Buy)
    SHORT = "S"     # Satış (Sell)  
    FLAT = "F"      # Pozisyon kapat
    NONE = ""       # Sinyal yok


@dataclass
class StrategyConfig:
    """ARS Trend strateji konfigürasyonu"""
    # ARS parametreleri
    ars_ema_period: int = 5
    ars_atr_period: int = 14
    ars_atr_mult: float = 0.7
    ars_min_band: float = 0.003
    ars_max_band: float = 0.020
    
    # Giriş sinyal parametreleri
    momentum_period: int = 8
    breakout_period: int = 15
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Çıkış parametreleri
    kar_al_pct: float = 3.0      # %
    iz_stop_pct: float = 1.5     # %
    
    @classmethod
    def for_timeframe(cls, minutes: int) -> 'StrategyConfig':
        """Timeframe'e göre otomatik parametre ayarla"""
        if minutes <= 1:
            return cls(
                ars_ema_period=3, ars_atr_period=10, ars_atr_mult=0.5,
                ars_min_band=0.002, ars_max_band=0.015,
                momentum_period=5, breakout_period=10
            )
        elif minutes <= 5:
            return cls(
                ars_ema_period=5, ars_atr_period=14, ars_atr_mult=0.7,
                ars_min_band=0.003, ars_max_band=0.020,
                momentum_period=8, breakout_period=15
            )
        elif minutes <= 15:
            return cls(
                ars_ema_period=3, ars_atr_period=14, ars_atr_mult=0.0,
                ars_min_band=0.0123, ars_max_band=0.0123,
                momentum_period=10, breakout_period=20
            )
        else:  # 60dk ve üstü
            return cls(
                ars_ema_period=3, ars_atr_period=14, ars_atr_mult=0.0,
                ars_min_band=0.0123, ars_max_band=0.0123,
                momentum_period=14, breakout_period=30
            )


class ARSTrendStrategy:
    """
    ARS Trend Takip Stratejisi
    
    Sinyal mantığı:
    - ARS > Fiyat: Ayı trendi (SHORT izin)
    - ARS < Fiyat: Boğa trendi (LONG izin)
    - Giriş: HHV/LLV kırılımı + Momentum + RSI filtresi
    - Çıkış: Trend tersine dönme, kar al veya izleyen stop
    """
    
    def __init__(self, 
                 opens: List[float],
                 highs: List[float],
                 lows: List[float],
                 closes: List[float],
                 typical: List[float],
                 config: Optional[StrategyConfig] = None):
        """
        Args:
            opens: Açılış fiyatları
            highs: Yüksek fiyatlar
            lows: Düşük fiyatlar
            closes: Kapanış fiyatları
            typical: Tipik fiyatlar
            config: Strateji konfigürasyonu
        """
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.config = config or StrategyConfig()
        
        # İndikatörleri hesapla
        self._calculate_indicators()
        
        # Trend yönünü hesapla
        self._calculate_trend()
    
    def _calculate_indicators(self):
        """Tüm indikatörleri hesapla"""
        cfg = self.config
        
        # ATR
        self.atr = ATR(self.highs, self.lows, self.closes, cfg.ars_atr_period)
        
        # ARS - Dinamik veya Classic mod
        ars_ema = EMA(self.typical, cfg.ars_ema_period)
        self.ars = [0.0] * self.n
        
        for i in range(1, self.n):
            if cfg.ars_atr_mult > 0:
                # Dinamik mod
                dinamik_k = (self.atr[i] / ars_ema[i]) * cfg.ars_atr_mult if ars_ema[i] != 0 else cfg.ars_min_band
                dinamik_k = max(cfg.ars_min_band, min(cfg.ars_max_band, dinamik_k))
            else:
                # Classic mod
                dinamik_k = cfg.ars_min_band
            
            alt_band = ars_ema[i] * (1 - dinamik_k)
            ust_band = ars_ema[i] * (1 + dinamik_k)
            
            if alt_band > self.ars[i - 1]:
                self.ars[i] = alt_band
            elif ust_band < self.ars[i - 1]:
                self.ars[i] = ust_band
            else:
                self.ars[i] = self.ars[i - 1]
        
        # Diğer indikatörler
        self.momentum = Momentum(self.closes, cfg.momentum_period)
        self.hhv = HHV(self.highs, cfg.breakout_period)
        self.llv = LLV(self.lows, cfg.breakout_period)
        self.rsi = RSI(self.closes, cfg.rsi_period)
    
    def _calculate_trend(self):
        """ARS bazlı trend yönünü hesapla"""
        self.trend = [0] * self.n  # 1=Boğa, -1=Ayı
        
        for i in range(1, self.n):
            if self.closes[i] > self.ars[i]:
                self.trend[i] = 1  # Boğa
            elif self.closes[i] < self.ars[i]:
                self.trend[i] = -1  # Ayı
            else:
                self.trend[i] = self.trend[i - 1]
    
    def check_long_entry(self, i: int) -> bool:
        """
        LONG giriş koşullarını kontrol et
        1. ARS Boğa trendinde
        2. Yeni zirve kırılımı (HHV breakout)
        3. Pozitif momentum
        4. RSI aşırı alım değil (<70)
        """
        if i < 2:
            return False
        
        cfg = self.config
        
        # Koşul 1: Boğa trendi
        if self.trend[i] != 1:
            return False
        
        # Koşul 2: Yeni zirve kırılımı
        yeni_zirve = (self.highs[i] >= self.hhv[i - 1] and 
                      self.hhv[i] > self.hhv[i - 1])
        if not yeni_zirve:
            return False
        
        # Koşul 3: Pozitif momentum (>100 = %0 üstü)
        if self.momentum[i] <= 100:
            return False
        
        # Koşul 4: RSI < 70
        if self.rsi[i] >= cfg.rsi_overbought:
            return False
        
        return True
    
    def check_short_entry(self, i: int) -> bool:
        """
        SHORT giriş koşullarını kontrol et
        1. ARS Ayı trendinde
        2. Yeni dip kırılımı (LLV breakout)
        3. Negatif momentum
        4. RSI aşırı satım değil (>30)
        """
        if i < 2:
            return False
        
        cfg = self.config
        
        # Koşul 1: Ayı trendi
        if self.trend[i] != -1:
            return False
        
        # Koşul 2: Yeni dip kırılımı
        yeni_dip = (self.lows[i] <= self.llv[i - 1] and 
                    self.llv[i] < self.llv[i - 1])
        if not yeni_dip:
            return False
        
        # Koşul 3: Negatif momentum (<100 = %0 altı)
        if self.momentum[i] >= 100:
            return False
        
        # Koşul 4: RSI > 30
        if self.rsi[i] <= cfg.rsi_oversold:
            return False
        
        return True
    
    def check_long_exit(self, i: int, entry_price: float, max_price: float) -> Tuple[bool, str]:
        """
        LONG pozisyondan çıkış kontrolü
        Returns: (çıkış_var_mı, sebep)
        """
        cfg = self.config
        price = self.closes[i]
        
        # 1. Trend tersine döndü
        if self.trend[i] == -1 and self.trend[i - 1] == 1:
            return True, "trend_reversal"
        
        # 2. Kar al
        target = entry_price * (1 + cfg.kar_al_pct / 100)
        if price >= target:
            return True, "take_profit"
        
        # 3. İzleyen stop
        trailing_stop = max_price * (1 - cfg.iz_stop_pct / 100)
        if price < trailing_stop:
            return True, "trailing_stop"
        
        return False, ""
    
    def check_short_exit(self, i: int, entry_price: float, min_price: float) -> Tuple[bool, str]:
        """
        SHORT pozisyondan çıkış kontrolü
        Returns: (çıkış_var_mı, sebep)
        """
        cfg = self.config
        price = self.closes[i]
        
        # 1. Trend tersine döndü
        if self.trend[i] == 1 and self.trend[i - 1] == -1:
            return True, "trend_reversal"
        
        # 2. Kar al
        target = entry_price * (1 - cfg.kar_al_pct / 100)
        if price <= target:
            return True, "take_profit"
        
        # 3. İzleyen stop
        trailing_stop = min_price * (1 + cfg.iz_stop_pct / 100)
        if price > trailing_stop:
            return True, "trailing_stop"
        
        return False, ""
    
    def get_signal(self, i: int, current_position: str, 
                   entry_price: float = 0, 
                   extreme_price: float = 0) -> Signal:
        """
        Belirtilen bar için sinyal üret
        
        Args:
            i: Bar index
            current_position: "LONG", "SHORT" veya "FLAT"
            entry_price: Giriş fiyatı (pozisyondaysa)
            extreme_price: Max fiyat (LONG) veya min fiyat (SHORT)
        
        Returns:
            Signal enum
        """
        if i < 2:
            return Signal.NONE
        
        # Çıkış kontrolü
        if current_position == "LONG":
            should_exit, _ = self.check_long_exit(i, entry_price, extreme_price)
            if should_exit:
                return Signal.FLAT
        
        elif current_position == "SHORT":
            should_exit, _ = self.check_short_exit(i, entry_price, extreme_price)
            if should_exit:
                return Signal.FLAT
        
        # Giriş kontrolü (sadece FLAT pozisyondaysa)
        if current_position == "FLAT":
            if self.check_long_entry(i):
                return Signal.LONG
            
            if self.check_short_entry(i):
                return Signal.SHORT
        
        return Signal.NONE
    
    def get_ars(self) -> List[float]:
        """ARS değerlerini döndür"""
        return self.ars
    
    def get_trend(self) -> List[int]:
        """Trend değerlerini döndür"""
        return self.trend
