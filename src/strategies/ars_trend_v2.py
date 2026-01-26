# -*- coding: utf-8 -*-
"""
IdealQuant - ARS Trend Takip Stratejisi v2.0
IdealData ARS_Trend_v2 stratejisinin Python portu (1DK odaklı)
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

from indicators.core import EMA, ATR, RSI, Momentum, HHV, LLV, ARS_Dynamic

class Signal(str, Enum):
    LONG = "A"
    SHORT = "S"
    FLAT = "F"
    NONE = ""

@dataclass
class StrategyConfigV2:
    """ARS Trend v2 Strateji Konfigürasyonu (1DK Varsayılan)"""
    # ARS Parametreleri
    ars_ema_period: int = 3
    ars_atr_period: int = 10
    ars_atr_mult: float = 0.5
    ars_min_band: float = 0.002
    ars_max_band: float = 0.015
    
    # Giriş Sinyali Parametreleri
    momentum_period: int = 5
    breakout_period: int = 10
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Çıkış Parametreleri
    kar_al_pct: float = 3.0
    iz_stop_pct: float = 1.5
    
    # Vade Yönetimi
    vade_tipi: str = "ENDEKS" # "ENDEKS" veya "SPOT"

class ARSTrendStrategyV2:
    """
    ARS Trend Takip Stratejisi v2.0
    
    Özellikler:
    - Dinamik ARS Bandı (ATR bazlı)
    - Trend Takibi + Breakout + Momentum
    - Kar Al ve İzleyen Stop
    - Vade Sonu Kapanışları (Opsiyonel)
    """
    
    def __init__(self, 
                 opens: List[float],
                 highs: List[float],
                 lows: List[float],
                 closes: List[float],
                 typical: List[float],
                 times: List[datetime],
                 config: Optional[StrategyConfigV2] = None):
                 
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.times = times
        self.config = config or StrategyConfigV2()
        
        # İndikatörleri hesapla
        self._calculate_indicators()
        
        # Vade sonu günlerini hesapla (Eğer times verildiyse)
        self.vade_sonu_gunleri = self._calculate_vade_sonlari() if times else set()
        
    def _calculate_indicators(self):
        cfg = self.config
        
        # 1. ARS (Dinamik)
        self.ars = ARS_Dynamic(
            self.typical, self.highs, self.lows, self.closes,
            ema_period=cfg.ars_ema_period,
            atr_period=cfg.ars_atr_period,
            atr_mult=cfg.ars_atr_mult,
            min_k=cfg.ars_min_band,
            max_k=cfg.ars_max_band
        )
        
        # 2. Trend Yönü
        self.trend_yonu = [0] * self.n
        for i in range(1, self.n):
            if self.closes[i] > self.ars[i]:
                self.trend_yonu[i] = 1
            elif self.closes[i] < self.ars[i]:
                self.trend_yonu[i] = -1
            else:
                self.trend_yonu[i] = self.trend_yonu[i-1]
                
        # 3. Giriş İndikatörleri
        self.momentum = Momentum(self.closes, cfg.momentum_period)
        self.hhv = HHV(self.highs, cfg.breakout_period)
        self.llv = LLV(self.lows, cfg.breakout_period)
        self.rsi = RSI(self.closes, cfg.rsi_period)
        
    def _calculate_vade_sonlari(self) -> set:
        """Vade sonu tarihlerini hesapla (Basitleştirilmiş)"""
        vade_dates = set()
        # Veri setindeki aylar
        dates = pd.to_datetime(self.times)
        months = dates.to_period('M').unique()
        
        for m in months:
            # Sadece çift aylar (Endeks Vadelisi) - SPOT ise her ay
            if self.config.vade_tipi == "ENDEKS" and m.month % 2 != 0:
                continue
                
            # Ayın son günü
            last_day = m.to_timestamp(how='end').date()
            
            # İş günü kontrolü (Basitçe hafta sonu kontrolü, tatil listesi eklenebilir)
            while last_day.weekday() >= 5: # 5=Sat, 6=Sun
                last_day -= timedelta(days=1)
            
            vade_dates.add(last_day)
            
        return vade_dates

    def get_signal(self, i: int, current_position: str, 
                   entry_price: float = 0, 
                   extreme_price: float = 0) -> Signal:
        
        if i < 50: return Signal.NONE
        
        cfg = self.config
        
        # --- VADE SONU KONTROLÜ ---
        # (Şimdilik pas geçilebilir veya basit kontrol eklenebilir)
        current_time = self.times[i]
        is_vade_sonu = current_time.date() in self.vade_sonu_gunleri
        
        if is_vade_sonu and current_time.time() >= time(17, 40):
            if current_position != "FLAT":
                return Signal.FLAT
        
        # --- ÇIKIŞ MANTIĞI ---
        if current_position == "LONG":
            # 1. Trend Tersine Dönüş
            if self.trend_yonu[i] == -1 and self.trend_yonu[i-1] == 1:
                return Signal.FLAT
            
            # 2. Kar Al (%3.0)
            target_price = entry_price * (1 + cfg.kar_al_pct / 100.0)
            if self.closes[i] >= target_price:
                return Signal.FLAT
                
            # 3. İzleyen Stop (%1.5)
            # extreme_price LONG için işlemin gördüğü en yüksek fiyat olmalı
            # (Backtest döngüsünde güncellenmeli)
            trailing_stop_price = extreme_price * (1 - cfg.iz_stop_pct / 100.0)
            if self.closes[i] < trailing_stop_price:
                return Signal.FLAT
                
        elif current_position == "SHORT":
            # 1. Trend Tersine Dönüş
            if self.trend_yonu[i] == 1 and self.trend_yonu[i-1] == -1:
                return Signal.FLAT
                
            # 2. Kar Al (%3.0)
            target_price = entry_price * (1 - cfg.kar_al_pct / 100.0)
            if self.closes[i] <= target_price:
                return Signal.FLAT
                
            # 3. İzleyen Stop (%1.5)
            # extreme_price SHORT için işlemin gördüğü en düşük fiyat olmalı
            trailing_stop_price = extreme_price * (1 + cfg.iz_stop_pct / 100.0)
            if self.closes[i] > trailing_stop_price:
                return Signal.FLAT
        
        # --- GİRİŞ MANTIĞI ---
        if current_position == "FLAT": # Sadece FLAT iken giriş ara
            
            # LONG GİRİŞ
            if self.trend_yonu[i] == 1:
                yeni_zirve = self.highs[i] >= self.hhv[i-1] and self.hhv[i] > self.hhv[i-1]
                pozitif_mom = self.momentum[i] > 100
                rsi_uygun = self.rsi[i] < cfg.rsi_overbought
                
                if yeni_zirve and pozitif_mom and rsi_uygun:
                    return Signal.LONG
            
            # SHORT GİRİŞ
            elif self.trend_yonu[i] == -1:
                yeni_dip = self.lows[i] <= self.llv[i-1] and self.llv[i] < self.llv[i-1]
                negatif_mom = self.momentum[i] < 100
                rsi_uygun = self.rsi[i] > cfg.rsi_oversold
                
                if yeni_dip and negatif_mom and rsi_uygun:
                    return Signal.SHORT
                    
        return Signal.NONE
