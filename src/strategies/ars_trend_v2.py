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

from src.indicators.core import EMA, ATR, RSI, Momentum, HHV, LLV, ARS_Dynamic, MoneyFlowIndex

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
    
    # MFI Breakout Parametreleri (Grup 3 - Yeni)
    mfi_period: int = 14
    mfi_hhv_period: int = 14  # MFI üst breakout
    mfi_llv_period: int = 14  # MFI alt breakout
    mfi_breakout_enabled: bool = True
    
    # Hacim Breakout Parametreleri (Grup 3 - Yeni)
    volume_hhv_period: int = 14  # Hacim üst breakout
    volume_llv_period: int = 14  # Hacim alt breakout
    volume_breakout_enabled: bool = True
    
    # Çıkış Parametreleri
    kar_al_pct: float = 3.0
    iz_stop_pct: float = 1.5
    
    # Vade Yönetimi
    vade_tipi: str = "ENDEKS" # "ENDEKS" veya "SPOT"
    
    def get_max_period(self) -> int:
        """En uzun indikatör periyodunu hesapla - Isınma periyodu için"""
        periods = [
            self.ars_ema_period,
            self.ars_atr_period,
            self.momentum_period,
            self.breakout_period,
            self.rsi_period,
            self.mfi_period,
            self.mfi_hhv_period,
            self.mfi_llv_period,
            self.volume_hhv_period,
            self.volume_llv_period,
        ]
        return max(periods) + 10  # +10 güvenlik marjı

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
                 volumes: Optional[List[float]] = None,  # Lot/Hacim verisi
                 config: Optional[StrategyConfigV2] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
                 
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.times = times
        self.volumes = volumes or [0.0] * self.n  # Varsayılan 0
        
        self.config = config or StrategyConfigV2()
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # İndikatörleri hesapla
        self._calculate_indicators()
        
        # Vade sonu günlerini hesapla (Eğer times verildiyse)
        self.vade_sonu_gunleri = self._calculate_vade_sonlari() if times else set()
        
        # Vade geçişi barlarını tespit et (GAP/Isınma için)
        self.vade_gecis_barlari = self._detect_vade_transitions() if times else set()
        self.warmup_period = self.config.get_max_period()
        
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
        
        # 4. MFI Breakout (Grup 3 - Yeni)
        if cfg.mfi_breakout_enabled:
            self.mfi = MoneyFlowIndex(self.highs, self.lows, self.closes, 
                                       self.volumes, cfg.mfi_period)
            self.mfi_hhv = HHV(self.mfi, cfg.mfi_hhv_period)
            self.mfi_llv = LLV(self.mfi, cfg.mfi_llv_period)
        else:
            self.mfi = [50.0] * self.n
            self.mfi_hhv = [50.0] * self.n
            self.mfi_llv = [50.0] * self.n
        
        # 5. Hacim Breakout (Grup 3 - Yeni)
        if cfg.volume_breakout_enabled:
            self.volume_hhv = HHV(self.volumes, cfg.volume_hhv_period)
            self.volume_llv = LLV(self.volumes, cfg.volume_llv_period)
        else:
            self.volume_hhv = [0.0] * self.n
            self.volume_llv = [0.0] * self.n
        
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
    
    def _detect_vade_transitions(self) -> set:
        """Vade geçişi barlarını tespit et (Isınma periyodu için)
        
        Yeni vade başladığında (vade sonu günü sonraki ilk işlem günü),
        en uzun indikatör periyodu kadar sinyal üretilmemeli.
        
        Returns:
            set: Vade geçişi olan bar indeksleri
        """
        transition_bars = set()
        
        if not self.times or not self.vade_sonu_gunleri:
            return transition_bars
        
        prev_date = None
        for i, t in enumerate(self.times):
            current_date = t.date()
            
            if prev_date is not None:
                # Günün ilk barı mı?
                if current_date != prev_date:
                    # Önceki gün vade sonu mu?
                    if prev_date in self.vade_sonu_gunleri:
                        transition_bars.add(i)
            
            prev_date = current_date
        
        return transition_bars
    
    def _is_in_warmup(self, i: int) -> bool:
        """Bar ısınma periyodunda mı kontrol et"""
        for vade_bar in self.vade_gecis_barlari:
            if vade_bar <= i < vade_bar + self.warmup_period:
                return True
        return False

    def get_signal(self, i: int, current_position: str, 
                   entry_price: float = 0, 
                   extreme_price: float = 0) -> Signal:
        
        if i < 50: return Signal.NONE
        
        cfg = self.config
        
        # --- ISINMA PERİYODU KONTROLÜ (Vade Geçişi) ---
        if self._is_in_warmup(i):
            # Isınma periyodundayken sadece çıkış sinyali ver, giriş yok
            if current_position != "FLAT":
                return Signal.FLAT  # Pozisyonu kapat
            return Signal.NONE  # Yeni giriş yok
        
        # --- VADE SONU KONTROLÜ ---
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
                
                # MFI Breakout (Yeni) - MFI yeni zirve yapıyor
                mfi_onay = True
                if cfg.mfi_breakout_enabled:
                    mfi_onay = self.mfi[i] >= self.mfi_hhv[i-1]
                
                # Hacim Breakout (Yeni) - Hacim ortalamanın üstünde
                volume_onay = True
                if cfg.volume_breakout_enabled:
                    volume_onay = self.volumes[i] >= self.volume_hhv[i-1] * 0.8  # %80 eşik
                
                if yeni_zirve and pozitif_mom and rsi_uygun and mfi_onay and volume_onay:
                    return Signal.LONG
            
            # SHORT GİRİŞ
            elif self.trend_yonu[i] == -1:
                yeni_dip = self.lows[i] <= self.llv[i-1] and self.llv[i] < self.llv[i-1]
                negatif_mom = self.momentum[i] < 100
                rsi_uygun = self.rsi[i] > cfg.rsi_oversold
                
                # MFI Breakout (Yeni) - MFI yeni dip yapıyor
                mfi_onay = True
                if cfg.mfi_breakout_enabled:
                    mfi_onay = self.mfi[i] <= self.mfi_llv[i-1]
                
                # Hacim Breakout (Yeni) - Hacim ortalamanın üstünde
                volume_onay = True
                if cfg.volume_breakout_enabled:
                    volume_onay = self.volumes[i] >= self.volume_hhv[i-1] * 0.8  # %80 eşik
                
                if yeni_dip and negatif_mom and rsi_uygun and mfi_onay and volume_onay:
                    return Signal.SHORT
                    
        return Signal.NONE
