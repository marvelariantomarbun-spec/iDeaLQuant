# -*- coding: utf-8 -*-
"""
IdealQuant/ScoreBasedStrategy (Simplified Gateway)
Trend Takipçisi (Strateji 2) için Güvenli Giriş Kapısı.
İndikatörler: ARS + MACD-V + ADX + NetLot
(RVI ve QStick kaldırıldı)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time

from src.indicators.core import EMA, ATR, ADX, SMA, ARS, NetLot, MACDV
from .common import Signal
from .holidays import BAYRAM_TARIHLERI, RESMI_TATILLER, is_holiday_eve, vade_sonu_is_gunu

@dataclass
class ScoreConfig:
    """Skor Strateji Konfigürasyonu (Global Optimized v4.1 - 20 Parametre)"""
    # 1. Grup: Skor Ayarları (2)
    min_score: int = 3
    exit_score: int = 3
    
    # 2. Grup: ARS (2)
    ars_period: int = 3
    ars_k: float = 0.01
    
    # 3. Grup: ADX (2)
    adx_period: int = 17
    adx_threshold: float = 25.0
    
    # 4. Grup: MACD-V (4)
    macdv_short: int = 13
    macdv_long: int = 28
    macdv_signal: int = 8
    macdv_threshold: float = 0.0 # MACD-V çizgisinin 0 üstü mü?
    
    # 5. Grup: NetLot (2)
    netlot_period: int = 5  # SMA periyodu
    netlot_threshold: float = 20.0 
    
    # 6. Grup: Yatay Filtre (8)
    ars_mesafe_threshold: float = 0.25
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_multiplier: float = 0.8
    bb_avg_period: int = 50
    yatay_ars_bars: int = 10 # ARS'nin değişmediği bar sayısı
    yatay_adx_threshold: float = 20.0
    filter_score_threshold: int = 2 # Yatay filtrenin geçmesi için gereken puan
    
    # Vade Yönetimi
    vade_tipi: str = "ENDEKS"  # "ENDEKS" veya "SPOT"
     


class ScoreBasedStrategy:
    """
    4 İndikatörlü Skor Tabanlı Strateji (Simplified)
    Hedef: Güçlü trendleri tespit edip Strateji 2'ye yol vermek.
    
    İndikatörler:
    1. ARS (Trend Yönü)
    2. MACD-V (Volatilite Doğrulanmış Momentum)
    3. NetLot (Hacim Baskısı)
    4. ADX (Trend Gücü)
    
    Giriş: LONG Score >= min_score VE SHORT Score < 2 (Yatay Filtre +)
    """
    
    def __init__(self, 
                 opens: List[float],
                 highs: List[float],
                 lows: List[float],
                 closes: List[float],
                 typical: List[float],
                 config: Optional[ScoreConfig] = None,
                 indicators_df = None,
                 dates: Optional[List[datetime]] = None):
                 
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.config = config or ScoreConfig()
        self.dates = dates or []
        
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        cfg = self.config

        # --- Python Native Hesaplama ---
        self.ars = ARS(self.typical, cfg.ars_period, cfg.ars_k)
        
        self.netlot = NetLot(self.opens, self.highs, self.lows, self.closes)
        self.netlot_ma = SMA(self.netlot, cfg.netlot_period)
        
        self.adx = ADX(self.highs, self.lows, self.closes, cfg.adx_period)
        
        self.macdv, self.macdv_sig = MACDV(self.closes, self.highs, self.lows, 
                                           cfg.macdv_short, cfg.macdv_long, cfg.macdv_signal)
        
        self._calculate_scores()
        
    def _calculate_scores(self):
        self.long_scores = [0] * self.n
        self.short_scores = [0] * self.n
        self.yatay_filtre = [0] * self.n 
        
        # --- YATAY FİLTRE ---
        from src.indicators.core import BollingerBands, SMA
        cfg = self.config
        upper, middle, lower = BollingerBands(self.closes, cfg.bb_period, cfg.bb_std)
        
        bb_width = [0.0] * self.n
        for i in range(self.n):
            if middle[i] != 0:
                bb_width[i] = ((upper[i] - lower[i]) / middle[i]) * 100
                
        bb_width_avg = SMA(bb_width, cfg.bb_avg_period)
        
        for i in range(50, self.n):
            ars_sabit = True
            for j in range(1, cfg.yatay_ars_bars + 1): 
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
            if self.adx[i] > cfg.yatay_adx_threshold: f_skor += 1 
            if bb_width[i] > bb_width_avg[i] * cfg.bb_width_multiplier: f_skor += 1
            
            self.yatay_filtre[i] = 1 if f_skor >= cfg.filter_score_threshold else 0

            # --- SİNYAL SKORLARI ---
            ars_long = self.closes[i] > self.ars[i]
            netlot_long = self.netlot_ma[i] > cfg.netlot_threshold
            adx_guclu = self.adx[i] > cfg.adx_threshold
            macdv_long = self.macdv[i] > (self.macdv_sig[i] + cfg.macdv_threshold)
            
            ars_short = self.closes[i] < self.ars[i]
            netlot_short = self.netlot_ma[i] < -cfg.netlot_threshold
            macdv_short = self.macdv[i] < (self.macdv_sig[i] - cfg.macdv_threshold)
            
            l_score = 0
            if ars_long: l_score += 1
            if netlot_long: l_score += 1
            if adx_guclu: l_score += 1
            if macdv_long: l_score += 1
            
            s_score = 0
            if ars_short: s_score += 1
            if netlot_short: s_score += 1
            if adx_guclu: s_score += 1
            if macdv_short: s_score += 1
            
            self.long_scores[i] = l_score
            self.short_scores[i] = s_score

    def get_signal(self, i: int, current_position: str, 
                   entry_price: float = 0, 
                   extreme_price: float = 0,
                   return_flat_reason: bool = False) -> Signal:
        
        if i < 50: 
            return Signal.NONE
        
        # ===== VADE/TATİL YÖNETİMİ =====
        if self.dates and i < len(self.dates):
            dt = self.dates[i]
            t = dt.time()
            
            # Seans kontrolü (09:30-18:15, 19:00-23:00)
            gun_seansi = time(9, 30) <= t < time(18, 15)
            aksam_seansi = time(19, 0) <= t < time(23, 0)
            
            if not (gun_seansi or aksam_seansi):
                return Signal.NONE
            
            # Vade ayı kontrolü
            vade_ayi = (self.config.vade_tipi == "SPOT") or (dt.month % 2 == 0)
            vade_sonu_gun = vade_ayi and (dt.date() == vade_sonu_is_gunu(dt))
            
            # Arefe kontrolü
            arefe = is_holiday_eve(dt.date())
            
            # SENARYO 1: Arefe + Vade Sonu → 11:30 flat
            if arefe and vade_sonu_gun and t > time(11, 30):
                if current_position != "FLAT":
                    if return_flat_reason:
                        return (Signal.FLAT, "arefe_vade_sonu")
                    return Signal.FLAT
            
            # SENARYO 2: Sadece Arefe → 11:30 flat
            elif arefe and not vade_sonu_gun and t > time(11, 30):
                if current_position != "FLAT":
                    if return_flat_reason:
                        return (Signal.FLAT, "arefe")
                    return Signal.FLAT
            
            # SENARYO 3: Normal Vade Sonu → 17:40 flat
            elif vade_sonu_gun and t > time(17, 40):
                if current_position != "FLAT":
                    if return_flat_reason:
                        return (Signal.FLAT, "vade_sonu")
                    return Signal.FLAT
            
            # Arefe/Vade günlerinde flat saatlerinde işlem yapma
            if (arefe and t > time(11, 30)) or (vade_sonu_gun and not arefe and t > time(17, 40)):
                if return_flat_reason:
                    return (Signal.NONE, None)
                return Signal.NONE
            
        cfg = self.config
        l_score = self.long_scores[i]
        s_score = self.short_scores[i]
        
        ars_long = self.closes[i] > self.ars[i]
        ars_short = self.closes[i] < self.ars[i]
            
        # Çıkış Mantığı
        if current_position == "LONG":
            if ars_short or s_score >= cfg.exit_score:
                if return_flat_reason:
                    return (Signal.FLAT, None)
                return Signal.FLAT
                
        elif current_position == "SHORT":
            if ars_long or l_score >= cfg.exit_score:
                if return_flat_reason:
                    return (Signal.FLAT, None)
                return Signal.FLAT
                
        # Giriş Mantığı (Strateji 2'ye yol verme)
        if current_position == "FLAT":
            if self.yatay_filtre[i] == 1:
                if l_score >= cfg.min_score and s_score < 2:
                    if return_flat_reason:
                        return (Signal.LONG, None)
                    return Signal.LONG
                    
                if s_score >= cfg.min_score and l_score < 2:
                    if return_flat_reason:
                        return (Signal.SHORT, None)
                    return Signal.SHORT
        
        if return_flat_reason:
            return (Signal.NONE, None)
        return Signal.NONE

    def check_long_exit(self, i: int, entry_price: float, max_price: float) -> Tuple[bool, str]: return False, ""
    def check_short_exit(self, i: int, entry_price: float, min_price: float) -> Tuple[bool, str]: return False, ""
    
    @classmethod
    def from_config_dict(cls, data_cache, config_dict: dict, dates: Optional[List[datetime]] = None):
        """
        Optimizer için fabrika metodu - config dict'ten strateji oluşturur.
        
        Args:
            data_cache: Opens, highs, lows, closes, typical içeren cache objesi
            config_dict: Strateji parametreleri (ScoreConfig alanlarıyla eşleşmeli)
            dates: Opsiyonel tarih listesi (vade/tatil yönetimi için)
        """
        config = ScoreConfig(
            min_score=config_dict.get('min_score', 3),
            exit_score=config_dict.get('exit_score', 3),
            ars_period=config_dict.get('ars_period', 3),
            ars_k=config_dict.get('ars_k', 0.01),
            adx_period=config_dict.get('adx_period', 17),
            adx_threshold=config_dict.get('adx_threshold', 25.0),
            macdv_short=config_dict.get('macdv_short', 13),
            macdv_long=config_dict.get('macdv_long', 28),
            macdv_signal=config_dict.get('macdv_signal', 8),
            macdv_threshold=config_dict.get('macdv_threshold', 0.0),
            netlot_period=config_dict.get('netlot_period', 5),
            netlot_threshold=config_dict.get('netlot_threshold', 20.0),
            ars_mesafe_threshold=config_dict.get('ars_mesafe_threshold', 0.25),
            bb_period=config_dict.get('bb_period', 20),
            bb_std=config_dict.get('bb_std', 2.0),
            bb_width_multiplier=config_dict.get('bb_width_multiplier', 0.8),
            bb_avg_period=config_dict.get('bb_avg_period', 50),
            yatay_ars_bars=config_dict.get('yatay_ars_bars', 10),
            yatay_adx_threshold=config_dict.get('yatay_adx_threshold', 20.0),
            filter_score_threshold=config_dict.get('filter_score_threshold', 2),
            vade_tipi=config_dict.get('vade_tipi', 'ENDEKS'),
        )
        
        return cls(
            opens=list(data_cache.opens) if hasattr(data_cache, 'opens') else data_cache['opens'],
            highs=list(data_cache.highs) if hasattr(data_cache, 'highs') else data_cache['highs'],
            lows=list(data_cache.lows) if hasattr(data_cache, 'lows') else data_cache['lows'],
            closes=list(data_cache.closes) if hasattr(data_cache, 'closes') else data_cache['closes'],
            typical=list(data_cache.typical) if hasattr(data_cache, 'typical') else data_cache['typical'],
            config=config,
            dates=dates or (data_cache.dates if hasattr(data_cache, 'dates') else None),
        )
    
    def generate_all_signals(self) -> Tuple:
        """
        Tüm barlar için sinyal üret - Optimizer backtesti için.
        
        Returns:
            Tuple: (signals, exits_long, exits_short)
                - signals: np.array (1=LONG, -1=SHORT, 0=NONE)
                - exits_long: np.array (True/False)
                - exits_short: np.array (True/False)
        """
        import numpy as np
        
        n = self.n
        signals = np.zeros(n, dtype=int)
        exits_long = np.zeros(n, dtype=bool)
        exits_short = np.zeros(n, dtype=bool)
        
        position = "FLAT"
        
        for i in range(n):
            sig = self.get_signal(i, position)
            
            if sig == Signal.LONG:
                signals[i] = 1
                position = "LONG"
            elif sig == Signal.SHORT:
                signals[i] = -1
                position = "SHORT"
            elif sig == Signal.FLAT:
                if position == "LONG":
                    exits_long[i] = True
                elif position == "SHORT":
                    exits_short[i] = True
                position = "FLAT"
        
        return signals, exits_long, exits_short
