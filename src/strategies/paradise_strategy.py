# -*- coding: utf-8 -*-
"""
IdealQuant - Paradise Stratejisi (Strateji 3)
HH/LL Breakout + Momentum + EMA/DSMA Trend filtresi

Referans: reference/Paradise.txt (IdealData C# kodu)
"""

from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
import numpy as np
import pandas as pd

from src.indicators.core import EMA, SMA, ATR, Momentum, HHV, LLV
from .common import Signal
from .holidays import (
    BAYRAM_TARIHLERI, RESMI_TATILLER,
    is_bayram_tatili, is_arefe, is_resmi_tatil, is_tatil_gunu,
    vade_sonu_is_gunu, is_seans_icinde
)


@dataclass
class ParadiseConfig:
    """Paradise Strateji Konfigurasyonu (11 parametre + 2 mod)"""
    # 1. Grup: Trend (3)
    ema_period: int = 21
    dsma_period: int = 50     # Double SMA: SMA(SMA(C, N), N)
    ma_period: int = 20
    
    # 2. Grup: Breakout (2)
    hh_period: int = 25       # HH/LL Periyodu
    vol_hhv_period: int = 14  # Hacim HHV Periyodu
    
    # 3. Grup: Momentum (3)
    mom_period: int = 60      # Momentum Periyodu
    mom_alt: float = 98.0     # Momentum alt bant (100 - delta)
    mom_ust: float = 102.0    # Momentum ust bant (100 + delta)
    
    # 4. Grup: Risk / Cikis (3)
    atr_period: int = 14      # ATR Periyodu
    atr_sl: float = 2.0       # Stop Loss carpani
    atr_tp: float = 4.0       # Take Profit carpani
    atr_trail: float = 2.5    # Trailing Stop carpani
    
    # Mod Parametreleri (sabit, optimize edilmez)
    vade_tipi: str = "ENDEKS"      # "ENDEKS" (cift ay) veya "SPOT" (her ay)
    yon_modu: str = "CIFT"         # "CIFT" (AL+SAT) veya "SADECE_AL" (Long-only)
    
    def get_max_period(self) -> int:
        """En uzun indikator periyodunu hesapla - Isinma periyodu icin"""
        periods = [
            self.ema_period,
            self.dsma_period * 2,  # DSMA icin 2x gerekli
            self.ma_period,
            self.hh_period,
            self.vol_hhv_period,
            self.mom_period,
            self.atr_period,
        ]
        return max(periods) + 10  # +10 guvenlik marji


class ParadiseStrategy:
    """
    Paradise Stratejisi (Strateji 3)
    
    Giris: HH/LL Breakout + EMA vs DSMA trend + Close vs MA + Momentum banti + Hacim
    Cikis: ATR-bazli SL / TP / Trailing Stop
    
    Referans: reference/Paradise.txt
    """
    
    def __init__(self,
                 opens: List[float],
                 highs: List[float],
                 lows: List[float],
                 closes: List[float],
                 typical: List[float],
                 times: List[datetime],
                 volumes: Optional[List[float]] = None,
                 config: Optional[ParadiseConfig] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.times = times
        self.volumes = volumes if volumes is not None else [1.0] * self.n
        
        self.config = config or ParadiseConfig()
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Indikatorleri hesapla
        self._calculate_indicators()
        
        # Vade sonu gunlerini hesapla
        self.vade_sonu_gunleri = self._calculate_vade_sonlari() if times else set()
        
        # Vade gecisi barlarini tespit et
        self.vade_gecis_barlari = self._detect_vade_transitions() if times else set()
        self.warmup_period = self.config.get_max_period()
        
        # Warmup State
        self.vade_cooldown_bar = self.warmup_period
        self.warmup_bars = max(50, self.vade_cooldown_bar)
        self.warmup_aktif = False
        self.warmup_baslangic_bar = -999
        self.arefe_flat = False
        
    def _calculate_indicators(self):
        cfg = self.config
        
        # 1. Trend Indikatorleri
        self.ema = EMA(self.closes, cfg.ema_period)
        dsma_inner = SMA(self.closes, cfg.dsma_period)  # Ilk SMA
        self.dsma = SMA(dsma_inner, cfg.dsma_period)     # Ikinci SMA (Double SMA)
        self.ma = SMA(self.closes, cfg.ma_period)
        
        # 2. Momentum
        self.momentum = Momentum(self.closes, cfg.mom_period)
        
        # 3. HH / LL (Highest High / Lowest Low) 
        self.hh = HHV(self.highs, cfg.hh_period)
        self.ll = LLV(self.lows, cfg.hh_period)
        
        # 4. ATR (cikis icin)
        self.atr = ATR(self.highs, self.lows, self.closes, cfg.atr_period)
        
        # 5. Hacim HHV
        self.vol_hhv = HHV(self.volumes, cfg.vol_hhv_period)
    
    def _calculate_vade_sonlari(self) -> Set[date]:
        """Vade sonu tarihlerini hesapla"""
        vade_dates = set()
        dates = pd.to_datetime(self.times)
        months = dates.to_period('M').unique()
        
        for m in months:
            if self.config.vade_tipi == "ENDEKS" and m.month % 2 != 0:
                continue
            month_date = m.to_timestamp().date()
            vade_gunu = vade_sonu_is_gunu(month_date, self.config.vade_tipi)
            vade_dates.add(vade_gunu)
        
        return vade_dates
    
    def _detect_vade_transitions(self) -> set:
        """Vade gecisi barlarini tespit et"""
        transitions = set()
        dates = pd.to_datetime(self.times)
        
        for i in range(1, len(dates)):
            if dates[i].month != dates[i-1].month:
                if self.config.vade_tipi == "ENDEKS" and dates[i].month % 2 == 1:
                    transitions.add(i)
                elif self.config.vade_tipi == "SPOT":
                    transitions.add(i)
        
        return transitions
    
    def get_signal(self, i: int, current_position: str,
                   entry_price: float = 0.0,
                   extreme_price: float = 0.0) -> Signal:
        """
        Bar i icin sinyal uret.
        
        Args:
            i: Bar indeksi
            current_position: "FLAT", "LONG", "SHORT"
            entry_price: Giris fiyati
            extreme_price: En yuksek/dusuk fiyat (trailing icin)
        """
        cfg = self.config
        
        # Erken barlar icin sinyal yok
        if i < self.warmup_bars or i < 1:
            return Signal.NONE
        
        # --- SEANS VE VADE KONTROLU ---
        if self.times:
            dt = self.times[i] if isinstance(self.times[i], datetime) else pd.Timestamp(self.times[i]).to_pydatetime()
            t = dt.time()
            current_date = dt.date()
            
            # Seans kontrolu
            gun_seansi = time(9, 30) <= t < time(18, 15)
            aksam_seansi = time(19, 0) <= t < time(23, 0)
            if not (gun_seansi or aksam_seansi):
                return Signal.NONE
            
            # Vade sonu gun kontrolu
            is_vade_sonu = current_date in self.vade_sonu_gunleri
            
            # Arefe kontrolu
            arefe = is_arefe(current_date)
            
            # Arefe + Vade sonu: 11:30 sonrasi flat
            if arefe and is_vade_sonu and t > time(11, 30):
                if current_position != "FLAT":
                    return Signal.FLAT
                return Signal.NONE
            
            # Arefe (vade sonu degil): 11:30 sonrasi flat
            if arefe and not is_vade_sonu and t > time(11, 30):
                if current_position != "FLAT":
                    return Signal.FLAT
                self.arefe_flat = True
                return Signal.NONE
                
            # Vade sonu: 17:40 sonrasi flat
            if is_vade_sonu and not arefe and t > time(17, 40):
                if current_position != "FLAT":
                    return Signal.FLAT
                self.warmup_aktif = True
                self.warmup_baslangic_bar = -999
                return Signal.NONE
            
            # Arefe veya vade sonu yasakli saatler
            if (arefe and t > time(11, 30)) or (is_vade_sonu and not arefe and t > time(17, 40)):
                return Signal.NONE
            
            # Warmup aktifse bekle
            if self.warmup_aktif and self.warmup_baslangic_bar == -999:
                yeni_seans = False
                if aksam_seansi and i > 0:
                    prev_dt = self.times[i-1] if isinstance(self.times[i-1], datetime) else pd.Timestamp(self.times[i-1]).to_pydatetime()
                    if prev_dt.time() < time(19, 0):
                        yeni_seans = True
                if gun_seansi and time(9, 30) <= t < time(9, 35):
                    if i > 0:
                        prev_dt = self.times[i-1] if isinstance(self.times[i-1], datetime) else pd.Timestamp(self.times[i-1]).to_pydatetime()
                        if current_date != prev_dt.date():
                            yeni_seans = True
                if yeni_seans:
                    self.warmup_baslangic_bar = i
            
            if self.warmup_aktif and self.warmup_baslangic_bar > 0:
                if (i - self.warmup_baslangic_bar) < self.vade_cooldown_bar:
                    return Signal.NONE
                else:
                    self.warmup_aktif = False
            
            # Arefe flat durumu temizleme
            if self.arefe_flat and i > 0:
                prev_dt = self.times[i-1] if isinstance(self.times[i-1], datetime) else pd.Timestamp(self.times[i-1]).to_pydatetime()
                if current_date != prev_dt.date():
                    self.arefe_flat = False
        
        # --- CIKIS MANTIGI (ATR-bazli) ---
        atr_val = self.atr[i] if i < len(self.atr) else 0.0
        
        if current_position == "LONG":
            # Stop Loss
            sl_price = entry_price - (atr_val * cfg.atr_sl)
            if self.closes[i] <= sl_price:
                return Signal.FLAT
            
            # Take Profit
            tp_price = entry_price + (atr_val * cfg.atr_tp)
            if self.closes[i] >= tp_price:
                return Signal.FLAT
            
            # Trailing Stop
            trail_price = extreme_price - (atr_val * cfg.atr_trail)
            if self.closes[i] < trail_price:
                return Signal.FLAT
        
        elif current_position == "SHORT":
            # Stop Loss
            sl_price = entry_price + (atr_val * cfg.atr_sl)
            if self.closes[i] >= sl_price:
                return Signal.FLAT
            
            # Take Profit
            tp_price = entry_price - (atr_val * cfg.atr_tp)
            if self.closes[i] <= tp_price:
                return Signal.FLAT
            
            # Trailing Stop
            trail_price = extreme_price + (atr_val * cfg.atr_trail)
            if self.closes[i] > trail_price:
                return Signal.FLAT
        
        # --- GIRIS MANTIGI ---
        if current_position == "FLAT":
            mom = self.momentum[i]
            
            # Momentum bant filtresi (ortak on kosul)
            mom_bandinda = mom > cfg.mom_alt and mom < cfg.mom_ust
            
            if mom_bandinda:
                # AL: HH breakout + EMA > DSMA + Close > MA + MOM > 100 + Hacim
                hh_breakout = self.hh[i] > self.hh[i-1]
                ema_trend = self.ema[i] > self.dsma[i]
                fiyat_ma_ustu = self.closes[i] > self.ma[i]
                mom_pozitif = mom > 100
                vol_ok = self.volumes[i] >= self.vol_hhv[i-1] * 0.8  # Hacim teyidi
                
                if hh_breakout and ema_trend and fiyat_ma_ustu and mom_pozitif and vol_ok:
                    return Signal.LONG
                
                # SAT: LL breakdown + EMA < DSMA + Close < MA + MOM < 100 + Hacim
                if cfg.yon_modu == "CIFT":
                    ll_breakdown = self.ll[i] < self.ll[i-1]
                    ema_trend_neg = self.ema[i] < self.dsma[i]
                    fiyat_ma_alti = self.closes[i] < self.ma[i]
                    mom_negatif = mom < 100
                    
                    if ll_breakdown and ema_trend_neg and fiyat_ma_alti and mom_negatif and vol_ok:
                        return Signal.SHORT
        
        return Signal.NONE
    
    def generate_all_signals(self) -> tuple:
        """
        Tum barlar icin sinyal uret - Optimizer backtesti icin.
        
        Returns:
            Tuple: (signals, exits_long, exits_short)
        """
        n = self.n
        signals = np.zeros(n, dtype=int)
        exits_long = np.zeros(n, dtype=bool)
        exits_short = np.zeros(n, dtype=bool)
        
        position = "FLAT"
        entry_price = 0.0
        extreme_price = 0.0
        
        for i in range(n):
            # Extreme price guncelle
            if position == "LONG":
                extreme_price = max(extreme_price, self.highs[i])
            elif position == "SHORT":
                extreme_price = min(extreme_price, self.lows[i])
            
            sig = self.get_signal(i, position, entry_price, extreme_price)
            
            if sig == Signal.LONG:
                signals[i] = 1
                position = "LONG"
                entry_price = self.closes[i]
                extreme_price = self.highs[i]
            elif sig == Signal.SHORT:
                # SADECE_AL modunda SHORT -> mevcut LONG kapat (FLAT)
                if self.config.yon_modu == "SADECE_AL":
                    if position == "LONG":
                        exits_long[i] = True
                        position = "FLAT"
                        entry_price = 0.0
                        extreme_price = 0.0
                else:
                    signals[i] = -1
                    position = "SHORT"
                    entry_price = self.closes[i]
                    extreme_price = self.lows[i]
            elif sig == Signal.FLAT:
                if position == "LONG":
                    exits_long[i] = True
                elif position == "SHORT":
                    exits_short[i] = True
                position = "FLAT"
                entry_price = 0.0
                extreme_price = 0.0
        
        return signals, exits_long, exits_short
    
    @classmethod
    def from_config_dict(cls, data_cache, config_dict: dict, times: Optional[List[datetime]] = None):
        """
        Optimizer icin fabrika metodu - config dict'ten strateji olusturur.
        """
        config = ParadiseConfig(
            ema_period=int(config_dict.get('ema_period', 21)),
            dsma_period=int(config_dict.get('dsma_period', 50)),
            ma_period=int(config_dict.get('ma_period', 20)),
            hh_period=int(config_dict.get('hh_period', 25)),
            vol_hhv_period=int(config_dict.get('vol_hhv_period', 14)),
            mom_period=int(config_dict.get('mom_period', 60)),
            mom_alt=float(config_dict.get('mom_alt', 98.0)),
            mom_ust=float(config_dict.get('mom_ust', 102.0)),
            atr_period=int(config_dict.get('atr_period', 14)),
            atr_sl=float(config_dict.get('atr_sl', 2.0)),
            atr_tp=float(config_dict.get('atr_tp', 4.0)),
            atr_trail=float(config_dict.get('atr_trail', 2.5)),
            vade_tipi=config_dict.get('vade_tipi', 'ENDEKS'),
            yon_modu=config_dict.get('yon_modu', 'CIFT'),
        )
        
        # Data cache'den degerleri al
        opens = list(data_cache.opens) if hasattr(data_cache, 'opens') else data_cache['opens']
        highs = list(data_cache.highs) if hasattr(data_cache, 'highs') else data_cache['highs']
        lows = list(data_cache.lows) if hasattr(data_cache, 'lows') else data_cache['lows']
        closes = list(data_cache.closes) if hasattr(data_cache, 'closes') else data_cache['closes']
        typical = list(data_cache.typical) if hasattr(data_cache, 'typical') else data_cache['typical']
        volumes = list(data_cache.volumes) if hasattr(data_cache, 'volumes') else data_cache.get('volumes', None)
        _times = times or (list(data_cache.times) if hasattr(data_cache, 'times') else data_cache.get('times', []))
        
        return cls(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            typical=typical,
            times=_times,
            volumes=volumes,
            config=config,
        )
