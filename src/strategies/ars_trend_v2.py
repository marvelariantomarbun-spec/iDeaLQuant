# -*- coding: utf-8 -*-
"""
IdealQuant - ARS Trend Takip Stratejisi v2.0
IdealData ARS_Trend_v2 stratejisinin Python portu (1DK odaklı)
"""

from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
import numpy as np
import pandas as pd

from src.indicators.core import EMA, ATR, Momentum, HHV, LLV, ARS_Dynamic, MoneyFlowIndex
from .common import Signal

# ===============================================================================================
# DİNAMİK BAYRAM TARİHLERİ (2024-2030) - IdealData ile birebir aynı
# ===============================================================================================
BAYRAM_TARIHLERI = {
    # Ramazan Bayramı (3 gün)
    2024: {'ramazan': date(2024, 4, 10), 'kurban': date(2024, 6, 16)},
    2025: {'ramazan': date(2025, 3, 30), 'kurban': date(2025, 6, 6)},
    2026: {'ramazan': date(2026, 3, 20), 'kurban': date(2026, 5, 27)},
    2027: {'ramazan': date(2027, 3, 9), 'kurban': date(2027, 5, 16)},
    2028: {'ramazan': date(2028, 2, 26), 'kurban': date(2028, 5, 5)},
    2029: {'ramazan': date(2029, 2, 14), 'kurban': date(2029, 4, 24)},
    2030: {'ramazan': date(2030, 2, 3), 'kurban': date(2030, 4, 13)},
}

# Resmi Tatiller (MM-DD formatında)
RESMI_TATILLER = [
    (1, 1),   # Yılbaşı
    (4, 23),  # 23 Nisan
    (5, 1),   # 1 Mayıs
    (5, 19),  # 19 Mayıs
    (7, 15),  # 15 Temmuz
    (8, 30),  # 30 Ağustos
    (10, 29), # 29 Ekim
]

def is_bayram_tatili(d: date) -> bool:
    """Bayram tatili mi kontrol et (Ramazan 3 gün, Kurban 4 gün)"""
    yil = d.year
    if yil not in BAYRAM_TARIHLERI:
        return False
    
    bayramlar = BAYRAM_TARIHLERI[yil]
    ramazan = bayramlar['ramazan']
    kurban = bayramlar['kurban']
    
    # Ramazan Bayramı (3 gün)
    if ramazan <= d <= ramazan + timedelta(days=3):
        return True
    
    # Kurban Bayramı (4 gün)
    if kurban <= d <= kurban + timedelta(days=4):
        return True
    
    return False

def is_arefe(d: date) -> bool:
    """Arefe günü mü kontrol et"""
    yil = d.year
    if yil not in BAYRAM_TARIHLERI:
        return False
    
    bayramlar = BAYRAM_TARIHLERI[yil]
    ramazan_arefe = bayramlar['ramazan'] - timedelta(days=1)
    kurban_arefe = bayramlar['kurban'] - timedelta(days=1)
    
    return d == ramazan_arefe or d == kurban_arefe

def is_resmi_tatil(d: date) -> bool:
    """Resmi tatil mi kontrol et"""
    return (d.month, d.day) in RESMI_TATILLER

def is_tatil_gunu(d: date) -> bool:
    """Herhangi bir tatil günü mü (hafta sonu dahil)"""
    # Hafta sonu
    if d.weekday() >= 5:
        return True
    # Resmi tatil
    if is_resmi_tatil(d):
        return True
    # Bayram tatili
    if is_bayram_tatili(d):
        return True
    return False

def vade_sonu_is_gunu(dt: date, vade_tipi: str = "ENDEKS") -> date:
    """
    Vade sonu iş gününü hesapla - IdealData ile birebir aynı mantık
    
    Args:
        dt: Tarih
        vade_tipi: "ENDEKS" (çift ay) veya "SPOT" (her ay)
    
    Returns:
        Vade sonu iş günü
    """
    import calendar
    
    # Ayın son günü
    ay_sonu = date(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])
    d = ay_sonu
    
    # Max 15 gün geri git
    for _ in range(15):
        # Hafta sonu
        if d.weekday() >= 5:
            d -= timedelta(days=1)
            continue
        
        # Resmi tatil
        if is_resmi_tatil(d):
            d -= timedelta(days=1)
            continue
        
        # Bayram tatili
        if is_bayram_tatili(d):
            d -= timedelta(days=1)
            continue
        
        break
    
    return d

def is_seans_icinde(t: time) -> bool:
    """Seans saati içinde mi kontrol et (09:30-18:15, 19:00-23:00)"""
    gun_seansi = time(9, 30) <= t < time(18, 15)
    aksam_seansi = time(19, 0) <= t < time(23, 0)
    return gun_seansi or aksam_seansi

@dataclass
class StrategyConfigV2:
    """ARS Trend v2 Strateji Konfigürasyonu (v4.1 - 21 Parametre)"""
    # 1. Grup: ARS Parametreleri (5)
    ars_ema_period: int = 3
    ars_atr_period: int = 10
    ars_atr_mult: float = 0.5
    ars_min_band: float = 0.002
    ars_max_band: float = 0.015
    
    # 2. Grup: Giriş Filtreleri (7)
    momentum_period: int = 5
    momentum_threshold: float = 100.0
    breakout_period: int = 10
    mfi_period: int = 14
    mfi_hhv_period: int = 14
    mfi_llv_period: int = 14
    volume_hhv_period: int = 14
    
    # 3. Grup: Çıkış / Risk (6)
    atr_exit_period: int = 14    # ATR_Exit_Period
    atr_sl_mult: float = 2.0     # ATR_SL_Mult
    atr_tp_mult: float = 5.0     # ATR_TP_Mult
    atr_trail_mult: float = 2.0  # ATR_Trail_Mult
    exit_confirm_bars: int = 2   # Exit_Confirm_Bars
    exit_confirm_mult: float = 1.0 # Exit_Confirm_Mult
    
    # 4. Grup: İnce Ayar (3)
    volume_mult: float = 0.8    # Hacim breakout çarpanı
    volume_llv_period: int = 14
    use_atr_exit: bool = True
    
    # Vade Yönetimi
    vade_tipi: str = "ENDEKS" # "ENDEKS" veya "SPOT"
    
    def get_max_period(self) -> int:
        """En uzun indikatör periyodunu hesapla - Isınma periyodu için"""
        periods = [
            self.ars_ema_period,
            self.ars_atr_period,
            self.momentum_period,
            self.breakout_period,
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
        self.volumes = volumes if volumes is not None else [1.0] * self.n  # Varsayılan 1.0 (MFI için)
        
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
        # ARS için kullandığımız ATR
        # (Hesaplaması burada bitti ama ATR çıkışta da lazım, o yüzden aşağıda tekrar hesaplamamak için
        # ARS_Dynamic fonksiyonu ATR'yi dışarı vermiyor. Mecburen tekrar hesaplayacağız veya
        # ARS_Dynamic'i modifiye edeceğiz. Şimdilik tekrar hesaplama (maliyet düşük).
        
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
        
        # ATR Hesapla (Exit ve ARS için ortak kullanılabilir, veya bağımsız)
        # ATR Hesapla (Exit ve ARS için ortak kullanılabilir, veya bağımsız)
        # ARS için ars_atr_period, Exit için atr_exit_period kullanılır
        self.atr_ars = ATR(self.highs, self.lows, self.closes, cfg.ars_atr_period)
        self.atr_exit = ATR(self.highs, self.lows, self.closes, cfg.atr_exit_period)
        
        # 4. MFI Breakout
        self.mfi = MoneyFlowIndex(self.highs, self.lows, self.closes, 
                                   self.volumes, cfg.mfi_period)
        self.mfi_hhv = HHV(self.mfi, cfg.mfi_hhv_period)
        self.mfi_llv = LLV(self.mfi, cfg.mfi_llv_period)
        
        # 5. Hacim Breakout
        self.volume_hhv = HHV(self.volumes, cfg.volume_hhv_period)
        self.volume_llv = LLV(self.volumes, cfg.volume_llv_period)
        
    def _calculate_vade_sonlari(self) -> Set[date]:
        """Vade sonu tarihlerini hesapla - IdealData ile birebir aynı mantık"""
        vade_dates = set()
        # Veri setindeki aylar
        dates = pd.to_datetime(self.times)
        months = dates.to_period('M').unique()
        
        for m in months:
            # Sadece çift aylar (Endeks Vadelisi) - SPOT ise her ay
            if self.config.vade_tipi == "ENDEKS" and m.month % 2 != 0:
                continue
            
            # Vade sonu iş gününü hesapla (tatil/bayram dahil)
            month_date = m.to_timestamp().date()
            vade_gunu = vade_sonu_is_gunu(month_date, self.config.vade_tipi)
            vade_dates.add(vade_gunu)
            
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
    
    def get_warmup_status(self, i: int, prev_flat_reason: str = None, prev_flat_date: date = None) -> dict:
        """
        Warmup durumunu kontrol et - Backtest döngüsünde kullanılmak üzere
        
        Args:
            i: Bar indeksi
            prev_flat_reason: Önceki flat nedeni ('vade_sonu', 'arefe', 'arefe_vade_sonu', None)
            prev_flat_date: Önceki flat tarihi
            
        Returns:
            dict: {
                'in_warmup': bool,  # Warmup aktif mi?
                'warmup_start': int,  # Warmup başlangıç barı
                'warmup_remaining': int,  # Kalan bar sayısı
                'skip_warmup': bool  # Warmup atlanacak mı? (sadece arefe sonrası)
            }
        """
        current_date = self.times[i].date()
        current_t = self.times[i].time()
        
        result = {
            'in_warmup': False,
            'warmup_start': -1,
            'warmup_remaining': 0,
            'skip_warmup': False
        }
        
        # Arefe sonrası (vade geçişi değilse) warmup atlanır
        if prev_flat_reason == 'arefe' and prev_flat_date is not None:
            if current_date > prev_flat_date:
                result['skip_warmup'] = True
                return result
        
        # Vade geçişi veya arefe+vade sonrası warmup gerekli
        if prev_flat_reason in ['vade_sonu', 'arefe_vade_sonu'] and prev_flat_date is not None:
            # Yeni seansın ilk barı mı?
            is_new_session = False
            
            if i > 0:
                prev_date = self.times[i-1].date()
                prev_t = self.times[i-1].time()
                
                # Akşam seansı başlangıcı (19:00) - aynı gün vade geçişi için
                if current_date == prev_flat_date and current_t >= time(19, 0) and prev_t < time(19, 0):
                    is_new_session = True
                
                # Sabah seansı başlangıcı (09:30) - tatil/bayram sonrası için
                if current_date > prev_flat_date and current_t >= time(9, 30):
                    is_new_session = True
            
            if is_new_session:
                result['in_warmup'] = True
                result['warmup_start'] = i
                result['warmup_remaining'] = self.warmup_period
                
        return result

    def get_signal(self, i: int, current_position: str, 
                   entry_price: float = 0, 
                   extreme_price: float = 0,
                   return_flat_reason: bool = False) -> Signal:
        
        if i < 50: 
            return (Signal.NONE, None) if return_flat_reason else Signal.NONE
        
        cfg = self.config
        current_time = self.times[i]
        current_date = current_time.date()
        current_t = current_time.time()
        
        flat_reason = None  # 'vade_sonu', 'arefe', 'arefe_vade_sonu'
        
        # --- SEANS KONTROLÜ (09:30-18:15, 19:00-23:00) ---
        if not is_seans_icinde(current_t):
            return (Signal.NONE, None) if return_flat_reason else Signal.NONE  # Seans dışında sinyal yok
        
        # --- VADE/TATİL YÖNETİMİ ---
        # Not: Warmup kontrolü backtest döngüsünde yapılmalı (state tracking gerektirir)
        # Burada sadece FLAT sinyalleri üretiyoruz
        
        is_vade_sonu = current_date in self.vade_sonu_gunleri
        is_arefe_gunu = is_arefe(current_date)
        
        # ===== SENARYO 1: Arefe + Vade Sonu (11:30'da flat) =====
        if is_arefe_gunu and is_vade_sonu and current_t > time(11, 30):
            if current_position != "FLAT":
                flat_reason = 'arefe_vade_sonu'
                return (Signal.FLAT, flat_reason) if return_flat_reason else Signal.FLAT
                
        # ===== SENARYO 2: Sadece Arefe (11:30'da flat, ertesi gün warmup YOK) =====
        elif is_arefe_gunu and not is_vade_sonu and current_t > time(11, 30):
            if current_position != "FLAT":
                flat_reason = 'arefe'
                return (Signal.FLAT, flat_reason) if return_flat_reason else Signal.FLAT
                
        # ===== SENARYO 3: Normal Vade Sonu (17:40'da flat) =====
        elif is_vade_sonu and current_t >= time(17, 40):
            if current_position != "FLAT":
                flat_reason = 'vade_sonu'
                return (Signal.FLAT, flat_reason) if return_flat_reason else Signal.FLAT
        
        # --- ÇIKIŞ MANTIĞI ---
        # --- ÇIKIŞ MANTIĞI ---
        
        # 0. Parametreleri Hazırla
        if cfg.use_atr_exit:
            atr_val = self.atr_exit[i]
            # Dinamik K Hesaplama (ATR Exit Period ile değil, ARS'nin kendi dinamik yapısından mı? 
            # Kullanıcı örneği: dinamikK = 0.0123 (ATR bazlı band). 
            # ARS_Dynamic fonksiyonu bunu içsel hesaplıyor. Ancak dışarıya sadece ARS değerini veriyoruz.
            # Yaklaşık olarak: (ATR / EMA) * Multiplier. ARS class'ında bu değer saklanmıyor.
            # Basitlik için: ARS ile Fiyat farkını ATR cinsinden kontrol etmek yerine,
            # Kullanıcı "Mesafe: Fiyat, ARS'tan en az (dinamikK * Confirm_Mult) kadar uzaklaştı" dedi.
            # ARS Dynamic K formülü: K = (ATR / EMA) * Mult.
            # Mesafe Eşiği = ARS * (K * Confirm_Mult) 
            # = ARS * ((ATR/EMA * Mult) * Confirm_Mult)
            # Kabaca: ARS * (ATR/Close * Mult * Confirm_Mult) ~= ATR * Mult * Confirm_Mult
            # Basitleştirilmiş Mesafe: ATR * ars_atr_mult * exit_confirm_mult
            
            dist_threshold = atr_val * cfg.ars_atr_mult * cfg.exit_confirm_mult
            
        else:
            atr_val = 0.0
            dist_threshold = 0.0

        if current_position == "LONG":
            # 1. Trend Tersine Dönüş (Double Confirmation)
            # LONG'dayken çıkış: Fiyat ARS'nin ALTINA inmeli
            
            should_exit_trend = False
            
            # A. Çoklu Bar Kontrolü
            bars_below = True
            for k in range(cfg.exit_confirm_bars):
                idx = i - k
                if idx < 0: 
                    bars_below = False
                    break
                if self.closes[idx] >= self.ars[idx]: # ARS üstünde veya eşitse ihlal yok
                    bars_below = False
                    break
            
            # B. Mesafe Kontrolü (Son bar için)
            # Fiyat ARS'den belirli miktar aşağıda mı? ARS - Close > Threshold
            distance_ok = (self.ars[i] - self.closes[i]) > dist_threshold
            
            if bars_below and distance_ok:
                should_exit_trend = True
            
            # Klasik Tek Bar Dönüş (Yedek/Legacy) - Eğer ATR Exit kapalıysa
            if not cfg.use_atr_exit:
                 if self.trend_yonu[i] == -1 and self.trend_yonu[i-1] == 1:
                     should_exit_trend = True
            
            if should_exit_trend:
                return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
            
            # 2. Kar Al
            if cfg.use_atr_exit:
                 target_price = entry_price + (atr_val * cfg.atr_tp_mult)
            else:
                 target_price = entry_price * (1 + cfg.kar_al_pct / 100.0)
                 
            if self.closes[i] >= target_price:
                return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
                
            # 3. İzleyen Stop / Stop Loss
            if cfg.use_atr_exit:
                # Trailing Stop: En yüksek fiyat - X * ATR
                trailing_stop_price = extreme_price - (atr_val * cfg.atr_trail_mult)
                
                # Stop Loss (Initial): Giriş - Y * ATR 
                # extreme_price başlangıçta entry_price olduğu için, eğer fiyat hiç yükselmediyse
                # Entry - Trail_Mult * ATR olur. 
                # Kullanıcı ayrıca "ATR_SL_Mult" tanımladı.
                # Genellikle SL başta geçerlidir. Trailing devreye girene kadar?
                # Veya: StopPrice = Max(Entry - SL_Mult*ATR, Extreme - Trail_Mult*ATR)
                
                sl_price = entry_price - (atr_val * cfg.atr_sl_mult)
                ts_price = extreme_price - (atr_val * cfg.atr_trail_mult)
                
                actual_stop = max(sl_price, ts_price)
                
                if self.closes[i] < actual_stop:
                    return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
            else:
                trailing_stop_price = extreme_price * (1 - cfg.iz_stop_pct / 100.0)
                if self.closes[i] < trailing_stop_price:
                    return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
                
        elif current_position == "SHORT":
            # 1. Trend Tersine Dönüş (Double Confirmation)
            # SHORT'dayken çıkış: Fiyat ARS'nin ÜSTÜNE çıkmalı
            
            should_exit_trend = False
            
            # A. Çoklu Bar Kontrolü
            bars_above = True
            for k in range(cfg.exit_confirm_bars):
                idx = i - k
                if idx < 0: 
                    bars_above = False
                    break
                if self.closes[idx] <= self.ars[idx]: # ARS altında veya eşitse ihlal yok
                    bars_above = False
                    break
            
            # B. Mesafe Kontrolü (Son bar için)
            # Fiyat ARS'den yukarıda mı? Close - ARS > Threshold
            distance_ok = (self.closes[i] - self.ars[i]) > dist_threshold
            
            if bars_above and distance_ok:
                should_exit_trend = True
                
            # Klasik Tek Bar Dönüş (Yedek)
            if not cfg.use_atr_exit:
                if self.trend_yonu[i] == 1 and self.trend_yonu[i-1] == -1:
                    should_exit_trend = True

            if should_exit_trend:
                return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
                
            # 2. Kar Al
            if cfg.use_atr_exit:
                target_price = entry_price - (atr_val * cfg.atr_tp_mult)
            else:
                target_price = entry_price * (1 - cfg.kar_al_pct / 100.0)
                
            if self.closes[i] <= target_price:
                return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
                
            # 3. İzleyen Stop
            if cfg.use_atr_exit:
                sl_price = entry_price + (atr_val * cfg.atr_sl_mult)
                ts_price = extreme_price + (atr_val * cfg.atr_trail_mult)
                
                actual_stop = min(sl_price, ts_price)
                
                if self.closes[i] > actual_stop:
                    return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
            else:
                trailing_stop_price = extreme_price * (1 + cfg.iz_stop_pct / 100.0)
                if self.closes[i] > trailing_stop_price:
                    return (Signal.FLAT, None) if return_flat_reason else Signal.FLAT
        
        # --- GİRİŞ MANTIĞI ---
        if current_position == "FLAT": # Sadece FLAT iken giriş ara
            
            # LONG GİRİŞ
            if self.trend_yonu[i] == 1:
                yeni_zirve = self.highs[i] >= self.hhv[i-1] and self.hhv[i] > self.hhv[i-1]
                pozitif_mom = self.momentum[i] > cfg.momentum_threshold
                
                # MFI Breakout - MFI yeni zirve yapıyor
                mfi_onay = self.mfi[i] >= self.mfi_hhv[i-1]
                
                # Hacim Breakout - Hacim ortalamanın üstünde
                volume_onay = self.volumes[i] >= self.volume_hhv[i-1] * cfg.volume_mult
                
                if yeni_zirve and pozitif_mom and mfi_onay and volume_onay:
                    return (Signal.LONG, None) if return_flat_reason else Signal.LONG
            
            # SHORT GİRİŞ
            elif self.trend_yonu[i] == -1:
                yeni_dip = self.lows[i] <= self.llv[i-1] and self.llv[i] < self.llv[i-1]
                negatif_mom = self.momentum[i] < (200 - cfg.momentum_threshold)
                
                # MFI Breakout - MFI yeni dip yapıyor
                mfi_onay = self.mfi[i] <= self.mfi_llv[i-1]
                
                # Hacim Breakout - Hacim ortalamanın üstünde
                volume_onay = self.volumes[i] >= self.volume_hhv[i-1] * cfg.volume_mult
                
                if yeni_dip and negatif_mom and mfi_onay and volume_onay:
                    return (Signal.SHORT, None) if return_flat_reason else Signal.SHORT
                    
        return (Signal.NONE, None) if return_flat_reason else Signal.NONE
