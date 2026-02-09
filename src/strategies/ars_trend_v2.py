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
from .holidays import (
    BAYRAM_TARIHLERI, RESMI_TATILLER,
    is_bayram_tatili, is_arefe, is_resmi_tatil, is_tatil_gunu,
    vade_sonu_is_gunu, is_seans_icinde
)


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

    # Legacy/Fallback (use_atr_exit=False için)
    kar_al_pct: float = 1.5        # Yüzde bazlı kar alma (ör: %1.5)
    iz_stop_pct: float = 0.8       # Yüzde bazlı izleyen stop (ör: %0.8)
    
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
        self.atr_val_cache = {} # ATR değerlerini cachelemek için (get_signal içinde)
        
        # Warmup State (Strateji 1 ile uyumlu)
        self.vade_cooldown_bar = self.warmup_period
        self.warmup_bars = max(50, self.vade_cooldown_bar)
        self.warmup_aktif = False
        self.warmup_baslangic_bar = -999
        self.arefe_flat = False
        
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
        
        # Warmup bars kontrolü
        if i < self.warmup_bars: 
             return (Signal.NONE, None) if return_flat_reason else Signal.NONE
        
        cfg = self.config
        current_time = self.times[i]
        dt = current_time # Strateji 1 ile uyum için alias
        current_date = current_time.date()
        current_t = current_time.time()
        
        flat_reason = None
        
        # ===== VADE/TATİL YÖNETİMİ (Strateji 1 ile Birebir) =====
        prev_dt = self.times[i-1] if i > 0 else None
        
        # Seans kontrolü (09:30-18:15, 19:00-23:00)
        gun_seansi = time(9, 30) <= current_t < time(18, 15)
        aksam_seansi = time(19, 0) <= current_t < time(23, 0)
        
        if not (gun_seansi or aksam_seansi):
             return (Signal.NONE, None) if return_flat_reason else Signal.NONE
        
        # Vade sonu ve Arefe kontrolü
        # Strategy 2 zaten vade_sonu_gunleri setine sahip, onu kullanalım (daha hızlı)
        vade_sonu_gun = current_date in self.vade_sonu_gunleri
        arefe = is_arefe(current_date)
        
        # SENARYO 1: Arefe + Vade Sonu → 11:30 flat + warmup aktif
        if arefe and vade_sonu_gun and current_t > time(11, 30):
            self.warmup_aktif = True
            self.warmup_baslangic_bar = -999
            self.arefe_flat = False
            if current_position != "FLAT":
                if return_flat_reason:
                    return (Signal.FLAT, "arefe_vade_sonu")
                return Signal.FLAT
        
        # SENARYO 2: Sadece Arefe → 11:30 flat
        elif arefe and not vade_sonu_gun and current_t > time(11, 30):
            self.arefe_flat = True
            if current_position != "FLAT":
                if return_flat_reason:
                    return (Signal.FLAT, "arefe")
                return Signal.FLAT
        
        # SENARYO 3: Normal Vade Sonu → 17:40 flat + warmup aktif
        elif vade_sonu_gun and current_t > time(17, 40):
            self.warmup_aktif = True
            self.warmup_baslangic_bar = -999
            self.arefe_flat = False
            if current_position != "FLAT":
                if return_flat_reason:
                    return (Signal.FLAT, "vade_sonu")
                return Signal.FLAT
        
        # Arefe/Vade günlerinde flat saatlerinde işlem yapma
        if (arefe and current_t > time(11, 30)) or (vade_sonu_gun and not arefe and current_t > time(17, 40)):
            if return_flat_reason:
                return (Signal.NONE, None)
            return Signal.NONE
        
        # Warmup başlangıç barını tespit et (yeni seans başlangıcı)
        if self.warmup_aktif and self.warmup_baslangic_bar == -999:
            yeni_seans_baslangici = False
            if prev_dt:
                prev_t = prev_dt.time()
                # Akşam seansı başlangıcı
                if aksam_seansi and prev_t < time(19, 0):
                    yeni_seans_baslangici = True
                # Gün seansı başlangıcı (yeni gün)
                if gun_seansi and time(9, 30) <= current_t < time(9, 35):
                    if current_date != prev_dt.date():
                        yeni_seans_baslangici = True
            if yeni_seans_baslangici:
                self.warmup_baslangic_bar = i
        
        # Warmup cooldown kontrolü
        if self.warmup_aktif and self.warmup_baslangic_bar > 0:
            if (i - self.warmup_baslangic_bar) < self.vade_cooldown_bar:
                return (Signal.NONE, None) if return_flat_reason else Signal.NONE
            else:
                self.warmup_aktif = False
        
        # Arefe flat günü bitimi
        if self.arefe_flat and prev_dt and current_date != prev_dt.date():
            self.arefe_flat = False
        
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

    @classmethod
    def from_config_dict(cls, data_cache, config_dict: dict, times: Optional[List[datetime]] = None):
        """
        Optimizer için fabrika metodu - config dict'ten strateji oluşturur.
        
        Args:
            data_cache: Opens, highs, lows, closes, typical, volumes içeren cache objesi
            config_dict: Strateji parametreleri (StrategyConfigV2 alanlarıyla eşleşmeli)
            times: Tarih listesi (vade/tatil yönetimi için)
        """
        config = StrategyConfigV2(
            ars_ema_period=config_dict.get('ars_ema_period', 3),
            ars_atr_period=config_dict.get('ars_atr_period', 10),
            ars_atr_mult=config_dict.get('ars_atr_mult', 0.5),
            ars_min_band=config_dict.get('ars_min_band', 0.002),
            ars_max_band=config_dict.get('ars_max_band', 0.015),
            momentum_period=config_dict.get('momentum_period', 5),
            momentum_threshold=config_dict.get('momentum_threshold', 100.0),
            breakout_period=config_dict.get('breakout_period', 10),
            mfi_period=config_dict.get('mfi_period', 14),
            mfi_hhv_period=config_dict.get('mfi_hhv_period', 14),
            mfi_llv_period=config_dict.get('mfi_llv_period', 14),
            volume_hhv_period=config_dict.get('volume_hhv_period', 14),
            atr_exit_period=config_dict.get('atr_exit_period', 14),
            atr_sl_mult=config_dict.get('atr_sl_mult', 2.0),
            atr_tp_mult=config_dict.get('atr_tp_mult', 5.0),
            atr_trail_mult=config_dict.get('atr_trail_mult', 2.0),
            exit_confirm_bars=config_dict.get('exit_confirm_bars', 2),
            exit_confirm_mult=config_dict.get('exit_confirm_mult', 1.0),
            volume_mult=config_dict.get('volume_mult', 0.8),
            volume_llv_period=config_dict.get('volume_llv_period', 14),
            use_atr_exit=config_dict.get('use_atr_exit', True),
            vade_tipi=config_dict.get('vade_tipi', 'ENDEKS'),
            kar_al_pct=float(config_dict.get('kar_al_pct', 1.5)),
            iz_stop_pct=float(config_dict.get('iz_stop_pct', 0.8)),
        )
        
        # Data cache'den değerleri al
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
    
    def generate_all_signals(self) -> tuple:
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
        entry_price = 0.0
        extreme_price = 0.0
        
        for i in range(n):
            # Extreme price güncelle
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
