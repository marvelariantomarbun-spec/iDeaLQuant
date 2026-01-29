# -*- coding: utf-8 -*-
"""
IdealQuant - Yatay Piyasa Filtresi
Yatay piyasada işlem yapılmasını engelleyen modül
"""

from typing import List, Tuple
import numpy as np
from src.indicators.core import EMA, ATR, ADX, SMA, ARS


def calculate_ars_change_status(ars: List[float], lookback: int = 10) -> List[int]:
    """
    ARS değişim durumu - ARS kaç bardır aynı?
    Returns: 1=değişiyor (trend), 0=sabit (yatay)
    """
    n = len(ars)
    result = [0] * n
    
    for i in range(lookback, n):
        ars_same = True
        for j in range(1, lookback + 1):
            if ars[i] != ars[i - j]:
                ars_same = False
                break
        result[i] = 0 if ars_same else 1  # 0=yatay, 1=trendde
    
    return result


def calculate_ars_distance(closes: List[float], ars: List[float]) -> List[float]:
    """
    Fiyat-ARS mesafesi (%)
    Çok yakınsa yatay piyasa işareti
    """
    n = len(closes)
    result = [0.0] * n
    
    for i in range(1, n):
        if ars[i] != 0:
            result[i] = abs(closes[i] - ars[i]) / ars[i] * 100
    
    return result


def calculate_bb_width(closes: List[float], period: int = 20, std_mult: float = 2.0) -> Tuple[List[float], List[float]]:
    """
    Bollinger Band genişliği ve ortalaması
    Dar = yatay piyasa
    """
    n = len(closes)
    bb_width = [0.0] * n
    
    # BB hesapla
    bb_mid = SMA(closes, period)
    
    for i in range(period - 1, n):
        # Standart sapma hesapla
        window = closes[i - period + 1:i + 1]
        std = np.std(window)
        
        bb_up = bb_mid[i] + std_mult * std
        bb_down = bb_mid[i] - std_mult * std
        
        if bb_mid[i] != 0:
            bb_width[i] = ((bb_up - bb_down) / bb_mid[i]) * 100
    
    # BB Width ortalaması
    bb_width_avg = SMA(bb_width, 50)
    
    return bb_width, bb_width_avg


class YatayFiltre:
    """
    Birleşik Yatay Piyasa Filtresi
    
    Kullanım:
        filtre = YatayFiltre(closes, highs, lows, typical)
        
        for i in range(len(closes)):
            if filtre.islem_izni(i):
                # Strateji sinyallerini uygula
            else:
                # Bekle veya pozisyon kapat
    """
    
    def __init__(self, 
                 closes: List[float],
                 highs: List[float],
                 lows: List[float],
                 typical: List[float],
                 ars_k: float = 0.0123,
                 ars_ema_period: int = 3,
                 adx_period: int = 14,
                 adx_threshold: float = 20.0,
                 distance_threshold: float = 0.15,  # %0.15
                 ars_lookback: int = 10,
                 min_conditions: int = 2,
                 hysteresis: int = 5,  # Trend -> Yatay geçişi için bekleme
                 confirmation: int = 1): # Yatay -> Trend geçişi için bekleme
        """
        Args:
            closes: Kapanış fiyatları
            highs: Yüksek fiyatlar
            lows: Düşük fiyatlar
            typical: Tipik fiyatlar
            ars_k: ARS band genişliği (sabit mod)
            ars_ema_period: ARS EMA periyodu
            adx_period: ADX periyodu
            adx_threshold: ADX trend eşiği (üstü = trend)
            distance_threshold: Fiyat-ARS minimum mesafe (%)
            ars_lookback: ARS değişim kontrolü bar sayısı
            min_conditions: Minimum kaç koşul sağlanmalı
        """
        self.n = len(closes)
        self.min_conditions = min_conditions
        self.adx_threshold = adx_threshold
        self.distance_threshold = distance_threshold
        self.hysteresis = hysteresis
        self.confirmation = confirmation
        
        # ARS hesapla
        self.ars = ARS(typical, ars_ema_period, ars_k)
        
        # ARS değişim durumu
        self.ars_change = calculate_ars_change_status(self.ars, ars_lookback)
        
        # Fiyat-ARS mesafesi
        self.ars_distance = calculate_ars_distance(closes, self.ars)
        
        # ADX
        self.adx = ADX(highs, lows, closes, adx_period)
        
        # BB genişliği
        self.bb_width, self.bb_width_avg = calculate_bb_width(closes)
        
        # Filtre sonucu (ön hesaplama)
        self._calculate_filter()
    
    def _calculate_filter(self):
        """Tüm barlar için filtre değerini hesapla"""
        self.filter_result = [False] * self.n
        self.scores = [0] * self.n
        
        # Ham sonuçlar
        raw_filter = [False] * self.n
        
        for i in range(50, self.n):
            score = 0
            
            # Koşul 1: ARS hareketli mi?
            if self.ars_change[i] == 1:
                score += 1
            
            # Koşul 2: Fiyat ARS'dan yeterince uzak mı?
            if self.ars_distance[i] > self.distance_threshold:
                score += 1
            
            # Koşul 3: ADX yeterli mi (güçlü trend)?
            if self.adx[i] > self.adx_threshold:
                score += 1
            
            # Koşul 4: BB dar değil mi (volatilite var)?
            if self.bb_width_avg[i] > 0 and self.bb_width[i] > self.bb_width_avg[i] * 0.8:
                score += 1
            
            self.scores[i] = score
            raw_filter[i] = (score >= self.min_conditions)
            
        # Hysteresis uygula
        # Trend -> Yatay: 'hysteresis' bar boyunca False olmalı
        # Yatay -> Trend: 'confirmation' bar boyunca True olmalı
        
        current_state = True # Başlangıçta trend varsay
        trend_counter = 0
        flat_counter = 0
        
        for i in range(50, self.n):
            is_trend = raw_filter[i]
            
            if current_state: # Şu an Trend modundayız
                if not is_trend:
                    flat_counter += 1
                    trend_counter = 0
                    if flat_counter >= self.hysteresis:
                        current_state = False # Yataya geç
                else:
                    flat_counter = 0
            
            else: # Şu an Yatay modundayız
                if is_trend:
                    trend_counter += 1
                    flat_counter = 0
                    if trend_counter >= self.confirmation:
                        current_state = True # Trende geç
                else:
                    trend_counter = 0
                    
            self.filter_result[i] = current_state
    
    def islem_izni(self, bar_index: int) -> bool:
        """
        Belirtilen bar'da işlem yapılabilir mi?
        
        Returns:
            True = Trend piyasası, işlem yap
            False = Yatay piyasa, bekle
        """
        if bar_index < 0 or bar_index >= self.n:
            return False
        return self.filter_result[bar_index]
    
    def get_score(self, bar_index: int) -> int:
        """Belirtilen bar'ın trend skoru (0-4)"""
        if bar_index < 0 or bar_index >= self.n:
            return 0
        return self.scores[bar_index]
    
    def get_ars(self) -> List[float]:
        """ARS değerlerini döndür"""
        return self.ars
    
    def get_diagnostics(self, bar_index: int) -> dict:
        """Debug için detaylı bilgi"""
        if bar_index < 0 or bar_index >= self.n:
            return {}
        
        return {
            'bar': bar_index,
            'ars': self.ars[bar_index],
            'ars_change': self.ars_change[bar_index],
            'ars_distance': self.ars_distance[bar_index],
            'adx': self.adx[bar_index],
            'bb_width': self.bb_width[bar_index],
            'bb_width_avg': self.bb_width_avg[bar_index],
            'score': self.scores[bar_index],
            'trade_allowed': self.filter_result[bar_index]
        }
