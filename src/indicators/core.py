"""
IdealQuant - Technical Indicators
IdealData compatible implementations
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from numba import jit


# =============================================================================
# MOVING AVERAGES
# =============================================================================

def SMA(data: List[float], period: int) -> List[float]:
    """Simple Moving Average - IdealData compatible"""
    result = [0.0] * len(data)
    if len(data) < period:
        return result
    
    # Calculate simple moving average
    s = pd.Series(data)
    result = s.rolling(window=period).mean().fillna(0).tolist()
    
    # Fallback if pandas not available or slower (pure python implementation for consistency)
    result = [0.0] * len(data)
    if len(data) >= period:
        for i in range(period - 1, len(data)):
            total = sum(data[i - period + 1 : i + 1])
            result[i] = total / period
            
    return result


def EMA(data: List[float], period: int) -> List[float]:
    """Exponential Moving Average - IdealData compatible"""
    result = [0.0] * len(data)
    if len(data) < period:
        return result
    
    multiplier = 2.0 / (period + 1)
    
    # First EMA is SMA
    total = sum(data[:period])
    result[period - 1] = total / period
    
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    
    return result


def WMA(data: List[float], period: int) -> List[float]:
    """Weighted Moving Average"""
    result = [0.0] * len(data)
    if len(data) < period:
        return result
    
    denominator = period * (period + 1) / 2.0
    
    for i in range(period - 1, len(data)):
        total = 0.0
        for j in range(period):
            total += data[i - j] * (period - j)
        result[i] = total / denominator
    
    return result


def HullMA(data: List[float], period: int) -> List[float]:
    """Hull Moving Average"""
    wma1 = WMA(data, period // 2)
    wma2 = WMA(data, period)
    
    diff = [2 * wma1[i] - wma2[i] for i in range(len(data))]
    
    return WMA(diff, int(np.sqrt(period)))



def RMA(data: List[float], period: int) -> List[float]:
    """
    Running Moving Average (Wilder's Smoothing)
    Equivalent to EMA with alpha = 1 / period
    Used in RSI, ADX, ATR
    """
    n = len(data)
    result = [0.0] * n
    
    if n < period:
        return result
        
    # First value is simple average
    result[period-1] = sum(data[:period]) / period
    
    # Subsequent values: (Previous * (period - 1) + Current) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period - 1) + data[i]) / period
        
    return result


def MA(data: List[float], method: str, period: int) -> List[float]:
    """Generic Moving Average - IdealData compatible"""
    method = method.lower() if isinstance(method, str) else "simple"
    
    if method in ("simple", "sma"):
        return SMA(data, period)
    elif method in ("exp", "exponential", "ema"):
        return EMA(data, period)
    elif method in ("weighted", "wma"):
        return WMA(data, period)
    elif method in ("hull", "hma"):
        return HullMA(data, period)
    else:
        return SMA(data, period)


# =============================================================================
# OSCILLATORS
# =============================================================================

def RSI(closes: List[float], period: int = 14) -> List[float]:
    """Relative Strength Index - IdealData compatible (Wilder's Smoothing)"""
    n = len(closes)
    result = [50.0] * n
    
    if n <= period:
        return result
    
    deltas = [0.0] * n
    for i in range(1, n):
        deltas[i] = closes[i] - closes[i-1]
        
    gains = [max(0, d) for d in deltas]
    losses = [abs(min(0, d)) for d in deltas]
    
    # RMA logic manually implemented to match Wilder's RSI exactly
    avg_gain = [0.0] * n
    avg_loss = [0.0] * n
    
    # First value is SMA of gains/losses
    avg_gain[period] = sum(gains[1:period+1]) / period
    avg_loss[period] = sum(losses[1:period+1]) / period
    
    if avg_loss[period] == 0:
        result[period] = 100.0
    else:
        rs = avg_gain[period] / avg_loss[period]
        result[period] = 100.0 - (100.0 / (1.0 + rs))
        
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
        
        if avg_loss[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return result


def Momentum(closes: List[float], period: int = 10) -> List[float]:
    """Momentum - IdealData compatible (returns 100 + percentage change)"""
    result = [100.0] * len(closes)
    
    for i in range(period, len(closes)):
        if closes[i - period] != 0:
            result[i] = (closes[i] / closes[i - period]) * 100.0
    
    return result


def Stochastic(highs: List[float], lows: List[float], closes: List[float], 
               k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Stochastic Oscillator
    Returns: (K, D) tuple of lists
    """
    n = len(closes)
    k = [50.0] * n
    
    for i in range(k_period - 1, n):
        highest = max(highs[i - k_period + 1 : i + 1])
        lowest = min(lows[i - k_period + 1 : i + 1])
        
        if highest != lowest:
            k[i] = ((closes[i] - lowest) / (highest - lowest)) * 100.0
    
    d = SMA(k, d_period)
    
    return k, d


# =============================================================================
# VOLATILITY
# =============================================================================

def ATR(highs: List[float], lows: List[float], closes: List[float], 
        period: int = 14) -> List[float]:
    """Average True Range - IdealData compatible"""
    n = len(closes)
    result = [0.0] * n
    
    if n < 2:
        return result
    
    # Calculate True Range
    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # Use RMA helper for smoothing
    result = RMA(tr, period)
    
    return result


def BollingerBands(closes: List[float], period: int = 20, 
                   deviation: float = 2.0) -> tuple:
    """
    Bollinger Bands
    Returns: (upper, middle, lower) tuple of lists
    """
    middle = SMA(closes, period)
    n = len(closes)
    upper = [0.0] * n
    lower = [0.0] * n
    
    for i in range(period - 1, n):
        # Calculate standard deviation
        window = closes[i - period + 1 : i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = np.sqrt(variance)
        
        upper[i] = middle[i] + deviation * std
        lower[i] = middle[i] - deviation * std
    
    return upper, middle, lower


# =============================================================================
# TREND INDICATORS
# =============================================================================

def ADX(highs: List[float], lows: List[float], closes: List[float], 
        period: int = 14) -> List[float]:
    """Average Directional Index - IdealData compatible (RMA)"""
    n = len(closes)
    result = [0.0] * n
    
    if n < period + 1:
        return result
    
    # Calculate +DM, -DM, TR
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    
    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # Smooth with Wilder's method (RMA)
    smoothed_plus_dm = RMA(plus_dm, period)
    smoothed_minus_dm = RMA(minus_dm, period)
    smoothed_tr = RMA(tr, period)
    
    # Calculate DX
    dx = [0.0] * n
    for i in range(period, n):
        if smoothed_tr[i] != 0:
            p_di = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            m_di = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100
            
            di_sum = p_di + m_di
            if di_sum != 0:
                dx[i] = abs(p_di - m_di) / di_sum * 100
    
    # ADX is RMA of DX
    result = RMA(dx, period)
            
    return result


# =============================================================================
# HIGH/LOW FUNCTIONS
# =============================================================================

def HHV(highs: List[float], period: int) -> List[float]:
    """Highest High Value - IdealData compatible"""
    result = [0.0] * len(highs)
    
    for i in range(period - 1, len(highs)):
        result[i] = max(highs[i - period + 1 : i + 1])
    
    return result


def LLV(lows: List[float], period: int) -> List[float]:
    """Lowest Low Value - IdealData compatible"""
    result = [float('inf')] * len(lows)
    
    for i in range(period - 1, len(lows)):
        result[i] = min(lows[i - period + 1 : i + 1])
    
    # Replace inf with 0 for early bars
    for i in range(min(period - 1, len(lows))):
        result[i] = 0.0
    
    return result


# =============================================================================
# CUSTOM INDICATORS
# =============================================================================

def ARS(typical: List[float], ema_period: int = 3, k: float = 0.0123) -> List[float]:
    """
    Adaptive Regime Switch - Custom indicator
    Classic version with fixed percentage band
    """
    n = len(typical)
    result = [0.0] * n
    
    # Calculate EMA of typical price
    ema = EMA(typical, ema_period)
    
    for i in range(1, n):
        alt_band = ema[i] * (1 - k)
        ust_band = ema[i] * (1 + k)
        
        if alt_band > result[i - 1]:
            result[i] = alt_band
        elif ust_band < result[i - 1]:
            result[i] = ust_band
        else:
            result[i] = result[i - 1]
    
    return result


def ARS_Dynamic(typical: List[float], highs: List[float], lows: List[float], 
                closes: List[float], ema_period: int = 5, atr_period: int = 14,
                atr_mult: float = 0.7, min_k: float = 0.003, 
                max_k: float = 0.020) -> List[float]:
    """
    Adaptive Regime Switch - Dynamic version
    Band width adapts to volatility (ATR)
    Includes dynamic rounding based on ATR (IdealData compatible)
    """
    import math
    
    n = len(typical)
    result = [0.0] * n
    
    # Calculate EMA and ATR
    ema = EMA(typical, ema_period)
    atr = ATR(highs, lows, closes, atr_period)
    
    # First value
    result[0] = ema[0]
    
    for i in range(1, n):
        # Dynamic K based on ATR
        if ema[i] != 0:
            dynamic_k = (atr[i] / ema[i]) * atr_mult
            dynamic_k = max(min_k, min(max_k, dynamic_k))
        else:
            dynamic_k = min_k
        
        alt_band = ema[i] * (1 - dynamic_k)
        ust_band = ema[i] * (1 + dynamic_k)
        
        # Histerizis mantığı
        if alt_band > result[i - 1]:
            raw_ars = alt_band
        elif ust_band < result[i - 1]:
            raw_ars = ust_band
        else:
            raw_ars = result[i - 1]
            
        # Dinamik Yuvarlama (IdealData uyumlu)
        # roundStep = Max(0.01, ATR * 0.1)
        round_step = max(0.01, atr[i] * 0.1)
        
        # Standart matematik yuvarlama (0.5'i yukarı yuvarla)
        # IdealData'nın Sistem.SayiYuvarla() ile uyumlu
        if round_step > 0:
            result[i] = math.floor(raw_ars / round_step + 0.5) * round_step
        else:
            result[i] = raw_ars
            
    return result


def Qstick(opens: List[float], closes: List[float], period: int = 8) -> List[float]:
    """
    Qstick Indicator
    Average of (Close - Open)
    """
    n = len(closes)
    diff = [closes[i] - opens[i] for i in range(n)]
    return SMA(diff, period)


def RVI(opens: List[float], highs: List[float], lows: List[float], 
        closes: List[float], period: int = 10) -> tuple:
    """Relative Vigor Index"""
    n = len(closes)
    numerator = [0.0] * n
    denominator = [0.0] * n
    
    for i in range(3, n):
        # Symmetric weighted moving average of (Close - Open)
        num = (closes[i] - opens[i]) + \
              2 * (closes[i-1] - opens[i-1]) + \
              2 * (closes[i-2] - opens[i-2]) + \
              (closes[i-3] - opens[i-3])
        numerator[i] = num / 6
        
        # Symmetric weighted moving average of (High - Low)
        den = (highs[i] - lows[i]) + \
              2 * (highs[i-1] - lows[i-1]) + \
              2 * (highs[i-2] - lows[i-2]) + \
              (highs[i-3] - lows[i-3])
        denominator[i] = den / 6
    
    # Sum over period
    rvi = [0.0] * n
    for i in range(period + 3, n):
        num_sum = sum(numerator[i - period + 1 : i + 1])
        den_sum = sum(denominator[i - period + 1 : i + 1])
        
        if den_sum != 0:
            rvi[i] = num_sum / den_sum
    
    
    # Signal line (symmetric weighted MA of RVI)
    signal = [0.0] * n
    for i in range(3, n):
        signal[i] = (rvi[i] + 2 * rvi[i-1] + 2 * rvi[i-2] + rvi[i-3]) / 6
    
    return rvi, signal


def NetLot(opens: List[float], highs: List[float], lows: List[float], 
           closes: List[float]) -> List[float]:
    """Net Lot Indicator (Buying/Selling Pressure)"""
    n = len(closes)
    result = [0.0] * n
    
    for i in range(n):
        range_hl = highs[i] - lows[i]
        if range_hl > 0:
            pressure = (closes[i] - opens[i]) / range_hl
            result[i] = pressure * 100
    
    return result


def ChaikinMoneyFlow(highs: List[float], lows: List[float], 
                     closes: List[float], volumes: List[float], 
                     period: int = 20) -> List[float]:
    """Chaikin Money Flow (CMF)"""
    n = len(closes)
    mf_multiplier = [0.0] * n
    mf_volume = [0.0] * n
    
    for i in range(n):
        hl = highs[i] - lows[i]
        if hl > 0:
            mf_multiplier[i] = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        mf_volume[i] = mf_multiplier[i] * volumes[i]
    
    result = [0.0] * n
    if n < period:
        return result
        
    for i in range(period - 1, n):
        sum_mf_vol = sum(mf_volume[i - period + 1 : i + 1])
        sum_vol = sum(volumes[i - period + 1 : i + 1])
        
        if sum_vol != 0:
            result[i] = sum_mf_vol / sum_vol
            
    return result


def QQEF(closes: List[float], rsi_period: int = 14, smooth_period: int = 5) -> tuple:
    """
    Quantitative Qualitative Estimation Filter (QQEF)
    Returns: (QQEF, QQES) tuple
    
    Kıvanç Özbilgiç QQE formülü kullanır:
    - QQEF: RSI'ın EMA ile smoothed hali
    - QQES: ATR-bazlı trailing band (Signal Line)
    """
    n = len(closes)
    
    # 1. Calculate RSI (Wilder's)
    rsi = RSI(closes, rsi_period)
    
    # 2. Smooth RSI (using EMA) -> QQEF
    qqef = EMA(rsi, smooth_period)
    
    # 3. Calculate QQES (Trailing Band - Kıvanç Özbilgiç formülü)
    # TR = |QQEF[i] - QQEF[i-1]|
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = abs(qqef[i] - qqef[i-1])
    
    # WWMA (Wilder's smoothing) - alpha = 1/period
    wwalpha = 1.0 / rsi_period
    wwma = [0.0] * n
    for i in range(1, n):
        wwma[i] = wwalpha * tr[i] + (1 - wwalpha) * wwma[i-1]
    
    # ATRRSI (Double smoothed)
    atrrsi = [0.0] * n
    for i in range(1, n):
        atrrsi[i] = wwalpha * wwma[i] + (1 - wwalpha) * atrrsi[i-1]
    
    # QQES trailing band logic
    mult = 4.236  # Standart QQE multiplier
    qqes = [0.0] * n
    qqes[0] = qqef[0] if n > 0 else 50.0
    
    for i in range(1, n):
        qup = qqef[i] + atrrsi[i] * mult
        qdn = qqef[i] - atrrsi[i] * mult
        
        prev_qqes = qqes[i-1]
        prev_qqef = qqef[i-1]
        curr_qqef = qqef[i]
        
        # Trailing logic (PineScript uyumlu)
        if qup < prev_qqes:
            qqes[i] = qup
        elif curr_qqef > prev_qqes and prev_qqef < prev_qqes:
            qqes[i] = qdn
        elif qdn > prev_qqes:
            qqes[i] = qdn
        elif curr_qqef < prev_qqes and prev_qqef > prev_qqes:
            qqes[i] = qup
        else:
            qqes[i] = prev_qqes
    
    return qqef, qqes
