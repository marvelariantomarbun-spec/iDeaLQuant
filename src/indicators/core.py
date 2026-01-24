"""
IdealQuant - Technical Indicators
IdealData compatible implementations
"""

import numpy as np
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
    """Relative Strength Index - IdealData compatible"""
    result = [50.0] * len(closes)  # Default to 50 (neutral)
    if len(closes) < period + 1:
        return result
    
    # Calculate initial average gain/loss
    avg_gain = 0.0
    avg_loss = 0.0
    
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0:
            avg_gain += change
        else:
            avg_loss += abs(change)
    
    avg_gain /= period
    avg_loss /= period
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Subsequent values using Wilder smoothing
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        
        if change > 0:
            current_gain = change
            current_loss = 0.0
        else:
            current_gain = 0.0
            current_loss = abs(change)
        
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
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
    
    # First ATR is simple average
    if n >= period:
        result[period - 1] = sum(tr[:period]) / period
        
        # Subsequent ATRs using Wilder smoothing
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    
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
    """Average Directional Index - IdealData compatible"""
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
    
    # Smooth with Wilder's method
    smoothed_plus_dm = [0.0] * n
    smoothed_minus_dm = [0.0] * n
    smoothed_tr = [0.0] * n
    
    if n > period:
        smoothed_plus_dm[period] = sum(plus_dm[1:period + 1])
        smoothed_minus_dm[period] = sum(minus_dm[1:period + 1])
        smoothed_tr[period] = sum(tr[1:period + 1])
        
        for i in range(period + 1, n):
            smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period) + minus_dm[i]
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period) + tr[i]
    
    # Calculate DX
    dx = [0.0] * n
    for i in range(period, n):
        if smoothed_tr[i] != 0:
            plus_di = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
            
            di_sum = plus_di + minus_di
            if di_sum != 0:
                dx[i] = 100 * abs(plus_di - minus_di) / di_sum
    
    # Smooth DX to get ADX
    if n >= 2 * period:
        result[2 * period - 1] = sum(dx[period:2 * period]) / period
        
        for i in range(2 * period, n):
            result[i] = (result[i - 1] * (period - 1) + dx[i]) / period
    
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
    """
    n = len(typical)
    result = [0.0] * n
    
    # Calculate EMA and ATR
    ema = EMA(typical, ema_period)
    atr = ATR(highs, lows, closes, atr_period)
    
    for i in range(1, n):
        # Dynamic K based on ATR
        if ema[i] != 0:
            dynamic_k = (atr[i] / ema[i]) * atr_mult
            dynamic_k = max(min_k, min(max_k, dynamic_k))
        else:
            dynamic_k = min_k
        
        alt_band = ema[i] * (1 - dynamic_k)
        ust_band = ema[i] * (1 + dynamic_k)
        
        if alt_band > result[i - 1]:
            result[i] = alt_band
        elif ust_band < result[i - 1]:
            result[i] = ust_band
        else:
            result[i] = result[i - 1]
    
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
    """
    Relative Vigor Index
    Returns: (RVI, Signal) tuple
    """
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
