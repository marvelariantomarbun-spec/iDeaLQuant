"""
IdealQuant - Moving Average Indicators
Extended MA types for IdealData compatibility
"""

from typing import List
from .core import SMA, EMA, WMA


def DEMA(data: List[float], period: int) -> List[float]:
    """
    Double Exponential Moving Average
    DEMA = 2 * EMA(data) - EMA(EMA(data))
    """
    # First EMA
    ema1 = EMA(data, period)
    
    # Second EMA: Apply on valid part of ema1 (excluding initial 0s)
    # EMA1 is valid starting from index (period - 1)
    valid_start_1 = period - 1
    if valid_start_1 >= len(data):
        return [0.0] * len(data)
        
    ema1_valid = ema1[valid_start_1:]
    ema2_sub = EMA(ema1_valid, period)
    
    # Pad result (valid_start_1 zeros + ema2_sub)
    ema2 = [0.0] * valid_start_1 + ema2_sub
    
    return [2 * ema1[i] - ema2[i] for i in range(len(data))]


def TEMA(data: List[float], period: int) -> List[float]:
    """
    Triple Exponential Moving Average
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    """
    # 1. EMA
    ema1 = EMA(data, period)
    valid_start_1 = period - 1
    if valid_start_1 >= len(data): return [0.0] * len(data)

    # 2. EMA(EMA)
    ema1_valid = ema1[valid_start_1:]
    ema2_sub = EMA(ema1_valid, period)
    ema2 = [0.0] * valid_start_1 + ema2_sub
    
    # 3. EMA(EMA(EMA))
    # EMA2 is valid starting from (period - 1) + (period - 1)
    valid_start_2 = valid_start_1 + (period - 1)
    if valid_start_2 >= len(data): return [0.0] * len(data)
    
    ema2_valid = ema2[valid_start_2:]
    ema3_sub = EMA(ema2_valid, period)
    ema3 = [0.0] * valid_start_2 + ema3_sub
    
    return [3 * ema1[i] - 3 * ema2[i] + ema3[i] for i in range(len(data))]


def KAMA(data: List[float], er_period: int = 10, 
         fast_period: int = 2, slow_period: int = 30) -> List[float]:
    """
    Kaufman Adaptive Moving Average
    Uses Efficiency Ratio to adjust smoothing constant
    """
    n = len(data)
    result = [0.0] * n
    
    if n < er_period + 1:
        return result
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # First KAMA value
    # First KAMA value: Use SMA for better stability
    result[er_period] = sum(data[:er_period+1]) / (er_period + 1)
    
    for i in range(er_period + 1, n):
        # Efficiency Ratio = |Change| / Volatility
        change = abs(data[i] - data[i - er_period])
        volatility = sum(abs(data[j] - data[j-1]) for j in range(i - er_period + 1, i + 1))
        
        if volatility != 0:
            er = change / volatility
        else:
            er = 0
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA
        result[i] = result[i-1] + sc * (data[i] - result[i-1])
    
    return result


def FRAMA(highs: List[float], lows: List[float], closes: List[float], period: int = 16) -> List[float]:
    """
    Fractal Adaptive Moving Average
    Uses fractal dimension to adjust alpha.
    Standard Ehlers: Uses High/Low for dimension, Close for values.
    """
    import math
    
    n = len(closes)
    result = [0.0] * n
    half = period // 2
    
    if n < period:
        return result
    
    result[period - 1] = sum(closes[:period]) / period
    
    for i in range(period, n):
        # Calculate fractal dimension using High/Low
        # N1 = (Highest(High, half) - Lowest(Low, half)) / half
        # window 1: [i-half : i]
        h1 = max(highs[i-half : i])
        l1 = min(lows[i-half : i])
        n1 = (h1 - l1) / half
        
        # window 2: [i-period : i-half]
        h2 = max(highs[i-period : i-half])
        l2 = min(lows[i-period : i-half])
        n2 = (h2 - l2) / half
        
        # window 3: [i-period : i]
        h3 = max(highs[i-period : i])
        l3 = min(lows[i-period : i])
        n3 = (h3 - l3) / period
        
        if n1 + n2 > 0 and n3 > 0:
            d = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
        else:
            d = 0 # If invalid, assume Dimension 0 (alpha=Small)? Or D=1?
            # If flat, D=0 (Line). D=1 is random walk?
            # Ehlers: D should be between 1 and 2.
            # If (n1+n2) > n3, D > 1.
            # If flat, n1=0, n2=0, n3=0.
            d = 1
        
        # Alpha based on fractal dimension
        alpha = math.exp(-4.6 * (d - 1))
        alpha = max(0.01, min(1, alpha))
        
        result[i] = alpha * closes[i] + (1 - alpha) * result[i-1]
    
    return result


def WWMA(data: List[float], period: int) -> List[float]:
    """
    Welles Wilder Moving Average (same as RMA)
    Alpha = 1 / period
    """
    n = len(data)
    result = [0.0] * n
    
    if n < period:
        return result
    
    alpha = 1.0 / period
    result[period - 1] = sum(data[:period]) / period
    
    for i in range(period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result


def SMMA(data: List[float], period: int) -> List[float]:
    """
    Smoothed Moving Average
    Same as WWMA/RMA
    """
    return WWMA(data, period)


def ZLEMA(data: List[float], period: int) -> List[float]:
    """
    Zero Lag Exponential Moving Average
    Reduces lag by using de-lagged data
    """
    lag = (period - 1) // 2
    n = len(data)
    
    # De-lag the data
    delagged = [0.0] * n
    for i in range(lag, n):
        delagged[i] = 2 * data[i] - data[i - lag]
    
    return EMA(delagged, period)


def T3(data: List[float], period: int, v_factor: float = 0.7) -> List[float]:
    """
    T3 Moving Average (Tillson)
    Triple smoothed EMA with volume factor
    """
    c1 = -v_factor ** 3
    c2 = 3 * v_factor ** 2 + 3 * v_factor ** 3
    c3 = -6 * v_factor ** 2 - 3 * v_factor - 3 * v_factor ** 3
    c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2
    
    e1 = EMA(data, period)
    e2 = EMA(e1, period)
    e3 = EMA(e2, period)
    e4 = EMA(e3, period)
    e5 = EMA(e4, period)
    e6 = EMA(e5, period)
    
    n = len(data)
    return [c1*e6[i] + c2*e5[i] + c3*e4[i] + c4*e3[i] for i in range(n)]
