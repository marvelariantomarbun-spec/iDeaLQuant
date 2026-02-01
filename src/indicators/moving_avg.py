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
    ema1 = EMA(data, period)
    ema2 = EMA(ema1, period)
    
    return [2 * ema1[i] - ema2[i] for i in range(len(data))]


def TEMA(data: List[float], period: int) -> List[float]:
    """
    Triple Exponential Moving Average
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    """
    ema1 = EMA(data, period)
    ema2 = EMA(ema1, period)
    ema3 = EMA(ema2, period)
    
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
    result[er_period] = data[er_period]
    
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


def FRAMA(data: List[float], period: int = 16) -> List[float]:
    """
    Fractal Adaptive Moving Average
    Uses fractal dimension to adjust alpha
    """
    import math
    
    n = len(data)
    result = [0.0] * n
    half = period // 2
    
    if n < period:
        return result
    
    result[period - 1] = sum(data[:period]) / period
    
    for i in range(period, n):
        # Calculate fractal dimension
        n1 = (max(data[i-half:i]) - min(data[i-half:i])) / half
        n2 = (max(data[i-period:i-half]) - min(data[i-period:i-half])) / half
        n3 = (max(data[i-period:i]) - min(data[i-period:i])) / period
        
        if n1 + n2 > 0 and n3 > 0:
            d = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
        else:
            d = 1
        
        # Alpha based on fractal dimension
        alpha = math.exp(-4.6 * (d - 1))
        alpha = max(0.01, min(1, alpha))
        
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
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
