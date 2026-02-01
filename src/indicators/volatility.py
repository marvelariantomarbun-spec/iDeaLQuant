"""
IdealQuant - Volatility Indicators
ATR variants, Bollinger, Keltner, Envelope, etc.
"""

from typing import List, Tuple
import math
from .core import SMA, EMA, ATR


def BollingerUp(closes: List[float], period: int = 20, 
                deviation: float = 2.0) -> List[float]:
    """Bollinger Upper Band"""
    n = len(closes)
    result = [0.0] * n
    
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        result[i] = mean + deviation * std
    
    return result


def BollingerDown(closes: List[float], period: int = 20, 
                  deviation: float = 2.0) -> List[float]:
    """Bollinger Lower Band"""
    n = len(closes)
    result = [0.0] * n
    
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        result[i] = mean - deviation * std
    
    return result


def BollingerMid(closes: List[float], period: int = 20) -> List[float]:
    """Bollinger Middle Band (SMA)"""
    return SMA(closes, period)


def BollingerWidth(closes: List[float], period: int = 20, 
                   deviation: float = 2.0) -> List[float]:
    """Bollinger Band Width"""
    n = len(closes)
    result = [0.0] * n
    
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        
        if mean != 0:
            result[i] = (4 * deviation * std) / mean * 100
    
    return result


def BollingerPercentB(closes: List[float], period: int = 20, 
                       deviation: float = 2.0) -> List[float]:
    """Bollinger %B - Position within bands"""
    upper = BollingerUp(closes, period, deviation)
    lower = BollingerDown(closes, period, deviation)
    
    n = len(closes)
    result = [50.0] * n
    
    for i in range(n):
        diff = upper[i] - lower[i]
        if diff != 0:
            result[i] = ((closes[i] - lower[i]) / diff) * 100
    
    return result


def KeltnerUp(highs: List[float], lows: List[float], closes: List[float],
              ema_period: int = 20, atr_period: int = 10, 
              multiplier: float = 2.0) -> List[float]:
    """Keltner Channel Upper Band"""
    ema = EMA(closes, ema_period)
    atr = ATR(highs, lows, closes, atr_period)
    
    return [ema[i] + multiplier * atr[i] for i in range(len(closes))]


def KeltnerDown(highs: List[float], lows: List[float], closes: List[float],
                ema_period: int = 20, atr_period: int = 10, 
                multiplier: float = 2.0) -> List[float]:
    """Keltner Channel Lower Band"""
    ema = EMA(closes, ema_period)
    atr = ATR(highs, lows, closes, atr_period)
    
    return [ema[i] - multiplier * atr[i] for i in range(len(closes))]


def KeltnerChannel(highs: List[float], lows: List[float], closes: List[float],
                   ema_period: int = 20, atr_period: int = 10,
                   multiplier: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Keltner Channel
    Returns: (upper, middle, lower)
    """
    ema = EMA(closes, ema_period)
    atr = ATR(highs, lows, closes, atr_period)
    
    upper = [ema[i] + multiplier * atr[i] for i in range(len(closes))]
    lower = [ema[i] - multiplier * atr[i] for i in range(len(closes))]
    
    return upper, ema, lower


def EnvelopeUp(data: List[float], period: int = 20, 
               percent: float = 2.5) -> List[float]:
    """Envelope Upper Band"""
    ma = SMA(data, period)
    return [ma[i] * (1 + percent / 100) for i in range(len(data))]


def EnvelopeDown(data: List[float], period: int = 20, 
                 percent: float = 2.5) -> List[float]:
    """Envelope Lower Band"""
    ma = SMA(data, period)
    return [ma[i] * (1 - percent / 100) for i in range(len(data))]


def EnvelopeMid(data: List[float], period: int = 20) -> List[float]:
    """Envelope Middle Band"""
    return SMA(data, period)


def StandardDeviation(data: List[float], period: int = 20) -> List[float]:
    """Standard Deviation"""
    n = len(data)
    result = [0.0] * n
    
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        mean = sum(window) / period
        variance = sum((x - mean) ** 2 for x in window) / period
        result[i] = math.sqrt(variance)
    
    return result


def ChaikinVolatility(highs: List[float], lows: List[float], 
                      ema_period: int = 10, roc_period: int = 10) -> List[float]:
    """
    Chaikin Volatility
    Rate of change of high-low range EMA
    """
    n = len(highs)
    
    # High-Low range
    hl = [highs[i] - lows[i] for i in range(n)]
    
    # EMA of range
    ema = EMA(hl, ema_period)
    
    # ROC of EMA
    result = [0.0] * n
    for i in range(roc_period, n):
        if ema[i - roc_period] != 0:
            result[i] = ((ema[i] - ema[i - roc_period]) / ema[i - roc_period]) * 100
    
    return result


def TrueRange(highs: List[float], lows: List[float], 
              closes: List[float]) -> List[float]:
    """True Range (single calculation, no smoothing)"""
    n = len(closes)
    result = [0.0] * n
    
    result[0] = highs[0] - lows[0]
    
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        result[i] = max(hl, hc, lc)
    
    return result


def NATR(highs: List[float], lows: List[float], closes: List[float],
         period: int = 14) -> List[float]:
    """
    Normalized ATR (percentage of close)
    """
    atr = ATR(highs, lows, closes, period)
    n = len(closes)
    
    result = [0.0] * n
    for i in range(n):
        if closes[i] != 0:
            result[i] = (atr[i] / closes[i]) * 100
    
    return result
