"""
IdealQuant - Volume Indicators
OBV, Klinger, NVI, PVI, PVT, ADL, ChaikinOsc
"""

from typing import List
from .core import EMA, SMA


def OBV(closes: List[float], volumes: List[float]) -> List[float]:
    """
    On Balance Volume
    Cumulative volume based on price direction
    """
    n = len(closes)
    result = [0.0] * n
    
    if n == 0:
        return result
    
    result[0] = volumes[0]
    
    for i in range(1, n):
        if closes[i] > closes[i-1]:
            result[i] = result[i-1] + volumes[i]
        elif closes[i] < closes[i-1]:
            result[i] = result[i-1] - volumes[i]
        else:
            result[i] = result[i-1]
    
    return result


def PVT(closes: List[float], volumes: List[float]) -> List[float]:
    """
    Price Volume Trend
    Like OBV but weighted by price change percentage
    """
    n = len(closes)
    result = [0.0] * n
    
    for i in range(1, n):
        if closes[i-1] != 0:
            pct_change = (closes[i] - closes[i-1]) / closes[i-1]
            result[i] = result[i-1] + volumes[i] * pct_change
        else:
            result[i] = result[i-1]
    
    return result


def ADL(highs: List[float], lows: List[float], closes: List[float], 
        volumes: List[float]) -> List[float]:
    """
    Accumulation/Distribution Line
    """
    n = len(closes)
    result = [0.0] * n
    
    for i in range(n):
        hl = highs[i] - lows[i]
        if hl != 0:
            # Money Flow Multiplier
            mf_mult = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
            # Money Flow Volume
            mf_vol = mf_mult * volumes[i]
            
            if i > 0:
                result[i] = result[i-1] + mf_vol
            else:
                result[i] = mf_vol
        else:
            result[i] = result[i-1] if i > 0 else 0
    
    return result


def ChaikinOsc(highs: List[float], lows: List[float], closes: List[float], 
               volumes: List[float], fast: int = 3, slow: int = 10) -> List[float]:
    """
    Chaikin Oscillator
    EMA(ADL, fast) - EMA(ADL, slow)
    """
    adl = ADL(highs, lows, closes, volumes)
    
    ema_fast = EMA(adl, fast)
    ema_slow = EMA(adl, slow)
    
    return [ema_fast[i] - ema_slow[i] for i in range(len(closes))]


def NVI(closes: List[float], volumes: List[float]) -> List[float]:
    """
    Negative Volume Index
    Changes only on down-volume days
    """
    n = len(closes)
    result = [1000.0] * n  # Start at 1000
    
    for i in range(1, n):
        if volumes[i] < volumes[i-1]:
            # Down volume day - update NVI
            if closes[i-1] != 0:
                pct_change = (closes[i] - closes[i-1]) / closes[i-1]
                result[i] = result[i-1] * (1 + pct_change)
            else:
                result[i] = result[i-1]
        else:
            result[i] = result[i-1]
    
    return result


def PVI(closes: List[float], volumes: List[float]) -> List[float]:
    """
    Positive Volume Index
    Changes only on up-volume days
    """
    n = len(closes)
    result = [1000.0] * n  # Start at 1000
    
    for i in range(1, n):
        if volumes[i] > volumes[i-1]:
            # Up volume day - update PVI
            if closes[i-1] != 0:
                pct_change = (closes[i] - closes[i-1]) / closes[i-1]
                result[i] = result[i-1] * (1 + pct_change)
            else:
                result[i] = result[i-1]
        else:
            result[i] = result[i-1]
    
    return result


def KlingerOsc(highs: List[float], lows: List[float], closes: List[float],
               volumes: List[float], fast: int = 34, slow: int = 55) -> List[float]:
    """
    Klinger Volume Oscillator
    Measures volume flow
    """
    n = len(closes)
    
    # Typical Price
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(n)]
    
    # Trend direction
    trend = [0.0] * n
    for i in range(1, n):
        if tp[i] > tp[i-1]:
            trend[i] = 1
        elif tp[i] < tp[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    
    # dm (price range)
    dm = [highs[i] - lows[i] for i in range(n)]
    
    # cm (cumulative movement)
    cm = [0.0] * n
    for i in range(1, n):
        if trend[i] == trend[i-1]:
            cm[i] = cm[i-1] + dm[i]
        else:
            cm[i] = dm[i-1] + dm[i]
    
    # Volume Force
    vf = [0.0] * n
    for i in range(n):
        if cm[i] != 0:
            vf[i] = volumes[i] * abs(2 * (dm[i] / cm[i]) - 1) * trend[i] * 100
    
    # EMA of VF
    ema_fast = EMA(vf, fast)
    ema_slow = EMA(vf, slow)
    
    return [ema_fast[i] - ema_slow[i] for i in range(n)]


def MassIndex(highs: List[float], lows: List[float], 
              period: int = 9, sum_period: int = 25) -> List[float]:
    """
    Mass Index
    Identifies trend reversals based on range expansion
    """
    n = len(highs)
    result = [0.0] * n
    
    # Range
    diff = [highs[i] - lows[i] for i in range(n)]
    
    # EMA of range
    ema1 = EMA(diff, period)
    ema2 = EMA(ema1, period)
    
    # Ratio
    ratio = [ema1[i] / ema2[i] if ema2[i] != 0 else 0 for i in range(n)]
    
    # Sum of ratio
    for i in range(sum_period - 1, n):
        result[i] = sum(ratio[i - sum_period + 1 : i + 1])
    
    return result


def EaseOfMovement(highs: List[float], lows: List[float], 
                   volumes: List[float], period: int = 14) -> List[float]:
    """
    Ease of Movement
    Volume-weighted price movement
    """
    n = len(highs)
    emv = [0.0] * n
    
    for i in range(1, n):
        dm = ((highs[i] + lows[i]) / 2) - ((highs[i-1] + lows[i-1]) / 2)
        box_ratio = (volumes[i] / 10000) / (highs[i] - lows[i]) if (highs[i] - lows[i]) != 0 else 0
        
        if box_ratio != 0:
            emv[i] = dm / box_ratio
    
    return SMA(emv, period)


def ForceIndex(closes: List[float], volumes: List[float], 
               period: int = 13) -> List[float]:
    """
    Force Index
    Price change * Volume
    """
    n = len(closes)
    fi = [0.0] * n
    
    for i in range(1, n):
        fi[i] = (closes[i] - closes[i-1]) * volumes[i]
    
    return EMA(fi, period)
