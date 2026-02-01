"""
IdealQuant - Oscillator Indicators
RSI variants, MACD, CCI, Stochastic variants, Williams %R
"""

from typing import List, Tuple
from .core import SMA, EMA, RSI, HHV, LLV


def CCI(highs: List[float], lows: List[float], closes: List[float], 
        period: int = 20) -> List[float]:
    """
    Commodity Channel Index
    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    """
    n = len(closes)
    result = [0.0] * n
    
    # Typical Price
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(n)]
    
    # SMA of TP
    sma_tp = SMA(tp, period)
    
    for i in range(period - 1, n):
        # Mean Deviation
        window = tp[i - period + 1 : i + 1]
        mean = sma_tp[i]
        mean_dev = sum(abs(x - mean) for x in window) / period
        
        if mean_dev != 0:
            result[i] = (tp[i] - sma_tp[i]) / (0.015 * mean_dev)
    
    return result


def MACD(data: List[float], fast: int = 12, slow: int = 26, 
         signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """
    Moving Average Convergence Divergence
    Returns: (macd_line, signal_line, histogram)
    """
    n = len(data)
    
    ema_fast = EMA(data, fast)
    ema_slow = EMA(data, slow)
    
    # MACD Line
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(n)]
    
    # Signal Line
    signal_line = EMA(macd_line, signal)
    
    # Histogram
    histogram = [macd_line[i] - signal_line[i] for i in range(n)]
    
    return macd_line, signal_line, histogram


def StochRSI(closes: List[float], rsi_period: int = 14, 
             stoch_period: int = 14, k_smooth: int = 3, 
             d_smooth: int = 3) -> Tuple[List[float], List[float]]:
    """
    Stochastic RSI
    StochRSI = (RSI - LLV(RSI)) / (HHV(RSI) - LLV(RSI))
    Returns: (K, D)
    """
    n = len(closes)
    
    # Calculate RSI
    rsi = RSI(closes, rsi_period)
    
    # Stochastic of RSI
    highest_rsi = HHV(rsi, stoch_period)
    lowest_rsi = LLV(rsi, stoch_period)
    
    stoch_rsi = [0.0] * n
    for i in range(n):
        diff = highest_rsi[i] - lowest_rsi[i]
        if diff != 0:
            stoch_rsi[i] = ((rsi[i] - lowest_rsi[i]) / diff) * 100
    
    # Smooth K
    k = SMA(stoch_rsi, k_smooth)
    
    # D is SMA of K
    d = SMA(k, d_smooth)
    
    return k, d


def WilliamsR(highs: List[float], lows: List[float], closes: List[float], 
              period: int = 14) -> List[float]:
    """
    Williams %R
    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    """
    n = len(closes)
    result = [-50.0] * n
    
    for i in range(period - 1, n):
        highest = max(highs[i - period + 1 : i + 1])
        lowest = min(lows[i - period + 1 : i + 1])
        
        if highest != lowest:
            result[i] = ((highest - closes[i]) / (highest - lowest)) * -100
    
    return result


def ROC(data: List[float], period: int = 10) -> List[float]:
    """
    Rate of Change (Percent)
    ROC = ((Close - Close[n]) / Close[n]) * 100
    """
    n = len(data)
    result = [0.0] * n
    
    for i in range(period, n):
        if data[i - period] != 0:
            result[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
    
    return result


def UltimateOscillator(highs: List[float], lows: List[float], closes: List[float],
                       period1: int = 7, period2: int = 14, 
                       period3: int = 28) -> List[float]:
    """
    Ultimate Oscillator
    Combines three timeframes with weights
    """
    n = len(closes)
    result = [50.0] * n
    
    # Calculate BP and TR
    bp = [0.0] * n
    tr = [0.0] * n
    
    for i in range(1, n):
        bp[i] = closes[i] - min(lows[i], closes[i-1])
        tr[i] = max(highs[i], closes[i-1]) - min(lows[i], closes[i-1])
    
    # Calculate averages for each period
    for i in range(period3, n):
        bp1 = sum(bp[i - period1 + 1 : i + 1])
        tr1 = sum(tr[i - period1 + 1 : i + 1])
        
        bp2 = sum(bp[i - period2 + 1 : i + 1])
        tr2 = sum(tr[i - period2 + 1 : i + 1])
        
        bp3 = sum(bp[i - period3 + 1 : i + 1])
        tr3 = sum(tr[i - period3 + 1 : i + 1])
        
        avg1 = bp1 / tr1 if tr1 != 0 else 0
        avg2 = bp2 / tr2 if tr2 != 0 else 0
        avg3 = bp3 / tr3 if tr3 != 0 else 0
        
        # Weighted average (4:2:1)
        result[i] = ((avg1 * 4 + avg2 * 2 + avg3) / 7) * 100
    
    return result


def TRIX(data: List[float], period: int = 15) -> List[float]:
    """
    Triple Smoothed EMA Rate of Change
    TRIX = ROC(EMA(EMA(EMA(data))))
    """
    n = len(data)
    
    # Triple EMA
    ema1 = EMA(data, period)
    ema2 = EMA(ema1, period)
    ema3 = EMA(ema2, period)
    
    # 1-period Rate of Change * 100
    result = [0.0] * n
    for i in range(1, n):
        if ema3[i-1] != 0:
            result[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 100
    
    return result


def DPO(data: List[float], period: int = 20) -> List[float]:
    """
    Detrended Price Oscillator
    DPO = Close - SMA(Close, period / 2 + 1 periods ago)
    """
    n = len(data)
    result = [0.0] * n
    
    sma = SMA(data, period)
    shift = period // 2 + 1
    
    for i in range(shift, n):
        result[i] = data[i] - sma[i - shift] if i >= shift else 0
    
    return result


def ChandeMomentum(data: List[float], period: int = 9) -> List[float]:
    """
    Chande Momentum Oscillator (CMO)
    CMO = (Sum(Up) - Sum(Down)) / (Sum(Up) + Sum(Down)) * 100
    """
    n = len(data)
    result = [0.0] * n
    
    # Calculate up and down moves
    up = [0.0] * n
    down = [0.0] * n
    
    for i in range(1, n):
        diff = data[i] - data[i-1]
        if diff > 0:
            up[i] = diff
        else:
            down[i] = abs(diff)
    
    for i in range(period, n):
        sum_up = sum(up[i - period + 1 : i + 1])
        sum_down = sum(down[i - period + 1 : i + 1])
        
        total = sum_up + sum_down
        if total != 0:
            result[i] = ((sum_up - sum_down) / total) * 100
    
    return result


def RMI(data: List[float], mom_period: int = 4, rsi_period: int = 14) -> List[float]:
    """
    Relative Momentum Index
    Like RSI but uses momentum instead of price change
    """
    n = len(data)
    result = [50.0] * n
    
    if n <= rsi_period + mom_period:
        return result
    
    # Momentum differences
    up = [0.0] * n
    down = [0.0] * n
    
    for i in range(mom_period, n):
        diff = data[i] - data[i - mom_period]
        if diff > 0:
            up[i] = diff
        else:
            down[i] = abs(diff)
    
    # RMA smoothing
    avg_up = [0.0] * n
    avg_down = [0.0] * n
    
    start = rsi_period + mom_period
    if start >= n:
        return result
    
    avg_up[start] = sum(up[mom_period:start+1]) / rsi_period
    avg_down[start] = sum(down[mom_period:start+1]) / rsi_period
    
    for i in range(start + 1, n):
        avg_up[i] = (avg_up[i-1] * (rsi_period - 1) + up[i]) / rsi_period
        avg_down[i] = (avg_down[i-1] * (rsi_period - 1) + down[i]) / rsi_period
        
        if avg_down[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_up[i] / avg_down[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def AwesomeOscillator(highs: List[float], lows: List[float], 
                      fast: int = 5, slow: int = 34) -> List[float]:
    """
    Awesome Oscillator
    AO = SMA(Median, fast) - SMA(Median, slow)
    """
    n = len(highs)
    
    # Median Price
    median = [(highs[i] + lows[i]) / 2 for i in range(n)]
    
    sma_fast = SMA(median, fast)
    sma_slow = SMA(median, slow)
    
    return [sma_fast[i] - sma_slow[i] for i in range(n)]


def ElliotWaveOscillator(highs: List[float], lows: List[float],
                         fast: int = 5, slow: int = 35) -> List[float]:
    """
    Elliot Wave Oscillator (EWO)
    Same as Awesome Oscillator but with different default periods
    """
    return AwesomeOscillator(highs, lows, fast, slow)
