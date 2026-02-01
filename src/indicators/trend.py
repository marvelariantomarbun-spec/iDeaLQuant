"""
IdealQuant - Trend Indicators
ADX components, Aroon, Parabolic SAR, Ichimoku
"""

from typing import List, Tuple, NamedTuple
from .core import EMA, RMA, ATR, HHV, LLV


def DirectionalIndicatorPlus(highs: List[float], lows: List[float], 
                              closes: List[float], period: int = 14) -> List[float]:
    """
    Directional Indicator Plus (+DI)
    """
    n = len(closes)
    result = [0.0] * n
    
    if n < period + 1:
        return result
    
    # Calculate +DM and TR
    plus_dm = [0.0] * n
    tr = [0.0] * n
    
    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # Smooth with Wilder's method
    smoothed_dm = RMA(plus_dm, period)
    smoothed_tr = RMA(tr, period)
    
    for i in range(period, n):
        if smoothed_tr[i] != 0:
            result[i] = (smoothed_dm[i] / smoothed_tr[i]) * 100
    
    return result


def DirectionalIndicatorMinus(highs: List[float], lows: List[float], 
                               closes: List[float], period: int = 14) -> List[float]:
    """
    Directional Indicator Minus (-DI)
    """
    n = len(closes)
    result = [0.0] * n
    
    if n < period + 1:
        return result
    
    # Calculate -DM and TR
    minus_dm = [0.0] * n
    tr = [0.0] * n
    
    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    
    # Smooth with Wilder's method
    smoothed_dm = RMA(minus_dm, period)
    smoothed_tr = RMA(tr, period)
    
    for i in range(period, n):
        if smoothed_tr[i] != 0:
            result[i] = (smoothed_dm[i] / smoothed_tr[i]) * 100
    
    return result


# Alias
DI_Plus = DirectionalIndicatorPlus
DI_Minus = DirectionalIndicatorMinus


def AroonUp(highs: List[float], period: int = 25) -> List[float]:
    """
    Aroon Up
    Measures bars since highest high
    """
    n = len(highs)
    result = [0.0] * n

    window_len = period + 1
    for i in range(window_len - 1, n):
        window = highs[i - window_len + 1 : i + 1]
        highest = max(window)

        # Tie-break: en son görülen (IdealData davranışına daha yakın)
        highest_idx = len(window) - 1 - list(reversed(window)).index(highest)

        # Bars since highest -> AroonUp
        bars_since = (window_len - 1) - highest_idx
        result[i] = ((period - bars_since) / period) * 100
    
    return result


def AroonDown(lows: List[float], period: int = 25) -> List[float]:
    """
    Aroon Down
    Measures bars since lowest low
    """
    n = len(lows)
    result = [0.0] * n

    window_len = period + 1
    for i in range(window_len - 1, n):
        window = lows[i - window_len + 1 : i + 1]
        lowest = min(window)

        # Tie-break: en son görülen
        lowest_idx = len(window) - 1 - list(reversed(window)).index(lowest)

        bars_since = (window_len - 1) - lowest_idx
        result[i] = ((period - bars_since) / period) * 100
    
    return result


def AroonOsc(highs: List[float], lows: List[float], 
             period: int = 25) -> List[float]:
    """
    Aroon Oscillator
    AroonOsc = AroonUp - AroonDown
    """
    up = AroonUp(highs, period)
    down = AroonDown(lows, period)
    
    return [up[i] - down[i] for i in range(len(highs))]


def Aroon(highs: List[float], lows: List[float], 
          period: int = 25) -> Tuple[List[float], List[float], List[float]]:
    """
    Complete Aroon indicator
    Returns: (AroonUp, AroonDown, AroonOsc)
    """
    up = AroonUp(highs, period)
    down = AroonDown(lows, period)
    osc = [up[i] - down[i] for i in range(len(highs))]
    
    return up, down, osc


def ParabolicSAR(highs: List[float], lows: List[float], 
                 af_start: float = 0.02, af_step: float = 0.02, 
                 af_max: float = 0.20) -> List[float]:
    """
    Parabolic SAR
    """
    n = len(highs)
    result = [0.0] * n
    
    if n < 2:
        return result
    
    # Initialize
    # Determine initial trend based on first bar
    # If Close > Open (or first 2 bars up), start Long
    # Simple check: Compare High[0] and Low[0] to reference? 
    # IdealData: usually checks if C[0] > C[1] but we only have current.
    # Let's assume Long if Close[0] > Open[0] (need opens?)
    # Alternative: start with Long=True if H[1] > H[0]... wait, loops starts at 1.
    
    is_long = True
    if n > 1:
        if highs[1] < highs[0] and lows[1] < lows[0]:
            is_long = False
    
    if is_long:
        sar = lows[0]
        ep = highs[0]
    else:
        sar = highs[0]
        ep = lows[0]
        
    af = af_start
    result[0] = sar  # SAR for *next* bar effectively, or current? SAR is usually plotted 'stops' for current bar.
    # IdealData: Standard SAR logic.
    
    # Standard SAR often skips the first value or sets it to min/max.
    
    for i in range(1, n):
        prev_sar = sar
        
        # Calculate new SAR
        sar = prev_sar + af * (ep - prev_sar)
        
        if is_long:
            # Ensure SAR is below prior two lows
            sar = min(sar, lows[i-1])
            if i >= 2:
                sar = min(sar, lows[i-2])
            
            # Check for reversal
            if lows[i] < sar:
                is_long = False
                sar = ep
                ep = lows[i]
                af = af_start
            else:
                # Update EP and AF
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            # Ensure SAR is above prior two highs
            sar = max(sar, highs[i-1])
            if i >= 2:
                sar = max(sar, highs[i-2])
            
            # Check for reversal
            if highs[i] > sar:
                is_long = True
                sar = ep
                ep = highs[i]
                af = af_start
            else:
                # Update EP and AF
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)
        
        result[i] = sar
    
    return result


class IchimokuResult(NamedTuple):
    """Ichimoku Cloud components"""
    tenkan: List[float]      # Conversion Line (9)
    kijun: List[float]       # Base Line (26)
    senkou_a: List[float]    # Leading Span A
    senkou_b: List[float]    # Leading Span B
    chikou: List[float]      # Lagging Span


def Ichimoku(highs: List[float], lows: List[float], closes: List[float],
             tenkan_period: int = 9, kijun_period: int = 26, 
             senkou_b_period: int = 52) -> IchimokuResult:
    """
    Ichimoku Kinko Hyo
    Returns all 5 components
    """
    n = len(closes)
    
    def midpoint(data_h: List[float], data_l: List[float], period: int) -> List[float]:
        result = [0.0] * len(data_h)
        for i in range(period - 1, len(data_h)):
            highest = max(data_h[i - period + 1 : i + 1])
            lowest = min(data_l[i - period + 1 : i + 1])
            result[i] = (highest + lowest) / 2
        return result
    
    # Tenkan-sen (Conversion Line)
    tenkan = midpoint(highs, lows, tenkan_period)
    
    # Kijun-sen (Base Line)
    kijun = midpoint(highs, lows, kijun_period)
    
    # Senkou Span A (Leading Span A) - shifted forward 26 periods
    senkou_a = [0.0] * n
    for i in range(n - kijun_period):
        senkou_a[i + kijun_period] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B) - shifted forward 26 periods
    senkou_b_raw = midpoint(highs, lows, senkou_b_period)
    senkou_b = [0.0] * n
    for i in range(n - kijun_period):
        senkou_b[i + kijun_period] = senkou_b_raw[i]
    
    # Chikou Span (Lagging Span) - shifted back 26 periods
    chikou = [0.0] * n
    for i in range(kijun_period, n):
        chikou[i - kijun_period] = closes[i]
    
    return IchimokuResult(tenkan, kijun, senkou_a, senkou_b, chikou)


def PriceChannelUp(highs: List[float], period: int = 20) -> List[float]:
    """
    Price Channel Upper Band (Donchian Channel)
    Usually excludes current bar for breakout logic.
    """
    hhv = HHV(highs, period)
    # Shift right by 1 to exclude current bar
    return [0.0] + hhv[:-1]


def PriceChannelDown(lows: List[float], period: int = 20) -> List[float]:
    """
    Price Channel Lower Band (Donchian Channel)
    """
    llv = LLV(lows, period)
    return [0.0] + llv[:-1]


def PriceChannel(highs: List[float], lows: List[float], 
                 period: int = 20) -> Tuple[List[float], List[float], List[float]]:
    """
    Price Channel (Donchian Channel)
    Returns: (upper, middle, lower)
    """
    upper = HHV(highs, period)
    lower = LLV(lows, period)
    middle = [(upper[i] + lower[i]) / 2 for i in range(len(highs))]
    
    return upper, middle, lower


def VHF(closes: List[float], period: int = 28) -> List[float]:
    """
    Vertical Horizontal Filter
    Measures trend strength
    """
    n = len(closes)
    result = [0.0] * n
    
    for i in range(period, n):
        highest = max(closes[i - period + 1 : i + 1])
        lowest = min(closes[i - period + 1 : i + 1])
        
        # Sum of absolute changes
        sum_changes = sum(abs(closes[j] - closes[j-1]) 
                          for j in range(i - period + 2, i + 1))
        
        if sum_changes != 0:
            result[i] = abs(highest - lowest) / sum_changes
    
    return result


def LinearReg(data: List[float], period: int = 14) -> List[float]:
    """
    Linear Regression Value
    """
    n = len(data)
    result = [0.0] * n
    
    for i in range(period - 1, n):
        # Y values
        y = data[i - period + 1 : i + 1]
        
        # X values (0, 1, 2, ..., period-1)
        x_sum = period * (period - 1) / 2
        x2_sum = period * (period - 1) * (2 * period - 1) / 6
        y_sum = sum(y)
        xy_sum = sum(j * y[j] for j in range(period))
        
        # Calculate slope and intercept
        denom = period * x2_sum - x_sum * x_sum
        if denom != 0:
            slope = (period * xy_sum - x_sum * y_sum) / denom
            intercept = (y_sum - slope * x_sum) / period
            
            # Value at current point
            result[i] = intercept + slope * (period - 1)
    
    return result


def LinearRegSlope(data: List[float], period: int = 14) -> List[float]:
    """
    Linear Regression Slope
    """
    n = len(data)
    result = [0.0] * n
    
    for i in range(period - 1, n):
        y = data[i - period + 1 : i + 1]
        
        x_sum = period * (period - 1) / 2
        x2_sum = period * (period - 1) * (2 * period - 1) / 6
        y_sum = sum(y)
        xy_sum = sum(j * y[j] for j in range(period))
        
        denom = period * x2_sum - x_sum * x_sum
        if denom != 0:
            result[i] = (period * xy_sum - x_sum * y_sum) / denom
    
    return result
