import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.indicators.core import ARS, MACDV, NetLot, MA, ADX

class ARSPulseStrategy:
    """
    ARS Pulse Strategy (Hybrid Score):
    Combines Trend (ARS), Momentum (MACDV), and Volume (NetLot)
    with Horizontal filtering.
    """
    def __init__(self, ema_period: int = 3, k_value: float = 1.23, 
                 macdv_k: int = 13, macdv_u: int = 28, macdv_sig: int = 8,
                 netlot_period: int = 5, adx_th: int = 25, netlot_th: int = 10):
        self.ema_period = int(ema_period)
        self.k = float(k_value) / 100.0
        self.macdv_k = int(macdv_k)
        self.macdv_u = int(macdv_u)
        self.macdv_sig = int(macdv_sig)
        self.netlot_period = int(netlot_period)
        self.adx_th = float(adx_th)
        self.netlot_th = float(netlot_th)
        
    def run(self, bars: pd.DataFrame) -> np.ndarray:
        closes = bars['Kapanis'].values
        highs = bars['Yuksek'].values
        lows = bars['Dusuk'].values
        opens = bars['Acilis'].values
        typical = (highs + lows + closes) / 3.0
        
        # 1. Indicators
        ars_vals = np.array(ARS(typical.tolist(), ema_period=self.ema_period, k=self.k, round_step=0.025))
        macdv, macdv_sig = MACDV(closes.tolist(), highs.tolist(), lows.tolist(), k=self.macdv_k, u=self.macdv_u, sig=self.macdv_sig)
        netlot = np.array(NetLot(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(), period=self.netlot_period))
        adx = np.array(ADX(highs.tolist(), lows.tolist(), closes.tolist(), period=14))
        
        n = len(closes)
        signals = np.zeros(n)
        
        for i in range(1, n):
            # ARS is the core indicator. Trigger on crossover.
            if closes[i] > ars_vals[i] and closes[i-1] <= ars_vals[i-1]:
                # POTENTIAL LONG TRIGGER
                # Mandatory Filters
                if macdv[i] > macdv_sig[i] and netlot[i] > self.netlot_th and adx[i] > self.adx_th:
                    signals[i] = 1
                else:
                    signals[i] = 0 # Reject signal
            elif closes[i] < ars_vals[i] and closes[i-1] >= ars_vals[i-1]:
                # POTENTIAL SHORT TRIGGER
                if macdv[i] < macdv_sig[i] and netlot[i] < -self.netlot_th and adx[i] > self.adx_th:
                    signals[i] = -1
                else:
                    signals[i] = 0 # Reject signal
            else:
                # Keep current position
                signals[i] = signals[i-1]
                
                # Exit logic: ARS Crossover reversal
                if signals[i-1] == 1 and closes[i] < ars_vals[i]:
                    signals[i] = 0
                elif signals[i-1] == -1 and closes[i] > ars_vals[i]:
                    signals[i] = 0
                   
        return signals, None # Scores removed as per user request

def backtest_ars_pulse(bars: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    strat = ARSPulseStrategy(**params)
    signals, scores = strat.run(bars)
    
    closes = bars['Kapanis'].values
    returns = np.diff(closes) / closes[:-1]
    strat_returns = signals[:-1] * returns
    
    net_profit = np.sum(strat_returns)
    net_points = np.sum(signals[:-1] * np.diff(closes))
    num_trades = np.sum(np.diff(signals) != 0)
    
    # PF
    wins = strat_returns[strat_returns > 0]
    losses = strat_returns[strat_returns < 0]
    sum_wins = np.sum(wins) if len(wins) > 0 else 0
    sum_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-9
    pf = sum_wins / sum_losses
    
    # MaxDD
    point_peak = -np.inf
    max_dd_points = 0
    running_points = 0
    point_returns = signals[:-1] * np.diff(closes)
    for i in range(len(point_returns)):
        running_points += point_returns[i]
        if running_points > point_peak: point_peak = running_points
        dd = point_peak - running_points
        if dd > max_dd_points: max_dd_points = dd

    return {
        'net_profit': float(net_profit),
        'net_points': float(net_points),
        'num_trades': int(num_trades),
        'pf': float(pf),
        'max_dd_points': float(max_dd_points),
        'fitness': float(net_profit) * (1.1 if pf > 2.0 else (pf if pf > 1.0 else 0.5))
    }
