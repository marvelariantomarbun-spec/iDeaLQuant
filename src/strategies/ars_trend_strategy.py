import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.indicators.core import ARS

class ARSTrendStrategy:
    """
    ARS Trend Strategy:
    - Buy: Close(i-1) < ARS(i-1) and Close(i) > ARS(i)
    - Sell: Close(i-1) > ARS(i-1) and Close(i) < ARS(i)
    """
    def __init__(self, ema_period: int = 5, k_value: float = 1.23):
        self.ema_period = ema_period
        self.k = k_value / 100.0  # Q3 calculation in C# snippet
        
    def run(self, bars: pd.DataFrame) -> np.ndarray:
        """
        Executes strategy and returns signals array: 1 (Long), -1 (Short), 0 (Flat)
        """
        closes = bars['Kapanis'].values
        highs = bars['Yuksek'].values
        lows = bars['Dusuk'].values
        
        # Typical Price
        typical = (highs + lows + closes) / 3.0
        
        # Calculate ARS (Fixed/Classic version)
        # Using 0.025 rounding per user request
        ars_vals = ARS(typical.tolist(), ema_period=self.ema_period, k=self.k, round_step=0.025)
        ars_vals = np.array(ars_vals)
        
        signals = np.zeros(len(closes))
        
        # Strategy Logic (Exact Crossover per USER_REQUEST):
        # Kapanış (i-1) < ARS(i-1) & Kapanış(i) > ARS(i) = A (Long)
        # Kapanış (i-1) > ARS(i-1) & Kapanış(i) < ARS(i) = S (Short)
        
        for i in range(1, len(closes)):
            # Crossover Up -> Buy/Long
            if closes[i-1] < ars_vals[i-1] and closes[i] > ars_vals[i]:
                signals[i] = 1
            # Crossover Down -> Sell/Short
            elif closes[i-1] > ars_vals[i-1] and closes[i] < ars_vals[i]:
                signals[i] = -1
            else:
                signals[i] = signals[i-1] # Stay in position
                
        return signals

def backtest_ars_trend(bars: pd.DataFrame, ema_period: int, k_value: float) -> Dict[str, Any]:
    strat = ARSTrendStrategy(ema_period=ema_period, k_value=k_value)
    signals = strat.run(bars)
    
    closes = bars['Kapanis'].values
    returns = np.diff(closes) / closes[:-1]
    
    # Shift signals to avoid lookahead (trade at N+1 Open/Close)
    strat_returns = signals[:-1] * returns
    
    # Calculate Wins and Losses for PF
    wins = strat_returns[strat_returns > 0]
    losses = strat_returns[strat_returns < 0]
    
    sum_wins = np.sum(wins) if len(wins) > 0 else 0
    sum_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-9
    
    pf = sum_wins / sum_losses
    net_profit = np.sum(strat_returns)
    net_points = np.sum(signals[:-1] * np.diff(closes))
    num_trades = np.sum(np.diff(signals) != 0)
    
    # Calculate Max Drawdown (Percentage based)
    equity_curve = np.cumsum(strat_returns)
    # Convert cumulative log returns to price-like scale for DD if needed, 
    # but since these are simple percentage returns now (np.diff(closes)/closes), 
    # cumsum is roughly profit. Let's use peak-minus-current.
    peak = -np.inf
    max_dd = 0
    running_equity = 0
    
    # Points based DD
    point_peak = -np.inf
    max_dd_points = 0
    running_points = 0
    
    point_returns = signals[:-1] * np.diff(closes)
    
    for i in range(len(strat_returns)):
        # Percentage DD
        running_equity += strat_returns[i]
        if running_equity > peak:
            peak = running_equity
        dd = peak - running_equity
        if dd > max_dd:
            max_dd = dd
            
        # Point DD
        running_points += point_returns[i]
        if running_points > point_peak:
            point_peak = running_points
        dd_pts = point_peak - running_points
        if dd_pts > max_dd_points:
            max_dd_points = dd_pts
            
    return {
        'net_profit': float(net_profit),
        'net_points': float(net_points),
        'num_trades': int(num_trades),
        'pf': float(pf),
        'max_dd': float(max_dd),
        'max_dd_points': float(max_dd_points),
        'fitness': float(net_profit) * (1.1 if pf > 2.0 else (pf if pf > 1.0 else 0.5))
    }
