# -*- coding: utf-8 -*-
"""
Strategy 4 Optimizer (TOMA + Momentum + TRIX)
Hedef: TOMA stratejisi için en iyi parametreleri bulmak.
Yöntem: Numba JIT + Vade/Tatil Maskesi (Hız + Doğruluk)
"""

import sys
import os
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool, cpu_count, current_process
from numba import jit

# Proje kök dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, Momentum, TRIX, HHV, LLV
from src.engine.data import OHLCV

# Global cache
g_cache = None
g_mask = None # Trading Mask (Vade/Tatil)

# --- INDICATOR CACHE ---
class IndicatorCache:
    def __init__(self, df):
        # Support both English and Turkish column names
        if 'close' in df.columns:
            self.closes = df['close'].values
            self.highs = df['high'].values
            self.lows = df['low'].values
            self.volume = df['volume'].values
        elif 'Kapanis' in df.columns:
            self.closes = df['Kapanis'].values
            self.highs = df['Yuksek'].values
            self.lows = df['Dusuk'].values
            self.volume = df['Lot'].values if 'Lot' in df.columns else df['Hacim'].values
        else:
            raise ValueError("DataFrame must contain 'close' or 'Kapanis' columns")
        
        self.toma_cache = {}    # (period, opt_pct) -> (toma, trend)
        self.mom_cache = {}     # period -> values
        self.trix_cache = {}    # period -> values
        self.hhv_cache = {}     # period -> values
        self.llv_cache = {}     # period -> values

    def get_toma(self, period, opt_pct):
        key = (period, int(opt_pct*100)) # float key riskini azaltmak icin int key
        if key not in self.toma_cache:
            # TOMA implementation in Python (pre-calc)
            # TOMA = EMA(C, period) * (1 +/- opt_pct) logic w/ Trailing
            # But the core indicator function is needed.
            # Using simple pre-calculation here or calling core if available.
            # Since TOMA logic is recursive (trailing), we might need a fast implementation here
            # or rely on src.indicators.trend if available.
            # Let's implement a fast numpy/numba version here for cache.
            self.toma_cache[key] = calc_toma_numpy(self.closes, int(period), float(opt_pct))
        return self.toma_cache[key]

    def get_mom(self, period):
        if period not in self.mom_cache:
            self.mom_cache[period] = np.array(Momentum(self.closes.tolist(), int(period)))
        return self.mom_cache[period]

    def get_trix(self, period):
        if period not in self.trix_cache:
            self.trix_cache[period] = np.array(TRIX(self.closes.tolist(), int(period)))
        return self.trix_cache[period]
        
    def get_hhv(self, period):
        if period not in self.hhv_cache:
            self.hhv_cache[period] = np.array(HHV(self.highs.tolist(), int(period)))
        return self.hhv_cache[period]

    def get_llv(self, period):
        if period not in self.llv_cache:
            self.llv_cache[period] = np.array(LLV(self.lows.tolist(), int(period)))
        return self.llv_cache[period]

def calc_toma_numpy(closes, period, opt_pct):
    """
    Calculate TOMA arrays (TomaValue, Trend)
    """
    # 1. Calculate EMA
    # Standard EMA
    alpha = 2 / (period + 1)
    ema = np.zeros_like(closes)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = (closes[i] - ema[i-1]) * alpha + ema[i-1]
        
    # 2. Apply TOMA Logic
    # Trend: 1 (Up), -1 (Down)
    # Toma: Trailing Stop level
    toma = np.zeros_like(closes)
    trend = np.zeros_like(closes, dtype=np.int32)
    
    # Init
    toma[0] = closes[0]
    trend[0] = 1
    
    opt = opt_pct / 100.0
    
    for i in range(1, len(closes)):
        prev_trend = trend[i-1]
        e = ema[i]
        
        if prev_trend == 1:
            # Uptrend
            new_trailing = e * (1 - opt)
            # Trailing stop can only go UP
            if new_trailing > toma[i-1]:
                toma[i] = new_trailing
            else:
                toma[i] = toma[i-1]
                
            # Check Reversal
            if closes[i] < toma[i]:
                trend[i] = -1
                toma[i] = e * (1 + opt) # Reset to upper band
            else:
                trend[i] = 1
                
        else: # prev_trend == -1
            # Downtrend
            new_trailing = e * (1 + opt)
            # Trailing stop can only go DOWN
            if new_trailing < toma[i-1]:
                toma[i] = new_trailing
            else:
                toma[i] = toma[i-1]
                
            # Check Reversal
            if closes[i] > toma[i]:
                trend[i] = 1
                toma[i] = e * (1 - opt) # Reset to lower band
            else:
                trend[i] = -1
                
    return toma, trend

# --- DATA LOADING ---
def load_data_and_mask(vade_tipi="ENDEKS"):
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        data = OHLCV.from_ideal_export(csv_path)
        if current_process().name == 'MainProcess':
            print(f"Veri Yuklendi: {len(data)} Bar")
            
        mask = data.get_trading_mask(vade_tipi)
        return data.df, mask
    except Exception as e:
        print(f"Hata: {e}")
        return None, None

def worker_init():
    global g_cache, g_mask
    df, mask = load_data_and_mask()
    if df is not None:
        g_cache = IndicatorCache(df)
        g_mask = mask # it is already a numpy array

# --- PARALLEL WORKER FUNCTIONS (Phase 2 & 3) ---
# Architecture: All arrays stored in global dict via initializer.
# Task tuples contain ONLY scalar params (keys/thresholds).
# This avoids serializing numpy arrays per-task (100,000x less memory).

_g_s4 = {}  # Global shared state for worker processes

def s4_parallel_init(shared_data):
    """Initializer for Phase 2/3 workers. shared_data is a dict of ALL needed arrays."""
    global _g_s4
    _g_s4 = shared_data
    # Suppress BrokenPipeError on Windows when parent process exits
    import signal
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (OSError, ValueError):
        pass

def s4_p2_eval(params):
    """
    Phase 2: Single parameter combination evaluation.
    params: (mom_p, trix_p, h2p, l2p, mh, lb1)  — ALL SCALARS
    Returns: (score, params_dict) or None
    """
    mom_p, trix_p, h2p, l2p, mh, lb1 = params
    g = _g_s4
    
    n = len(g['closes'])
    dummy = np.zeros(n)
    
    res = fast_backtest_strategy4(
        g['closes'], g['toma_trend'], g['toma_val'],
        g['hhv1'], g['llv1'],
        g['hhv'][h2p], g['llv'][l2p],
        dummy, dummy,
        g['mom'][mom_p], g['trix'][trix_p], g['mask'],
        -9999.0, mh,
        lb1, 100,
        0.0, 0.0
    )
    
    score = res[0] * res[2] if res[2] > 0 else 0.0
    if score > 0:
        return (score, {
            'mom_period': mom_p, 'trix_period': trix_p,
            'mom_limit_high': mh, 'trix_lb1': lb1,
            'hhv2': h2p, 'llv2': l2p,
            'net_profit': res[0], 'trades': res[1], 'pf': res[2],
            'max_dd': res[3], 'sharpe': res[4]
        })
    return None

def s4_p3_eval(params):
    """
    Phase 3: Single parameter combination evaluation.
    params: (h3p, l3p, ml, lb2, ka, iz, meta_dict)  — SCALARS + tiny metadata dict
    Returns: result dict (with fitness) or None
    """
    from src.optimization.fitness import quick_fitness
    
    h3p, l3p, ml, lb2, ka, iz, meta = params
    g = _g_s4
    
    res = fast_backtest_strategy4(
        g['closes'], g['toma_trend'], g['toma_val'],
        g['hhv1'], g['llv1'],
        g['hhv2_fixed'], g['llv2_fixed'],
        g['hhv'][h3p], g['llv'][l3p],
        g['mom_fixed'], g['trix_fixed'], g['mask'],
        ml, meta['fix_mh'],
        meta['fix_lb1'], lb2,
        ka / 100.0, iz / 100.0
    )
    
    np_val, tr, pf, dd, sh = res
    # Erken filtreleme: anlamsız sonuçları worker'da ele
    if np_val > 0 and pf >= 1.0 and tr >= 5:
        fitness = quick_fitness(np_val, pf, dd, tr, sharpe=sh)
        if fitness > 0:
            return {
                'toma_period': meta['fix_tp'], 'toma_opt': meta['fix_to'],
                'hhv1_period': meta['p1_hhv1'], 'llv1_period': meta['p1_llv1'],
                'mom_period': meta['fix_mom_p'], 'trix_period': meta['fix_trix_p'],
                'mom_limit_high': meta['fix_mh'], 'trix_lb1': meta['fix_lb1'],
                'hhv2_period': meta['p2_hhv2'], 'llv2_period': meta['p2_llv2'],
                'mom_limit_low': ml, 'trix_lb2': lb2,
                'hhv3_period': h3p, 'llv3_period': l3p,
                'kar_al': ka, 'iz_stop': iz,
                'net_profit': np_val, 'trades': tr, 'pf': pf, 'max_dd': dd,
                'sharpe': sh, 'fitness': fitness
            }
    return None

# --- FAST BACKTEST (Strategy 4 Logic - Multi-Layer) ---
@jit(nopython=True)
def fast_backtest_strategy4(closes, 
                            toma_trend, toma_val, # TOMA
                            hhv1, llv1, # Layer 3 Breakout (20)
                            hhv2, llv2, # Layer 1 Breakout (150/190)
                            hhv3, llv3, # Layer 2 Breakout (150/190)
                            mom_arr, trix_arr, # Indicators
                            mask_arr, # Trading Mask
                            # Params
                            mom_limit_low, mom_limit_high, # 98, 101.5
                            trix_lookback1, trix_lookback2, # 110, 140
                            kar_al_ratio, iz_stop_ratio):
    
    n = len(closes)
    pos = 0 
    entry_price = 0.0
    extreme_price = 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    # Sharpe accumulators (online mean/var)
    sum_pnl = 0.0
    sum_pnl_sq = 0.0
    n_closed = 0
    
    # Constants for signal mapping
    # 0: None, 1: Long (A), -1: Short (S)
    
    # Warmup: en büyük lookback parametresine göre dinamik hesapla
    # trix_lookback1/2 en büyük olabilir (200+ bar)
    min_warmup = max(200, trix_lookback1 + 1, trix_lookback2 + 1)
    
    son_yon = 0 # Last Direction
    
    for i in range(min_warmup, n): # Dinamik warmup
        
        # --- TRADING MASK CHECK ---
        if not mask_arr[i]:
            if pos != 0:
                # Force Close
                pnl = closes[i] - entry_price if pos == 1 else entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                sum_pnl += pnl
                sum_pnl_sq += pnl * pnl
                n_closed += 1
                pos = 0
                son_yon = 0
                
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd
            continue
            
        signal = 0 # ""
        
        # --- LAYER 1: MOM > 101.5 ---
        if mom_arr[i] > mom_limit_high:
            # Long: HH2 Breakout & TRIX1 Divergence (Lower than 110 bars ago, but ticking up)
            # TRIX1[i] < TRIX1[i-110] && TRIX1[i] > TRIX1[i-1]
            if hhv2[i] > hhv2[i-1] and trix_arr[i] < trix_arr[i - trix_lookback1] and trix_arr[i] > trix_arr[i-1]:
                signal = 1
            
            # Short: LL2 Breakout & TRIX1 Divergence (Higher than 110 bars ago, but ticking down)
            # TRIX1[i] > TRIX1[i-110] && TRIX1[i] < TRIX1[i-1]
            if llv2[i] < llv2[i-1] and trix_arr[i] > trix_arr[i - trix_lookback1] and trix_arr[i] < trix_arr[i-1]:
                signal = -1
                
        # --- LAYER 2: MOM < 98 ---
        if mom_arr[i] < mom_limit_low:
            # Long: HH3 Breakout & TRIX2 Divergence (Lower than 140 bars ago, but ticking up)
            if hhv3[i] > hhv3[i-1] and trix_arr[i] < trix_arr[i - trix_lookback2] and trix_arr[i] > trix_arr[i-1]:
                signal = 1
                
            # Short: LL3 Breakout & TRIX2 Divergence (Higher than 140 bars ago, but ticking down)
            if llv3[i] < llv3[i-1] and trix_arr[i] > trix_arr[i - trix_lookback2] and trix_arr[i] < trix_arr[i-1]:
                signal = -1

        # --- LAYER 3: TOMA (Priority) ---
        # Overwrites previous signals
        # Long: HH1 Breakout & C > TOMA
        if hhv1[i] > hhv1[i-1] and closes[i] > toma_val[i]:
            signal = 1
            
        # Short: LL1 Breakout & C < TOMA
        if llv1[i] < llv1[i-1] and closes[i] < toma_val[i]:
            signal = -1


        # --- EXECUTION LOGIC ---
        # "if (Sinyal != "" && SonYon != Sinyal)"
        if signal != 0 and son_yon != signal:
            # Close previous if exists
            if pos != 0:
                pnl = closes[i] - entry_price if pos == 1 else entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                sum_pnl += pnl
                sum_pnl_sq += pnl * pnl
                n_closed += 1
            
            # Open new
            pos = signal
            son_yon = signal
            entry_price = closes[i]
            extreme_price = closes[i]
            trades += 1


        # --- EXIT LOGIC (Kar Al / Stop) ---
        # Only if we have a position
        if pos == 1:
            if closes[i] > extreme_price: extreme_price = closes[i]
            
            exit_signal = False
            # Kar Al
            if kar_al_ratio > 0 and closes[i] >= entry_price * (1 + kar_al_ratio): exit_signal = True
            # Izleyen Stop
            if iz_stop_ratio > 0 and closes[i] < extreme_price * (1 - iz_stop_ratio): exit_signal = True
            
            if exit_signal:
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                sum_pnl += pnl
                sum_pnl_sq += pnl * pnl
                n_closed += 1
                pos = 0 # Flat
                # SonYon remains 1? In user code exit logic isn't explicit but normally exits don't change SonYon direction flag in Ideal, just Sinyal="F"
                # But user code loop doesn't have "F" logic shown. Assuming "Always In" unless specialized exit.
                # Adding Flat for consistency with optimization goals.
                
        elif pos == -1:
            if closes[i] < extreme_price: extreme_price = closes[i]
            
            exit_signal = False
            # Kar Al
            if kar_al_ratio > 0 and closes[i] <= entry_price * (1 - kar_al_ratio): exit_signal = True
            # Izleyen Stop
            if iz_stop_ratio > 0 and closes[i] > extreme_price * (1 + iz_stop_ratio): exit_signal = True
            
            if exit_signal:
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                sum_pnl += pnl
                sum_pnl_sq += pnl * pnl
                n_closed += 1
                pos = 0
                
        # Update DD
        if current_equity > peak_equity: peak_equity = current_equity
        dd = peak_equity - current_equity
        if dd > max_dd: max_dd = dd

    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    
    # Sharpe Ratio (Trade-Based)
    sharpe = 0.0
    if n_closed > 1:
        mean_pnl = sum_pnl / n_closed
        var_pnl = (sum_pnl_sq / n_closed) - (mean_pnl * mean_pnl)
        if var_pnl > 0:
            std_pnl = var_pnl ** 0.5
            sharpe = mean_pnl / std_pnl
    
    return net_profit, trades, pf, max_dd, sharpe

# --- WORKER TASK ---
def solve_chunk(args):
    # args: (toma_period, toma_opt, params)
    toma_p, toma_opt, params_grid = args
    
    global g_cache, g_mask
    if g_cache is None: return []
    
    results = []
    
    # Fixed periods (can be optimized later)
    mom_period = 1900
    trix_period = 120
    
    hhv1_p = 20
    llv1_p = 20
    
    hhv2_p = 150
    llv2_p = 190 #(LL2=190 in code)
    
    hhv3_p = 150
    llv3_p = 190
    
    # Get Cached Arrays
    try:
        toma_val, toma_trend = g_cache.get_toma(toma_p, toma_opt)
        mom_arr = g_cache.get_mom(mom_period)
        trix_arr = g_cache.get_trix(trix_period)
        
        hhv1 = g_cache.get_hhv(hhv1_p)
        llv1 = g_cache.get_llv(llv1_p)
        
        hhv2 = g_cache.get_hhv(hhv2_p)
        llv2 = g_cache.get_llv(llv2_p)
        
        hhv3 = g_cache.get_hhv(hhv3_p)
        llv3 = g_cache.get_llv(llv3_p)
    except Exception as e:
        # Fallback if cache fails
        return []
    
    # Param Grid Iteration
    kar_als = params_grid.get('kar_als', [0.0]) 
    iz_stops = params_grid.get('iz_stops', [0.0])
    mom_lows = params_grid.get('mom_lows', [98.0])
    mom_highs = params_grid.get('mom_highs', [101.5])
    trix_lb1s = params_grid.get('trix_lb1s', [110]) # Default 110
    trix_lb2s = params_grid.get('trix_lb2s', [140]) # Default 140
    
    for ka in kar_als:
        for iz in iz_stops:
            for ml in mom_lows:
                for mh in mom_highs:
                    for lb1 in trix_lb1s:
                        for lb2 in trix_lb2s:
                            np_val, tr, pf, dd, sh = fast_backtest_strategy4(
                                g_cache.closes, 
                                toma_trend, toma_val,
                                hhv1, llv1,
                                hhv2, llv2,
                                hhv3, llv3,
                                mom_arr, trix_arr,
                                g_mask,
                                ml, mh,
                                lb1, lb2,
                                ka/100.0, iz/100.0
                            )
                            
                            if np_val > 0:
                                results.append({
                                    'NP': np_val, 'PF': pf, 'DD': dd, 'Tr': tr,
                                    'Sharpe': sh,
                                    'T_Per': toma_p, 'T_Opt': toma_opt,
                                    'KA': ka, 'IS': iz, 'ML': ml, 'MH': mh,
                                    'LB1': lb1, 'LB2': lb2
                                })
                        
    return results

def run_strategy4_optimization():
    print("--- S4 (TOMA) Optimization Starting (Phase 3: Layer 2 - Momentum Low) ---")
    
    # Phase 3 GRID:
    # TOMA: Fixed Best (Period=97, Opt=1.5)
    # Layer 1: Fixed Best (MomHigh=101.5, TrixLB1=145)
    # Layer 2: Optimize MomLow (94-100) & TrixLB2 (100-180)
    
    grid = {
        'toma_periods': [97],
        'toma_opts': [1.5],
        'kar_als': [0.0], 
        'iz_stops': [0.0],
        'mom_lows': [round(x, 1) for x in np.arange(94.0, 100.5, 0.5)], # Step 0.5
        'mom_highs': [101.5], # Fixed Best from Phase 2
        'trix_lb1s': [145],  # Fixed Best from Phase 2
        'trix_lb2s': list(range(100, 181, 5)) # Step 5
    }
    
    tasks = []
    for tp in grid['toma_periods']:
        for to in grid['toma_opts']:
            tasks.append((tp, to, grid))
            
    # Total Tasks = 1 TOMA * 1 MomHigh * 13 MomLows * 17 Trix2s = 221 tasks per chunk
    
    print(f"Total Tasks (Chunks): {len(tasks)}")
    
    start_time = time()
    final_results = []
    
    with Pool(processes=min(16, cpu_count()), initializer=worker_init) as pool:
        for res in pool.imap_unordered(solve_chunk, tasks):
            final_results.extend(res)
            
    elapsed = time() - start_time
    print(f"Done in {elapsed:.1f}s. Results: {len(final_results)}")
    
    if final_results:
        df = pd.DataFrame(final_results)
        df['Score'] = df['NP'] * df['PF'] # Simple Score
        best = df.nlargest(1, 'Score').iloc[0]
        print(f"\nBEST S4 Result (Phase 3):\n{best.to_string()}")
        os.makedirs(r"d:\Projects\IdealQuant\results", exist_ok=True)
        df.sort_values('Score', ascending=False).head(50).to_csv(r"d:\Projects\IdealQuant\results\strategy4_results_phase3.csv")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_strategy4_optimization()
