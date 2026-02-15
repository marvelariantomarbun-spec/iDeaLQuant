# -*- coding: utf-8 -*-
"""
Strategy 1 Optimizer (Score Based)
Hedef: Score Based stratejisi için en iyi parametreleri bulmak.
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

from src.indicators.core import EMA, SMA, ATR, ADX, ARS, NetLot, MACDV, BollingerBands
from src.engine.data import OHLCV

# Global cache
g_cache = None
g_mask = None

# --- INDICATOR CACHE ---
class IndicatorCache:
    def __init__(self, df):
        self.opens = df['open'].values
        self.highs = df['high'].values
        self.lows = df['low'].values
        self.closes = df['close'].values
        self.typical = (self.highs + self.lows + self.closes) / 3.0
        
        # Volume/Lot check
        if 'lot' in df.columns:
             self.lots = df['lot'].values
        elif 'volume' in df.columns:
             self.lots = df['volume'].values
        else:
             self.lots = np.zeros_like(self.closes)

        self.ars_cache = {}
        self.adx_cache = {}
        self.macdv_cache = {} # tuple (macd, sig)
        self.netlot_ma_cache = {}
        self.bb_cache = {} # tuple (upper, mid, lower)

    def get_ars(self, period, k):
        key = (period, k)
        if key not in self.ars_cache:
            self.ars_cache[key] = np.array(ARS(self.typical.tolist(), int(period), float(k)))
        return self.ars_cache[key]

    def get_adx(self, period):
        if period not in self.adx_cache:
            # ADX standard
            self.adx_cache[period] = np.array(ADX(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(period)))
        return self.adx_cache[period]

    def get_macdv(self, short, long, sig):
        key = (short, long, sig)
        if key not in self.macdv_cache:
            m, s = MACDV(self.closes.tolist(), self.highs.tolist(), self.lows.tolist(), int(short), int(long), int(sig))
            self.macdv_cache[key] = (np.array(m), np.array(s))
        return self.macdv_cache[key]

    def get_netlot_ma(self, period):
        if period not in self.netlot_ma_cache:
            nl = NetLot(self.opens.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist())
            nl_ma = SMA(nl, int(period))
            self.netlot_ma_cache[period] = np.array(nl_ma)
        return self.netlot_ma_cache[period]
        
    def get_bb(self, period, std):
        key = (period, std)
        if key not in self.bb_cache:
             u, m, l = BollingerBands(self.closes.tolist(), int(period), float(std))
             self.bb_cache[key] = (np.array(u), np.array(m), np.array(l))
        return self.bb_cache[key]
        
# --- DATA LOADING ---
def load_data_and_mask(vade_tipi="ENDEKS"):
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        data = OHLCV.from_csv(csv_path, separator=';')
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
        g_mask = mask.values

# --- FAST BACKTEST (Score Based Logic) ---
@jit(nopython=True)
def fast_backtest_score(
    closes, ars_arr, adx_arr, macd_arr, sig_arr, netlot_ma_arr,
    bb_u, bb_m, bb_l, 
    mask_arr,
    # Params
    min_score, exit_score, contra_max,
    adx_thr, macd_thr, netlot_thr,
    yatay_adx_thr, filter_score_thr, bb_width_mult, ars_mesafe_thr,
    yatay_ars_bars
):
    n = len(closes)
    pos = 0 # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    # Pre-calc BB Width & Avg manually or pass it?
    # Doing it inside logic for simplicity (Numba handles loops well)
    # BB Width = (U - L) / M * 100
    bb_widths = np.zeros(n)
    for i in range(n):
        if bb_m[i] != 0:
            bb_widths[i] = ((bb_u[i] - bb_l[i]) / bb_m[i]) * 100
            
    # SMA of BB Width (50 period usually fixed)
    # Simple accumulation for SMA
    bb_width_sum = 0.0
    bb_width_avg = np.zeros(n)
    avg_p = 50
    for i in range(n):
        bb_width_sum += bb_widths[i]
        if i >= avg_p:
            bb_width_sum -= bb_widths[i-avg_p]
            bb_width_avg[i] = bb_width_sum / avg_p
        elif i > 0:
            bb_width_avg[i] = bb_width_sum / (i + 1)
            
    # Main Loop
    for i in range(100, n):
        # --- TRADING MASK CHECK ---
        if not mask_arr[i]:
            if pos != 0:
                pnl = 0.0
                if pos == 1: pnl = closes[i] - entry_price
                else: pnl = entry_price - closes[i]
                
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                max_dd = max(max_dd, peak_equity - current_equity)
            continue
            
        # --- SCORE CALCULATION ---
        # 1. ARS Score
        ars_long = closes[i] > ars_arr[i]
        ars_short = closes[i] < ars_arr[i]
        
        # 2. NetLot
        nl_long = netlot_ma_arr[i] > netlot_thr
        nl_short = netlot_ma_arr[i] < -netlot_thr
        
        # 3. ADX 
        adx_ok = adx_arr[i] > adx_thr
        
        # 4. MACD-V
        macd_long = macd_arr[i] > (sig_arr[i] + macd_thr)
        macd_short = macd_arr[i] < (sig_arr[i] - macd_thr)
        
        # Sum Scores
        l_score = 0
        if ars_long: l_score += 1
        if nl_long: l_score += 1
        if adx_ok: l_score += 1
        if macd_long: l_score += 1
        
        s_score = 0
        if ars_short: s_score += 1
        if nl_short: s_score += 1
        if adx_ok: s_score += 1
        if macd_short: s_score += 1
        
        # --- YATAY FILTER ---
        # 1. ARS Stability
        ars_unstable = False
        start_idx = i - yatay_ars_bars
        if start_idx < 0: start_idx = 0
        ref_ars = ars_arr[i]
        for k in range(start_idx, i):
            if ars_arr[k] != ref_ars:
                ars_unstable = True
                break
        
        # 2. ARS Distance
        ars_dist = 0.0
        if ars_arr[i] != 0:
            ars_dist = abs(closes[i] - ars_arr[i]) / ars_arr[i] * 100
            
        # Filter Score
        f_score = 0
        if ars_unstable: f_score += 1
        if ars_dist > ars_mesafe_thr: f_score += 1
        if adx_arr[i] > yatay_adx_thr: f_score += 1
        if bb_widths[i] > (bb_width_avg[i] * bb_width_mult): f_score += 1
        
        filter_pass = (f_score >= filter_score_thr)
        
        # --- EXIT LOGIC ---
        if pos == 1:
            # ARS Reversal OR Score Decay
            if ars_short or s_score >= exit_score:
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                max_dd = max(max_dd, peak_equity - current_equity)
                if current_equity > peak_equity: peak_equity = current_equity
                
        elif pos == -1:
            if ars_long or l_score >= exit_score:
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                max_dd = max(max_dd, peak_equity - current_equity)
                if current_equity > peak_equity: peak_equity = current_equity

        # --- ENTRY LOGIC ---
        if pos == 0 and filter_pass:
            # LONG
            if l_score >= min_score and s_score < contra_max:
                pos = 1
                entry_price = closes[i]
                trades += 1
            # SHORT
            elif s_score >= min_score and l_score < contra_max:
                pos = -1
                entry_price = closes[i]
                trades += 1
                
    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    return net_profit, trades, pf, max_dd

# --- WORKER TASK ---
def solve_chunk(args):
    # args: (ars_p, ars_k, params_grid)
    ars_p, ars_k, params_grid = args
    global g_cache, g_mask
    if g_cache is None: return []

    results = []
    
    # Retrieve Base Arrays (Fixed params for standard optimization)
    # ARS depends on loop args
    ars_arr = g_cache.get_ars(ars_p, ars_k)
    
    # Fixed other periods (Optimization usually targets Thresholds)
    adx_arr = g_cache.get_adx(17)
    macdv, sig = g_cache.get_macdv(13, 28, 8)
    netlot_ma = g_cache.get_netlot_ma(5)
    bb_u, bb_m, bb_l = g_cache.get_bb(20, 2.0)
    
    # Iterate Thresholds
    min_scores = params_grid.get('min_scores', [3])
    exit_scores = params_grid.get('exit_scores', [3])
    adx_thrs = params_grid.get('adx_thrs', [25.0])
    
    for ms in min_scores:
        for es in exit_scores:
            for at in adx_thrs:
                np_val, tr, pf, dd = fast_backtest_score(
                    g_cache.closes, ars_arr, adx_arr, macdv, sig, netlot_ma,
                    bb_u, bb_m, bb_l, g_mask,
                    ms, es, 2, # contra_max fixed
                    at, 0.0, 20.0, # macd_thr, netlot_thr fixed
                    20.0, 2, 0.8, 0.25, 10 # Filter params fixed
                )
                
                if np_val > 0:
                    results.append({
                        'NP': np_val, 'PF': pf, 'DD': dd, 'Tr': tr,
                        'ARS_P': ars_p, 'ARS_K': ars_k,
                        'MinS': ms, 'ExitS': es, 'ADX_T': at
                    })
    return results

def run_strategy1_optimization():
    print("--- S1 (Score) Optimization Starting (Numba + Mask) ---")
    
    grid = {
        'min_scores': [3, 4],
        'exit_scores': [2, 3],
        'adx_thrs': [20.0, 25.0, 30.0],
    }
    
    # Outer Loop: ARS settings
    ars_periods = [3, 5]
    ars_ks = [0.005, 0.01, 0.02]
    
    tasks = []
    for p in ars_periods:
        for k in ars_ks:
            tasks.append((p, k, grid))
            
    print(f"Total Tasks: {len(tasks)}")
    
    start_time = time()
    final_results = []
    
    with Pool(processes=min(16, cpu_count()), initializer=worker_init) as pool:
        for res in pool.imap_unordered(solve_chunk, tasks):
            final_results.extend(res)
            
    elapsed = time() - start_time
    print(f"Done in {elapsed:.1f}s. Results: {len(final_results)}")
    
    if final_results:
        df = pd.DataFrame(final_results)
        df['Score'] = df['NP'] * df['PF']
        best = df.nlargest(1, 'Score').iloc[0]
        print(f"\nBEST S1 Result:\n{best.to_string()}")
        df.sort_values('Score', ascending=False).head(50).to_csv(r"d:\Projects\IdealQuant\results\strategy1_results.csv")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_strategy1_optimization()
