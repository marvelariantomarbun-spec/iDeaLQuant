# -*- coding: utf-8 -*-
"""
Strategy 3 Optimizer (Paradise)
Hedef: Paradise stratejisi için en iyi parametreleri bulmak.
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

from src.indicators.core import EMA, SMA, ATR, Momentum, HHV, LLV
from src.engine.data import OHLCV

# Global cache
g_cache = None
g_mask = None

# --- INDICATOR CACHE ---
class IndicatorCache:
    def __init__(self, df):
        self.closes = df['close'].values
        self.highs = df['high'].values
        self.lows = df['low'].values
        self.volumes = df['volume'].values # Using 'volume' column
        
        # Determine actual volume column (lot vs volume)
        if 'lot' in df.columns:
             self.volumes = df['lot'].values
        
        self.ema_cache = {}
        self.dsma_cache = {}
        self.sma_cache = {}
        self.mom_cache = {}
        self.hhv_cache = {}
        self.llv_cache = {}
        self.atr_cache = {}
        self.vol_hhv_cache = {}

    def get_ema(self, period):
        if period not in self.ema_cache:
            self.ema_cache[period] = np.array(EMA(self.closes.tolist(), int(period)))
        return self.ema_cache[period]
        
    def get_dsma(self, period):
        if period not in self.dsma_cache:
            # DSMA = SMA(SMA(C, p), p)
            # Core SMA returns list
            s1 = SMA(self.closes.tolist(), int(period))
            s2 = SMA(s1, int(period))
            self.dsma_cache[period] = np.array(s2)
        return self.dsma_cache[period]

    def get_sma(self, period):
        if period not in self.sma_cache:
            self.sma_cache[period] = np.array(SMA(self.closes.tolist(), int(period)))
        return self.sma_cache[period]

    def get_mom(self, period):
        if period not in self.mom_cache:
            self.mom_cache[period] = np.array(Momentum(self.closes.tolist(), int(period)))
        return self.mom_cache[period]
        
    def get_hhv(self, period):
        if period not in self.hhv_cache:
            self.hhv_cache[period] = np.array(HHV(self.highs.tolist(), int(period)))
        return self.hhv_cache[period]

    def get_llv(self, period):
        if period not in self.llv_cache:
            self.llv_cache[period] = np.array(LLV(self.lows.tolist(), int(period)))
        return self.llv_cache[period]
        
    def get_atr(self, period):
        if period not in self.atr_cache:
            self.atr_cache[period] = np.array(ATR(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(period)))
        return self.atr_cache[period]

    def get_vol_hhv(self, period):
        if period not in self.vol_hhv_cache:
            self.vol_hhv_cache[period] = np.array(HHV(self.volumes.tolist(), int(period)))
        return self.vol_hhv_cache[period]

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

# --- FAST BACKTEST (Paradise Logic) ---
@jit(nopython=True)
def fast_backtest_paradise(closes, highs, lows, volumes,
                           ema_arr, dsma_arr, sma_arr, mom_arr, 
                           hh_arr, ll_arr, atr_arr, vol_hhv_arr,
                           mask_arr,
                           mom_limit_low, mom_limit_high, # 98, 102
                           atr_sl, atr_tp, atr_trail,
                           yon_modu_cift): # True=CIFT, False=SADECE_AL
    
    n = len(closes)
    
    pos = 0 # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    extreme_price = 0.0
    
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    # Logic:
    # AL: HH > Prev HH, EMA > DSMA, Close > SMA(20), Mom > 100, Vol > VolHHV * 0.8
    # SAT: LL < Prev LL, EMA < DSMA, Close < SMA(20), Mom < 100, Vol > VolHHV * 0.8
    
    for i in range(100, n): # Warmup
        
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
            
        atr_val = atr_arr[i]
        
        # --- EXIT LOGIC ---
        if pos == 1:
            if closes[i] > extreme_price: extreme_price = closes[i]
            
            # SL
            if closes[i] <= entry_price - (atr_val * atr_sl):
                pnl = closes[i] - entry_price # Slippage could be added here
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # TP
            elif closes[i] >= entry_price + (atr_val * atr_tp):
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # Trailing
            elif closes[i] < extreme_price - (atr_val * atr_trail):
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                
            if pos == 0:
                max_dd = max(max_dd, peak_equity - current_equity)
                if current_equity > peak_equity: peak_equity = current_equity

        elif pos == -1:
            if closes[i] < extreme_price: extreme_price = closes[i]
            
            # SL
            if closes[i] >= entry_price + (atr_val * atr_sl):
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # TP
            elif closes[i] <= entry_price - (atr_val * atr_tp):
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # Trailing
            elif closes[i] > extreme_price + (atr_val * atr_trail):
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                
            if pos == 0:
                max_dd = max(max_dd, peak_equity - current_equity)
                if current_equity > peak_equity: peak_equity = current_equity

        # --- ENTRY LOGIC ---
        if pos == 0:
            mom = mom_arr[i]
            mom_band = (mom > mom_limit_low) and (mom < mom_limit_high)
            
            if mom_band:
                # Vol Check
                vol_ok = volumes[i] >= vol_hhv_arr[i-1] * 0.8
                if vol_ok:
                    # LONG
                    if (hh_arr[i] > hh_arr[i-1] and 
                        ema_arr[i] > dsma_arr[i] and 
                        closes[i] > sma_arr[i] and
                        mom > 100):
                        
                        pos = 1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        trades += 1
                        
                    # SHORT (if allowed)
                    elif yon_modu_cift and (ll_arr[i] < ll_arr[i-1] and 
                                            ema_arr[i] < dsma_arr[i] and 
                                            closes[i] < sma_arr[i] and
                                            mom < 100):
                        
                        pos = -1
                        entry_price = closes[i]
                        extreme_price = closes[i]
                        trades += 1

    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    
    return net_profit, trades, pf, max_dd

# --- WORKER TASK ---
def solve_chunk(args):
    # params: atr_p, atr_sl, atr_tp, params_grid
    atr_p, atr_sl, atr_tp, params_grid = args
    
    global g_cache, g_mask
    if g_cache is None: return []
    
    results = []
    
    # Fixed params for now (optimization usually on Risk & Trend)
    ema_p = 21
    dsma_p = 50
    ma_p = 20
    hh_p = 25
    vol_p = 14
    mom_p = 60
    
    # Cached Arrays
    ema_arr = g_cache.get_ema(ema_p)
    dsma_arr = g_cache.get_dsma(dsma_p)
    sma_arr = g_cache.get_sma(ma_p)
    mom_arr = g_cache.get_mom(mom_p)
    hh_arr = g_cache.get_hhv(hh_p)
    ll_arr = g_cache.get_llv(hh_p)
    vol_hhv_arr = g_cache.get_vol_hhv(vol_p)
    atr_arr = g_cache.get_atr(atr_p)
    
    # Iterate other params
    atr_trails = params_grid.get('atr_trails', [2.5])
    mom_lows = params_grid.get('mom_lows', [98.0])
    mom_highs = params_grid.get('mom_highs', [102.0])
    
    for trl in atr_trails:
        for ml in mom_lows:
            for mh in mom_highs:
                np_val, tr, pf, dd = fast_backtest_paradise(
                    g_cache.closes, g_cache.highs, g_cache.lows, g_cache.volumes,
                    ema_arr, dsma_arr, sma_arr, mom_arr,
                    hh_arr, ll_arr, atr_arr, vol_hhv_arr,
                    g_mask,
                    ml, mh,
                    atr_sl, atr_tp, trl,
                    True # yon_modu_cift
                )
                
                if np_val > 0:
                    results.append({
                        'NP': np_val, 'PF': pf, 'DD': dd, 'Tr': tr,
                        'ATR_P': atr_p, 'SL': atr_sl, 'TP': atr_tp, 'TRL': trl,
                         'ML': ml, 'MH': mh
                    })
                        
    return results

def run_strategy3_optimization():
    print("--- S3 (Paradise) Optimization Starting (Numba + Mask) ---")
    
    grid = {
        'atr_periods': [10, 14, 20],
        'atr_sls': [1.5, 2.0, 2.5, 3.0],
        'atr_tps': [3.0, 4.0, 5.0, 6.0],
        'atr_trails': [2.0, 2.5, 3.0],
        'mom_lows': [98.0],
        'mom_highs': [102.0]
    }
    
    tasks = []
    for ap in grid['atr_periods']:
        for sl in grid['atr_sls']:
            for tp in grid['atr_tps']:
                tasks.append((ap, sl, tp, grid))
            
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
        df['Score'] = df['NP'] * df['PF'] # Simple Score
        best = df.nlargest(1, 'Score').iloc[0]
        print(f"\nBEST S3 Result:\n{best.to_string()}")
        df.sort_values('Score', ascending=False).head(50).to_csv(r"d:\Projects\IdealQuant\results\strategy3_results.csv")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_strategy3_optimization()
