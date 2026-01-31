# -*- coding: utf-8 -*-
"""
Strategy 2 Optimizer (ARS Trend v2)
Hedef: ARS Trend v2 stratejisi için en iyi parametreleri bulmak.
3 Aşamalı Optimizasyon: Satellite -> Drone -> Stability
"""

import sys
import os
import io
import pandas as pd
import numpy as np
from time import time
import itertools
from multiprocessing import Pool, cpu_count, current_process
from numba import jit

# Proje kök dizini (IdealQuant)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, ATR, RSI, Momentum, HHV, LLV, ARS_Dynamic, MoneyFlowIndex

# Global cache for workers
g_cache = None

# --- DATA LOADING ---
def load_data():
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        if current_process().name == 'MainProcess':
            print("Veri Yükleniyor...")
            
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
        df.dropna(inplace=True)
        
        if current_process().name == 'MainProcess':
            print(f"Veri Hazır: {len(df)} Bar")
            
        return df
    except Exception as e:
        print(f"Hata: {e}")
        return None

# --- INDICATOR CACHE ---
class IndicatorCache:
    def __init__(self, df):
        self.df = df
        self.opens = df['Acilis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.closes = df['Kapanis'].values
        self.typical = df['Tipik'].values
        self.n = len(self.closes)
        self.lots = df['Lot'].values  # Volume data (Lot)
        
        self.ars_cache = {}
        self.rsi_cache = {}
        self.mom_cache = {}
        self.hhv_cache = {}
        self.llv_cache = {}
        self.mfi_cache = {}  # MFI cache
        self.vol_hhv_cache = {}  # Volume HHV cache
        self.vol_llv_cache = {}  # Volume LLV cache

    def get_ars(self, ema_p, atr_p, atr_m):
        key = (ema_p, atr_p, round(atr_m, 2))
        if key not in self.ars_cache:
            # ARS Dynamic
            self.ars_cache[key] = np.array(ARS_Dynamic(
                self.typical.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist(),
                ema_period=int(ema_p), atr_period=int(atr_p), atr_mult=float(atr_m),
                min_k=0.002, max_k=0.015  # Fix min/max k to reduce search space
            ))
        return self.ars_cache[key]

    def get_rsi(self, p):
        if p not in self.rsi_cache:
            self.rsi_cache[p] = np.array(RSI(self.closes.tolist(), int(p)))
        return self.rsi_cache[p]
        
    def get_mom(self, p):
        if p not in self.mom_cache:
            self.mom_cache[p] = np.array(Momentum(self.closes.tolist(), int(p)))
        return self.mom_cache[p]
        
    def get_hhv(self, p):
        if p not in self.hhv_cache:
            self.hhv_cache[p] = np.array(HHV(self.highs.tolist(), int(p)))
        return self.hhv_cache[p]

    def get_llv(self, p):
        if p not in self.llv_cache:
            self.llv_cache[p] = np.array(LLV(self.lows.tolist(), int(p)))
        return self.llv_cache[p]
    
    def get_mfi(self, p):
        """Money Flow Index"""
        if p not in self.mfi_cache:
            self.mfi_cache[p] = np.array(MoneyFlowIndex(
                self.highs.tolist(), self.lows.tolist(), 
                self.closes.tolist(), self.lots.tolist(), int(p)
            ))
        return self.mfi_cache[p]
    
    def get_mfi_hhv(self, mfi_p, hhv_p):
        """MFI HHV (breakout up)"""
        key = (mfi_p, hhv_p)
        if key not in self.vol_hhv_cache:  # Reuse cache dict
            mfi = self.get_mfi(mfi_p)
            self.vol_hhv_cache[key] = np.array(HHV(mfi.tolist(), int(hhv_p)))
        return self.vol_hhv_cache.get(key)
    
    def get_mfi_llv(self, mfi_p, llv_p):
        """MFI LLV (breakout down)"""
        key = (mfi_p, llv_p)
        if key not in self.vol_llv_cache:
            mfi = self.get_mfi(mfi_p)
            self.vol_llv_cache[key] = np.array(LLV(mfi.tolist(), int(llv_p)))
        return self.vol_llv_cache.get(key)
    
    def get_volume_hhv(self, p):
        """Volume (Lot) HHV"""
        key = ('vol', p)
        if key not in self.vol_hhv_cache:
            self.vol_hhv_cache[key] = np.array(HHV(self.lots.tolist(), int(p)))
        return self.vol_hhv_cache[key]

# --- WORKER INIT ---
def worker_init():
    global g_cache
    df = load_data()
    if df is not None:
        g_cache = IndicatorCache(df)

# --- FAST BACKTEST (Strategy 2 Logic) ---
@jit(nopython=True)
def fast_backtest_strategy2(closes, highs, lows, volumes, ars_arr, hhv, llv, mom, rsi,
                            mfi_arr, mfi_hhv, mfi_llv, vol_hhv,
                            mom_p, brk_p, rsi_p, kar_al, iz_stop,
                            rsi_ob, rsi_os, use_mfi, use_vol):
    n = len(closes)
    
    # Pre-calculate Trend Direction
    # 1: Long Trend, -1: Short Trend
    # ARS logic: Close > ARS => Trend Up.
    # We pre-calculate simplistic trend: 
    trend_raw = np.zeros(n, dtype=np.int32)
    for j in range(n):
        if closes[j] > ars_arr[j]:
            trend_raw[j] = 1
        elif closes[j] < ars_arr[j]:
            trend_raw[j] = -1
        else:
            trend_raw[j] = 0
    
    pos = 0 # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    extreme_price = 0.0 # Tracking max/min price for Trailing Stop
    
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    # Constants scaled
    kar_al_ratio = kar_al / 100.0
    iz_stop_ratio = iz_stop / 100.0
    
    # Pre-calc conditions to speed up loop
    mom_long = (mom > 100)
    mom_short = (mom < 100)
    rsi_ok_long = (rsi < rsi_ob)
    rsi_ok_short = (rsi > rsi_os)
    
    # Using previous value for HHV/LLV is crucial (breakout of previous N bars)
    # hhv[i-1], llv[i-1]
    
    current_trend = 0
    
    for i in range(50, n):
        # Trend Update
        if trend_raw[i] != 0:
            current_trend = trend_raw[i]
            
        # --- EXIT LOGIC ---
        if pos == 1:
            # Update Extreme
            if closes[i] > extreme_price: extreme_price = closes[i]
            
            # 1. Trend Reversal
            if current_trend == -1: # Trend Short'a döndü
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # 2. Kar Al
            elif closes[i] >= entry_price * (1 + kar_al_ratio):
                pnl = closes[i] - entry_price # Basitce kapanis fiyati ile ciktik (slippage yok)
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # 3. Trailing Stop
            elif closes[i] < extreme_price * (1 - iz_stop_ratio):
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                
            if pos == 0:
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd

        elif pos == -1:
            # Update Extreme
            if closes[i] < extreme_price: extreme_price = closes[i]
            
            # 1. Trend Reversal
            if current_trend == 1:
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # 2. Kar Al
            elif closes[i] <= entry_price * (1 - kar_al_ratio):
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
            # 3. Trailing Stop
            elif closes[i] > extreme_price * (1 + iz_stop_ratio):
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                pos = 0
                
            if pos == 0:
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd

        # --- ENTRY LOGIC ---
        if pos == 0:
            if current_trend == 1:
                # LONG Conditions
                # Breakout: High > Prev HHV
                price_ok = (closes[i] > hhv[i-1] or highs[i] > hhv[i-1])
                
                # MFI Breakout (new): MFI >= Prev MFI HHV
                mfi_ok = True
                if use_mfi:
                    mfi_ok = mfi_arr[i] >= mfi_hhv[i-1]
                
                # Volume Breakout (new): Volume >= 80% of Prev Vol HHV
                vol_ok = True
                if use_vol:
                    vol_ok = volumes[i] >= vol_hhv[i-1] * 0.8
                
                if price_ok and mom_long[i] and rsi_ok_long[i] and mfi_ok and vol_ok:
                    pos = 1
                    entry_price = closes[i]
                    extreme_price = closes[i]
                    trades += 1
                    
            elif current_trend == -1:
                # SHORT Conditions
                price_ok = (closes[i] < llv[i-1] or lows[i] < llv[i-1])
                
                # MFI Breakout (new): MFI <= Prev MFI LLV
                mfi_ok = True
                if use_mfi:
                    mfi_ok = mfi_arr[i] <= mfi_llv[i-1]
                
                # Volume Breakout (new): Volume >= 80% of Prev Vol HHV
                vol_ok = True
                if use_vol:
                    vol_ok = volumes[i] >= vol_hhv[i-1] * 0.8
                
                if price_ok and mom_short[i] and rsi_ok_short[i] and mfi_ok and vol_ok:
                    pos = -1
                    entry_price = closes[i]
                    extreme_price = closes[i]
                    trades += 1

    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    return net_profit, trades, pf, max_dd

# --- WORKER TASK ---
def solve_chunk(args):
    ars_ema, ars_atr_p, ars_atr_m, params_grid = args
    
    global g_cache
    if g_cache is None: return [] 
    
    results = []
    
    # Extract other params
    mom_ps = params_grid['mom_ps']
    brk_ps = params_grid['brk_ps']
    kar_als = params_grid['kar_als']
    iz_stops = params_grid['iz_stops']
    
    # MFI/Volume params (new)
    mfi_period = params_grid.get('mfi_period', 14)
    use_mfi = params_grid.get('use_mfi', True)
    use_vol = params_grid.get('use_vol', True)
    vol_period = params_grid.get('vol_period', 14)
    
    # Fixed or narrow range params for RSI to reduce dim
    rsi_p = 14
    rsi_ob = 70
    rsi_os = 30
    
    closes = g_cache.closes
    volumes = g_cache.lots
    
    # Get Indicator Arrays
    ars_arr = g_cache.get_ars(ars_ema, ars_atr_p, ars_atr_m)
    rsi_arr = g_cache.get_rsi(rsi_p)
    
    # MFI/Volume arrays (new)
    mfi_arr = g_cache.get_mfi(mfi_period)
    mfi_hhv = g_cache.get_mfi_hhv(mfi_period, mfi_period)
    mfi_llv = g_cache.get_mfi_llv(mfi_period, mfi_period)
    vol_hhv = g_cache.get_volume_hhv(vol_period)
    
    for mp in mom_ps:
        mom_arr = g_cache.get_mom(mp)
        
        for bp in brk_ps:
            hhv_arr = g_cache.get_hhv(bp)
            llv_arr = g_cache.get_llv(bp)
            
            for ka in kar_als:
                for iz in iz_stops:
                    
                    np_val, tr, pf, dd = fast_backtest_strategy2(
                        closes, g_cache.highs, g_cache.lows, volumes,
                        ars_arr, hhv_arr, llv_arr, mom_arr, rsi_arr,
                        mfi_arr, mfi_hhv, mfi_llv, vol_hhv,
                        mp, bp, rsi_p, ka, iz, rsi_ob, rsi_os, use_mfi, use_vol
                    )
                    
                    if np_val > 0 and pf > 1.05 and tr > 5:
                        results.append({
                            'NP': np_val, 'PF': pf, 'DD': dd, 'Tr': tr,
                            'ARS_E': ars_ema, 'ARS_A': ars_atr_p, 'ARS_M': ars_atr_m,
                            'MOM': mp, 'BRK': bp, 'TP': ka, 'TS': iz,
                            'MFI': mfi_period, 'VOL': vol_period
                        })
                        
    return results

# --- OPTIMIZATION MANAGER ---
def run_parallel_stage(stage_name, params_grid):
    print(f"\\n--- {stage_name} BAŞLIYOR ---")
    
    ars_emas = params_grid['ars_emas']
    ars_atr_ps = params_grid['ars_atr_ps']
    ars_atr_ms = params_grid['ars_atr_ms']
    
    tasks = []
    
    for e in ars_emas:
        for p in ars_atr_ps:
            for m in ars_atr_ms:
                tasks.append((e, p, m, params_grid))
                
    print(f"Toplam Görev: {len(tasks)}")
    
    final_results = []
    start_time = time()
    
    with Pool(processes=min(32, cpu_count()), initializer=worker_init) as pool:
        for res in pool.imap_unordered(solve_chunk, tasks):
            final_results.extend(res)
            
    elapsed = time() - start_time
    print(f"Bitti. Süre: {elapsed:.1f}sn. Bulunan: {len(final_results)}")
    
    return final_results

def run_strategy2_optimization():
    if not os.path.exists("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"):
        print("Veri dosyası yok!")
        return

    # --- STAGE 1: BROAD SPECTRUM ---
    stage1_grid = {
        'ars_emas': [3, 5, 8],
        'ars_atr_ps': [10, 14],
        'ars_atr_ms': [0.5, 0.8, 1.0],
        'mom_ps': [3, 5],
        'brk_ps': [10, 20],
        'kar_als': [2.0, 3.0, 5.0],
        'iz_stops': [1.0, 2.0],
        # MFI/Volume (new)
        'mfi_period': 14,
        'vol_period': 14,
        'use_mfi': True,
        'use_vol': True
    }
    
    results1 = run_parallel_stage("STAGE 1 (UYDU)", stage1_grid)
    
    if not results1: return
    
    df1 = pd.DataFrame(results1)
    df1['Score'] = df1['NP'] * df1['PF'] / (1 + df1['DD']/1000)
    best = df1.nlargest(1, 'Score').iloc[0]
    
    print(f"\\nSTAGE 1 BEST:\\n{best.to_string()}")
    
    # --- STAGE 2: LOCAL ---
    # Zoom in around best params
    best_e = int(best['ARS_E'])
    best_ap = int(best['ARS_A'])
    
    stage2_grid = {
        'ars_emas': [best_e], 
        'ars_atr_ps': [best_ap],
        'ars_atr_ms': [best['ARS_M']], # Keep multiplier fixed or small range
        'mom_ps': [int(best['MOM'])],
        'brk_ps': [int(best['BRK'])-2, int(best['BRK']), int(best['BRK'])+2],
        'kar_als': [best['TP']-0.5, best['TP'], best['TP']+0.5],
        'iz_stops': [best['TS']-0.2, best['TS'], best['TS']+0.2],
        # MFI/Volume (same as stage1)
        'mfi_period': int(best['MFI']),
        'vol_period': int(best['VOL']),
        'use_mfi': True,
        'use_vol': True
    }
    # Note: Logic can be improved to expand ranges dynamically
    # For now, strict local search
    
    results2 = run_parallel_stage("STAGE 2 (DRONE)", stage2_grid)
    
    df2 = pd.DataFrame(results2)
    df2.sort_values('NP', ascending=False).to_csv("d:/Projects/IdealQuant/results/strategy2_final_results.csv", index=False)
    print("Sonuçlar kaydedildi.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        run_strategy2_optimization()
    except KeyboardInterrupt:
        print("İptal edildi.")
