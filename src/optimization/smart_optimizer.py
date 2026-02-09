# -*- coding: utf-8 -*-
"""
Smart Optimizer v3.1 - 3-Stage Comprehensive
Hedef: ARS + MACD-V + ADX + NetLot için tam kapsamlı tarama.
Stage 1: Geniş Tarama (Tüm parametreler)
Stage 2: Hassas Tarama (En iyi bölge)
Stage 3: Stabilite Analizi
"""

import sys
import os
import io
import pandas as pd
import numpy as np
from time import time
import itertools
from multiprocessing import Pool, cpu_count, current_process

# Proje kök dizini (IdealQuant)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.indicators.core import EMA, ATR, ADX, SMA, ARS, NetLot, MACDV

# Global cache for workers
g_cache = None

# --- DATA LOADING ---
def load_data():
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        if current_process().name == 'MainProcess':
            print("Veri yukleniyor...")
            
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
        df.dropna(inplace=True)
        
        if current_process().name == 'MainProcess':
            print(f"Veri Hazir: {len(df)} Bar")
            
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
        
        self.ars_cache = {}
        self.adx_cache = {}
        self.macdv_cache = {}
        
        # Base Indicators (Pre-calculated once)
        self.netlot = NetLot(self.opens.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist())
        self.netlot_ma = pd.Series(self.netlot).rolling(5).mean().fillna(0).values
        
        sma20 = df['Kapanis'].rolling(20).mean()
        std20 = df['Kapanis'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        self.bb_width = ((upper - lower) / sma20 * 100).fillna(0).values
        self.bb_width_avg = pd.Series(self.bb_width).rolling(50).mean().values

    def get_ars(self, p, k):
        key = (p, round(k, 4))
        if key not in self.ars_cache:
            self.ars_cache[key] = np.array(ARS(self.typical.tolist(), int(p), float(key[1])))
        return self.ars_cache[key]

    def get_adx(self, p):
        if key := p not in self.adx_cache: # Typo fix in next line
             self.adx_cache[p] = np.array(ADX(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(p)))
        return self.adx_cache[p]

    def get_macdv(self, s, l, sig):
        key = (s, l, sig)
        if key not in self.macdv_cache:
            m, sg = MACDV(self.closes.tolist(), self.highs.tolist(), self.lows.tolist(), 
                          int(s), int(l), int(sig))
            self.macdv_cache[key] = (np.array(m), np.array(sg))
        return self.macdv_cache[key]

# --- WORKER INIT ---
def worker_init():
    global g_cache
    df = load_data()
    if df is not None:
        g_cache = IndicatorCache(df)

# --- FAST BACKTEST ---
def fast_backtest(closes, signals, exits_long, exits_short):
    n = len(closes)
    pos = 0
    entry_price = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    for i in range(1, n):
        if pos == 0:
            if signals[i] == 1:
                pos = 1
                entry_price = closes[i]
                trades += 1
            elif signals[i] == -1:
                pos = -1
                entry_price = closes[i]
                trades += 1
        elif pos == 1:
            if exits_long[i]:
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd
                pos = 0
        elif pos == -1:
            if exits_short[i]:
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                current_equity += pnl
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd
                pos = 0
                
    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    return net_profit, trades, pf, max_dd

# --- WORKER TASK ---
def solve_chunk(args):
    ars_p, ars_k, params_grid, thresholds = args
    
    global g_cache
    if g_cache is None: return [] 
    
    results = []
    
    adx_ps = params_grid['adx_ps']
    
    # MACD-V Grid Expansion
    # macdv_grid is a list of tuples: [(s, l, sig), ...]
    macdv_settings = params_grid['macdv_set']
    
    min_scores = thresholds['min_scores']
    netlots = thresholds['netlots']
    exit_scores = thresholds.get('exit_scores', [3]) 
    
    closes = g_cache.closes
    
    # ARS Calc
    ars_arr = g_cache.get_ars(ars_p, ars_k)
    ars_diff = np.diff(ars_arr, prepend=0) != 0
    ars_degisti = pd.Series(ars_diff).rolling(10).sum().gt(0).values.astype(int)
    
    ars_mesafe = np.abs(closes - ars_arr) / np.where(ars_arr!=0, ars_arr, 1) * 100
    ars_long = (closes > ars_arr).astype(int)
    ars_short = (closes < ars_arr).astype(int)
    
    for adx_p in adx_ps:
        adx_arr = g_cache.get_adx(adx_p)
        adx_score = (adx_arr > 25.0).astype(int)
        
        # Yatay Filtre Components
        f1 = ars_degisti
        f2 = (ars_mesafe > 0.25).astype(int)
        f3 = (adx_arr > 20.0).astype(int)
        f4 = (g_cache.bb_width > g_cache.bb_width_avg * 0.8).astype(int)
        yatay_filtre = (f1 + f2 + f3 + f4) >= 2
        
        base_l = ars_long
        base_s = ars_short
        
        for macd_p in macdv_settings:
            # macd_p = (short, long, signal)
            macdv_val, macdv_sig = g_cache.get_macdv(macd_p[0], macd_p[1], macd_p[2])
            
            macdv_long = (macdv_val > macdv_sig).astype(int)
            macdv_short = (macdv_val < macdv_sig).astype(int)
        
            for nl_th in netlots:
                nl_long = (g_cache.netlot_ma > nl_th).astype(int)
                nl_short = (g_cache.netlot_ma < -nl_th).astype(int)
                
                final_l_score = base_l + macdv_long + nl_long + adx_score
                final_s_score = base_s + macdv_short + nl_short + adx_score
                
                for min_sc in min_scores:
                    for ex_sc in exit_scores:
                        sigs = np.zeros(len(closes), dtype=int)
                        
                        l_cond = yatay_filtre & (final_l_score >= min_sc) & (final_s_score < 2)
                        s_cond = yatay_filtre & (final_s_score >= min_sc) & (final_l_score < 2)
                        
                        sigs[l_cond] = 1
                        sigs[s_cond] = -1
                        
                        ex_l = (closes < ars_arr) | (final_s_score >= ex_sc)
                        ex_s = (closes > ars_arr) | (final_l_score >= ex_sc)
                        
                        np_val, tr, pf, dd = fast_backtest(closes, sigs, ex_l, ex_s)
                        
                        # Filter bad results early
                        if np_val > 5000 and pf > 1.20 and dd < 2000:
                            results.append({
                                'NP': np_val, 'PF': pf, 'DD': dd, 'Tr': tr,
                                'AP': ars_p, 'AK': ars_k,
                                'ADP': adx_p,
                                'MV': macd_p, # (s, l, sig)
                                'SC': min_sc, 'EX_SC': ex_sc, 'NL': nl_th
                            })
                            
    return results

# --- OPTIMIZATION MANAGER ---
def run_parallel_stage(stage_name, params_grid, thresholds):
    print(f"\n--- {stage_name} (PARALLEL) BASLIYOR ---")
    
    ars_emas = params_grid['ars_emas']
    ars_ks = params_grid['ars_ks']
    
    tasks = []
    
    macdv_count = len(params_grid['macdv_set'])
    inner_combs = len(params_grid['adx_ps']) * macdv_count * \
                  len(thresholds['min_scores']) * len(thresholds['netlots']) * \
                  len(thresholds.get('exit_scores', [1]))
                  
    for ars_p in ars_emas:
        for ars_k in ars_ks:
            tasks.append((ars_p, ars_k, params_grid, thresholds))
            
    total_combs = inner_combs * len(tasks)
    print(f"Toplam Gorev: {len(tasks)} | Toplam Kombinasyon: {total_combs:,}")
    print(f"Kullanılan CPU: {cpu_count()}")

    final_results = []
    start_time = time()
    
    with Pool(processes=min(32, cpu_count()), initializer=worker_init) as pool:
        for res in pool.imap_unordered(solve_chunk, tasks):
            final_results.extend(res)
            
    elapsed = time() - start_time
    if elapsed > 0:
        print(f"\n{stage_name} Bitti. Sure: {elapsed:.1f}sn. Hiz: {total_combs/elapsed:.0f} comb/s. Bulunan: {len(final_results)}")
    
    return final_results

def run_3_stage_process():
    if not os.path.exists("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"):
        print("Veri dosyası yok!")
        return

    # --- STAGE 1: BROAD SPECTRUM ---
    # Goal: Scan widely to find "Regions of Interest"
    macdv_combinations = []
    for s in [10, 12, 14]:
        for l in [24, 26, 28]:
             # Simple logic: Long > Short + 5 to avoid overlap
             if l > s + 5:
                 for sig in [7, 9, 11]:
                     macdv_combinations.append((s, l, sig))
    
    stage1_grid = {
        'ars_emas': list(range(3, 16, 3)), # 3, 6, 9, 12, 15
        'ars_ks': [0.005, 0.01, 0.015, 0.02],
        'adx_ps': [15, 25, 35, 45],
        'macdv_set': macdv_combinations # ~27 combos
    }
    stage1_th = {
        'min_scores': [2, 3],
        'netlots': [10, 20],
        'exit_scores': [3] # Fix to 3 for Stage 1 to reduce space, or [2,3]
    }
    
    results1 = run_parallel_stage("STAGE 1 (UYDU)", stage1_grid, stage1_th)
    
    if not results1: 
        print("Stage 1 sonuc dondurmedi.")
        return

    # --- ANALIZ ---
    df_res1 = pd.DataFrame(results1)
    df_res1['Score'] = df_res1['NP'] * df_res1['PF'] / (1 + df_res1['DD']/5000)
    top_results = df_res1.nlargest(20, 'Score') # Top 20 for Stage 2
    best_row = top_results.iloc[0]
    print(f"\n--- STAGE 1 LIDERI ---\n{best_row.to_string()}")
    
    # --- STAGE 2: LOCAL ZOOM ---
    # Take the best params and scan neighbors
    def get_range(val, step, count=2, is_float=False):
        if is_float:
            return [round(val - step*count + i*step, 4) for i in range(count*2+1) if val - step*count + i*step > 0]
        else:
            return [int(val - count + i) for i in range(count*2+1) if val - count + i > 1]
            
    best_mv = best_row['MV'] # Tuple (s, l, sig)
    
    # Generate neighbor MACD settings
    stage2_macdv = []
    for s_off in [-1, 0, 1]:
        for l_off in [-1, 0, 1]:
            for sig_off in [-1, 0, 1]:
                s = best_mv[0] + s_off
                l = best_mv[1] + l_off
                sig = best_mv[2] + sig_off
                if l > s+2 and s>2 and sig>2:
                    stage2_macdv.append((s, l, sig))
    stage2_macdv = list(set(stage2_macdv)) # Unique
    
    stage2_grid = {
        'ars_emas': get_range(best_row['AP'], 1, 2),
        'ars_ks': get_range(best_row['AK'], 0.001, 3, True),
        'adx_ps': get_range(best_row['ADP'], 2, 3), # Wider range for ADX
        'macdv_set': stage2_macdv
    }
    
    stage2_th = {
        'min_scores': [int(best_row['SC'])],
        'netlots': [10, 20],
        'exit_scores': [2, 3, 4] # Check exit score widely in Stage 2
    }
    
    results2 = run_parallel_stage("STAGE 2 (DRONE)", stage2_grid, stage2_th)
    
    # --- STAGE 3: STABILITY ---
    # Analyze the Stage 2 results to find the most stable cluster
    print("\n--- STAGE 3: STABILITE ANALIZI ---")
    df_res2 = pd.DataFrame(results2)
    df_res2['Score'] = df_res2['NP'] * df_res2['PF'] / (1 + df_res2['DD']/5000)
    df_res2 = df_res2.sort_values('Score', ascending=False)
    
    for i in range(min(5, len(df_res2))):
        row = df_res2.iloc[i]
        print(f"Rank {i+1}: NP={row['NP']:.0f} PF={row['PF']:.2f} DD={row['DD']:.0f}")
        print(f"       Params: ARS({row['AP']},{row['AK']}) ADX({row['ADP']}) MV{row['MV']} EXIT:{row['EX_SC']}")
        
    df_res2.to_csv("d:/Projects/IdealQuant/results/strategy1_final_results.csv", index=False)
    
    # Save top candidates for analysis
    df_res2.head(50).to_csv("d:/Projects/IdealQuant/results/strategy1_top50.csv", index=False)
    print(f"\nSonuclar kaydedildi: d:/Projects/IdealQuant/results/strategy1_final_results.csv")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        run_3_stage_process()
    except KeyboardInterrupt:
        print("Islem kullanici tarafindan durduruldu.")
