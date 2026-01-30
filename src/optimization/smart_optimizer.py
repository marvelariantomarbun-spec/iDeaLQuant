# -*- coding: utf-8 -*-
"""
Smart Optimizer v3 - Simplified & Parallel
Hedef: ARS + MACD-V + ADX + NetLot (RVI ve QStick çıkartıldı)
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
        if p not in self.adx_cache:
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
    else:
        print("Worker veriyi yükleyemedi!")

# --- FAST BACKTEST (JIT Candidate) ---
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
    """
    Tek bir ARS çifti (P, K) için iç döngüleri çalıştırır.
    Sadeleştirilmiş Versiyon: RVI ve QStick Yok
    """
    ars_p, ars_k, params_grid, thresholds = args
    
    global g_cache
    if g_cache is None: return [] 
    
    results = []
    
    # Unpack inner grids
    adx_ps = params_grid['adx_ps']
    macdv_settings = params_grid.get('macdv_set', [(12, 26, 9)]) 
    
    min_scores = thresholds['min_scores']
    netlots = thresholds['netlots']
    exit_scores = thresholds.get('exit_scores', [4]) 
    
    closes = g_cache.closes
    
    # ARS Calc
    ars_arr = g_cache.get_ars(ars_p, ars_k)
    ars_diff = np.diff(ars_arr, prepend=0) != 0
    ars_degisti = pd.Series(ars_diff).rolling(10).sum().gt(0).values.astype(int)
    
    # Pre-calc arrays
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
        
        # Base Score (ARS Only)
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
                
                # TOTAL SCORE: ARS(1) + MACD(1) + NL(1) + ADX(1) = 4
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
                        
                        if np_val > 2000 and pf > 1.10:
                            results.append({
                                'NP': np_val, 'PF': pf, 'DD': dd, 'Tr': tr,
                                'AP': ars_p, 'AK': ars_k,
                                'ADP': adx_p,
                                'MV': macd_p,
                                'SC': min_sc, 'EX_SC': ex_sc, 'NL': nl_th
                            })
                            
    return results

# --- OPTIMIZATION MANAGER ---
def run_parallel_stage(stage_name, params_grid, thresholds):
    print(f"\\n--- {stage_name} (PARALLEL) BAŞLIYOR ---")
    
    ars_emas = params_grid['ars_emas']
    ars_ks = params_grid['ars_ks']
    
    tasks = []
    
    # Inner combs calculation
    inner_combs = len(params_grid['adx_ps']) * \
                  len(params_grid.get('macdv_set', [1])) * \
                  len(thresholds['min_scores']) * len(thresholds['netlots']) * \
                  len(thresholds.get('exit_scores', [1]))
                  
    for ars_p in ars_emas:
        for ars_k in ars_ks:
            tasks.append((ars_p, ars_k, params_grid, thresholds))
            
    print(f"Toplam Görev: {len(tasks)} | Tahmini İç Kombinasyon: {inner_combs*len(tasks)}")
    print(f"Kullanılan CPU: {cpu_count()}")

    final_results = []
    start_time = time()
    
    with Pool(processes=min(32, cpu_count()), initializer=worker_init) as pool:
        for res in pool.imap_unordered(solve_chunk, tasks):
            final_results.extend(res)
            
    elapsed = time() - start_time
    total = inner_combs * len(tasks)
    if elapsed > 0:
        print(f"\\n{stage_name} Bitti. Süre: {elapsed:.1f}sn. Hız: {total/elapsed:.0f} comb/s. Bulunan: {len(final_results)}")
    
    return final_results

def run_two_stage_process():
    if not os.path.exists("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"):
        print("Veri dosyası yok!")
        return

    # --- STAGE 1: COARSE SEARCH ---
    stage1_grid = {
        'ars_emas': list(range(3, 22, 3)),
        'ars_ks': [0.005, 0.01, 0.015, 0.02, 0.025],
        'adx_ps': list(range(10, 65, 10)),
        'macdv_set': [(12, 26, 9)]
    }
    stage1_th = {
        'min_scores': [2, 3, 4], # Reduced because total indicators = 4
        'netlots': [10, 20],
        'exit_scores': [2, 3, 4] 
    }
    
    results1 = run_parallel_stage("STAGE 1", stage1_grid, stage1_th)
    
    if not results1: 
        print("Stage 1 sonuç döndürmedi.")
        return

    # --- ANALIZ ---
    df_res1 = pd.DataFrame(results1)
    # Score formula: Favor Profit factor more
    df_res1['Score'] = df_res1['NP'] * df_res1['PF'] / (1 + df_res1['DD']/5000)
    
    top_results = df_res1.nlargest(int(len(df_res1)*0.05 + 10), 'Score')
    best_row = top_results.iloc[0]
    print(f"\\n--- STAGE 1 KAZANANI ---\\n{best_row.to_string()}")
    
    # --- STAGE 2: FINE SEARCH ---
    def make_fine_range(center, step, count=2, is_float=False):
        if is_float:
            start = center - (step * count)
            vals = [start + i*step for i in range(count*2 + 1)]
            return [round(v, 4) for v in vals if v > 0]
        else:
            start = int(center) - count
            vals = [start + i for i in range(count*2 + 1)]
            return [int(v) for v in vals if v > 1]
            
    stage2_grid = {
        'ars_emas': make_fine_range(best_row['AP'], 1, 2),
        'ars_ks': make_fine_range(best_row['AK'], 0.001, 3, True),
        'adx_ps': make_fine_range(best_row['ADP'], 2, 2),
        'macdv_set': [(12, 26, 9)] 
    }
    
    stage2_th = {
        'min_scores': [int(best_row['SC'])],
        'netlots': [int(best_row['NL'])],
        'exit_scores': [int(best_row['EX_SC'])]
    }
    
    results2 = run_parallel_stage("STAGE 2", stage2_grid, stage2_th)
    
    # --- STABILITY ANALYSIS ---
    df_res2 = pd.DataFrame(results2)
    df_res2['Score'] = df_res2['NP'] * df_res2['PF'] / (1 + df_res2['DD']/5000)
    df_res2 = df_res2.sort_values('Score', ascending=False)
    
    print("\\n--- SONUÇLAR VE STABİLİTE ---")
    for i in range(min(5, len(df_res2))):
        row = df_res2.iloc[i]
        print(f"Rank {i+1}: NP={row['NP']:.0f} PF={row['PF']:.2f} DD={row['DD']:.0f}")
        print(f"       Params: ARS({row['AP']},{row['AK']}) ADX({row['ADP']}) MV{row['MV']}")
        
    df_res2.to_csv("d:/Projects/IdealQuant/tests/parallel_optimization_results.csv", index=False)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_two_stage_process()
