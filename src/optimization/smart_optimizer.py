# -*- coding: utf-8 -*-
"""
Smart Optimizer v2 - Two-Stage & Stability Analysis
Hedef: Yüksek PF ve Overfit Olmayan (Stabil) Parametreler
Yöntem: 
1. Coarse Search (Geniş Aralık, Büyük Adım)
2. Fine Search (En İyi Bölgelerde Hassas Tarama)
3. Stability Score (Komşu Parametrelerin Başarısı)
"""

import sys
import os
import io
import pandas as pd
import numpy as np
from time import time
import itertools

# Proje kök dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indicators.core import EMA, ATR, ADX, SMA, ARS, RVI, Qstick, NetLot

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- DATA LOADING ---
def load_data():
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        print("Veri Yükleniyor...")
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
        df.dropna(inplace=True)
        print(f"Veri Hazır: {len(df)} Bar")
        return df
    except Exception as e:
        print(f"Hata: {e}")
        return None

# --- CACHING SYSTEM ---
class IndicatorCache:
    def __init__(self, df):
        self.df = df
        self.opens = df['Acilis'].values
        self.highs = df['Yuksek'].values
        self.lows = df['Dusuk'].values
        self.closes = df['Kapanis'].values
        self.typical = df['Tipik'].values
        
        self.ars_cache = {}
        self.rvi_cache = {}
        self.adx_cache = {}
        self.qstick_cache = {}
        
        # Base Indicators
        self.netlot = NetLot(self.opens.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist())
        self.netlot_ma = pd.Series(self.netlot).rolling(5).mean().fillna(0).values
        
        sma20 = df['Kapanis'].rolling(20).mean()
        std20 = df['Kapanis'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        self.bb_width = ((upper - lower) / sma20 * 100).fillna(0).values
        self.bb_width_avg = pd.Series(self.bb_width).rolling(50).mean().values

    # Dinamik Cache: İstenen parametre yoksa hesapla ekle
    def get_ars(self, p, k):
        key = (p, round(k, 4))
        if key not in self.ars_cache:
            self.ars_cache[key] = np.array(ARS(self.typical.tolist(), int(p), float(key[1])))
        return self.ars_cache[key]

    def get_rvi(self, p):
        if p not in self.rvi_cache:
            r, s = RVI(self.opens.tolist(), self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(p))
            self.rvi_cache[p] = (np.array(r), np.array(s))
        return self.rvi_cache[p]

    def get_adx(self, p):
        if p not in self.adx_cache:
            self.adx_cache[p] = np.array(ADX(self.highs.tolist(), self.lows.tolist(), self.closes.tolist(), int(p)))
        return self.adx_cache[p]
        
    def get_qstick(self, p):
        if p not in self.qstick_cache:
            self.qstick_cache[p] = np.array(Qstick(self.opens.tolist(), self.closes.tolist(), int(p)))
        return self.qstick_cache[p]

# --- FAST BACKTEST ---
def fast_backtest(closes, signals, exits_long, exits_short):
    # Vectorized Simulation Simulation
    # positions: 1 (Long), -1 (Short), 0 (Flat)
    # This is a semi-vectorized loop for speed
    
    n = len(closes)
    # Using a simple loop is often faster than complex vector logic for state-dependent backtests
    # But we need ultra speed.
    
    # State tracking
    pos = 0
    entry_price = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    # PnL Curve for DD calculation
    
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

# --- OPTIMIZATION ENGINE ---
def run_optimization_stage(stage_name, cache, params_grid, thresholds):
    print(f"\n--- {stage_name} BAŞLIYOR ---")
    
    # Grid Unpacking
    ars_emas = params_grid['ars_emas']
    ars_ks = params_grid['ars_ks']
    rvi_ps = params_grid['rvi_ps']
    adx_ps = params_grid['adx_ps']
    qstick_ps = params_grid['qstick_ps']
    
    min_scores = thresholds['min_scores']
    netlots = thresholds['netlots']
    
    total_combs = len(ars_emas)*len(ars_ks)*len(rvi_ps)*len(adx_ps)*len(qstick_ps)*len(min_scores)*len(netlots)
    print(f"Taranacak Kombinasyon: {total_combs}")
    
    results = []
    counter = 0
    start_time = time()
    best_np = -99999
    
    closes = cache.closes
    
    # Nested Loops (Gerekirse itertools.product ile optimize edilebilir ama manual loop daha kontrollü)
    for ars_p in ars_emas:
        for ars_k in ars_ks:
            ars_arr = cache.get_ars(ars_p, ars_k)
            # Pre-calc dependent arrays
            ars_diff = np.diff(ars_arr, prepend=0) != 0
            ars_degisti = pd.Series(ars_diff).rolling(10).sum().gt(0).values.astype(int)
            ars_mesafe = np.abs(closes - ars_arr) / np.where(ars_arr!=0, ars_arr, 1) * 100
            
            ars_long = (closes > ars_arr).astype(int)
            ars_short = (closes < ars_arr).astype(int)
            
            for adx_p in adx_ps:
                adx_arr = cache.get_adx(adx_p)
                adx_score = (adx_arr > 25.0).astype(int) # Score comp
                
                # Yatay Filtre
                f1 = ars_degisti
                f2 = (ars_mesafe > 0.25).astype(int)
                f3 = (adx_arr > 20.0).astype(int)
                f4 = (cache.bb_width > cache.bb_width_avg * 0.8).astype(int)
                yatay_filtre = (f1 + f2 + f3 + f4) >= 2
                
                for rvi_p in rvi_ps:
                    rvi_val, rvi_sig = cache.get_rvi(rvi_p)
                    rvi_long = (rvi_val > rvi_sig).astype(int)
                    rvi_short = (rvi_val < rvi_sig).astype(int)
                    
                    for qs_p in qstick_ps:
                        qs_val = cache.get_qstick(qs_p)
                        qs_long = (qs_val > 0).astype(int)
                        qs_short = (qs_val < 0).astype(int)
                        
                        base_l = ars_long + rvi_long + qs_long
                        base_s = ars_short + rvi_short + qs_short
                        
                        for nl_th in netlots:
                            nl_long = (cache.netlot_ma > nl_th).astype(int)
                            nl_short = (cache.netlot_ma < -nl_th).astype(int)
                            
                            final_l_score = base_l + nl_long + adx_score
                            final_s_score = base_s + nl_short + adx_score
                            
                            for min_sc in min_scores:
                                # Apply Logic
                                sigs = np.zeros(len(closes), dtype=int)
                                
                                l_cond = yatay_filtre & (final_l_score >= min_sc) & (final_s_score < 2)
                                s_cond = yatay_filtre & (final_s_score >= min_sc) & (final_l_score < 2)
                                
                                sigs[l_cond] = 1
                                sigs[s_cond] = -1
                                
                                # Exit
                                ex_l = (closes < ars_arr) | (final_s_score >= 4)
                                ex_s = (closes > ars_arr) | (final_l_score >= 4)
                                
                                np_val, tr, pf, dd = fast_backtest(closes, sigs, ex_l, ex_s)
                                
                                if np_val > 2000 and pf > 1.10: # Filtre
                                    results.append({
                                        'NP': np_val,
                                        'PF': pf,
                                        'DD': dd,
                                        'Tr': tr,
                                        'AP': ars_p, 'AK': ars_k,
                                        'RP': rvi_p, 'QP': qs_p, 'ADP': adx_p,
                                        'SC': min_sc, 'NL': nl_th
                                    })
                                    if np_val > best_np: best_np = np_val
                                
                                counter += 1
                                if counter % 2000 == 0:
                                    print(f"Prog: {counter}/{total_combs} | Best: {best_np:.0f} | Found: {len(results)}", end='\r')
                                    
    print(f"\n{stage_name} Bitti. Süre: {time()-start_time:.1f}sn. Bulunan: {len(results)}")
    return results

def run_two_stage_process():
    df = load_data()
    if df is None: return
    cache = IndicatorCache(df)
    
    # --- STAGE 1: COARSE SEARCH (Geniş Aralık) ---
    stage1_grid = {
        'ars_emas': list(range(3, 22, 3)),   # 3, 6, 9... 21
        'ars_ks': [0.005, 0.01, 0.015, 0.02, 0.025], # Seyrek
        'rvi_ps': list(range(5, 55, 5)),     # 5, 10... 50
        'adx_ps': list(range(10, 65, 10)),   # 10, 20... 60
        'qstick_ps': list(range(5, 45, 5))   # 5, 10... 40
    }
    stage1_th = {
        'min_scores': [3, 4],
        'netlots': [10, 20]
    }
    
    results1 = run_optimization_stage("STAGE 1 (Coarse)", cache, stage1_grid, stage1_th)
    
    if not results1: 
        print("Stage 1 sonuç döndürmedi.")
        return

    # --- ANALIZ & HASSAS ARAMA ALANI BELİRLEME ---
    df_res1 = pd.DataFrame(results1)
    # Smart Score for Stage 1
    df_res1['Score'] = df_res1['NP'] * df_res1['PF'] / (1 + df_res1['DD']/5000)
    
    # En iyi %5'lik dilimi al
    top_results = df_res1.nlargest(int(len(df_res1)*0.05 + 10), 'Score')
    
    # Parametre Merkezlerini Bul (Cluster Center)
    # Basitçe: En iyi sonucun etrafını tara
    best_row = top_results.iloc[0]
    print(f"\n--- STAGE 1 KAZANANI ---\n{best_row.to_string()}")
    
    # --- STAGE 2: FINE SEARCH (Hassas Odaklanma) ---
    # En iyi değerin +/- civarını tara
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
        'ars_emas': make_fine_range(best_row['AP'], 1, 2),        # +/- 2
        'ars_ks': make_fine_range(best_row['AK'], 0.001, 3, True), # +/- 0.003
        'rvi_ps': make_fine_range(best_row['RP'], 1, 3),          # +/- 3
        'adx_ps': make_fine_range(best_row['ADP'], 2, 2),         # +/- 4
        'qstick_ps': make_fine_range(best_row['QP'], 1, 2)        # +/- 2
    }
    # Eşikler sabit (Best row'dan al)
    stage2_th = {
        'min_scores': [int(best_row['SC'])], # Sadece kazanan skoru tara, belki +/- 1? Hayır sabit
        'netlots': [int(best_row['NL'])]
    }
    
    print("\nStage 2 Grid Hedefleri:")
    print(stage2_grid)
    
    results2 = run_optimization_stage("STAGE 2 (Fine)", cache, stage2_grid, stage2_th)
    
    # --- STABILITY ANALYSIS ---
    df_res2 = pd.DataFrame(results2)
    df_res2['Score'] = df_res2['NP'] * df_res2['PF'] / (1 + df_res2['DD']/5000)
    
    # Kararlılık Puanı Ekle:
    # Her satır için, parametreleri yakın olan (veya aynı olan) komşuların ortalama başarısını bul
    # Burada grid küçük olduğu için tüm havuzun ortalaması zaten bir stabilite göstergesi olabilir
    # Ama biz en iyi sonucun, ortalamadan ne kadar saptığına (Outlier mı?) bakacağız.
    
    print("\n--- SONUÇLAR VE STABİLİTE ---")
    df_res2 = df_res2.sort_values('Score', ascending=False)
    
    # Stability Check for Top 5
    for i in range(min(5, len(df_res2))):
        row = df_res2.iloc[i]
        # Cluster Mean (Tüm Fine Search sonuçlarının ortalaması - çünkü hepsi komşu)
        cluster_mean_np = df_res2['NP'].mean()
        cluster_mean_pf = df_res2['PF'].mean()
        
        stability_ratio = row['NP'] / cluster_mean_np # 1'e yakınsa çok stabil (küme ile aynı), yüksekse outlier
        
        print(f"Rank {i+1}: NP={row['NP']:.0f} PF={row['PF']:.2f} DD={row['DD']:.0f} | StabRatio={stability_ratio:.2f}")
        print(f"       Params: ARS({row['AP']},{row['AK']}) ADX({row['ADP']}) RVI({row['RP']}) Q({row['QP']})")
        
    df_res2.to_csv("d:/Projects/IdealQuant/tests/two_stage_optimization_results.csv", index=False)

if __name__ == "__main__":
    run_two_stage_process()
