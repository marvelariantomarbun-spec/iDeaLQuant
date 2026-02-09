# -*- coding: utf-8 -*-
"""
Smart Optimizer - Caching & Nested Loops
Referans projedeki (AHLT) mantığa uygun olarak:
1. İndikatörleri Önbellekleme (Caching)
2. İçe İçe Döngüler (Nested Loops)
3. Smart Scoring (Kar + PF + Drawdown)
"""

import sys
import os
import io
import pandas as pd
import numpy as np
from time import time
from dataclasses import dataclass

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
        
        # Tipik Fiyat
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
        self.opens = df['Acilis'].values.tolist()
        self.highs = df['Yuksek'].values.tolist()
        self.lows = df['Dusuk'].values.tolist()
        self.closes = df['Kapanis'].values.tolist()
        self.typical = df['Tipik'].values.tolist()
        self.vols = df['Hacim'].values.tolist()
        
        # Caches
        self.ars_cache = {}
        self.rvi_cache = {}
        self.adx_cache = {}
        self.qstick_cache = {}
        
        # Base Indicators (Sabit)
        self.netlot = NetLot(self.opens, self.highs, self.lows, self.closes)
        self.netlot_ma = pd.Series(self.netlot).rolling(5).mean().fillna(0).values
        
        # BB (Sabit)
        sma20 = df['Kapanis'].rolling(20).mean()
        std20 = df['Kapanis'].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        self.bb_width = ((upper - lower) / sma20 * 100).fillna(0).values
        self.bb_width_avg = pd.Series(self.bb_width).rolling(50).mean().values

    def precalculate(self, ars_emas, ars_ks, rvi_ps, adx_ps, qstick_ps):
        print("İndikatörler Önbellekleniyor...")
        t0 = time()
        
        # ARS Cache
        for p in ars_emas:
            for k in ars_ks:
                key = (p, k)
                self.ars_cache[key] = np.array(ARS(self.typical, p, k))
                
        # RVI Cache
        for p in rvi_ps:
            r, s = RVI(self.opens, self.highs, self.lows, self.closes, p)
            self.rvi_cache[p] = (np.array(r), np.array(s))
            
        # ADX Cache
        for p in adx_ps:
            self.adx_cache[p] = np.array(ADX(self.highs, self.lows, self.closes, p))
            
        # QStick Cache
        for p in qstick_ps:
            self.qstick_cache[p] = np.array(Qstick(self.opens, self.closes, p))
            
        print(f"Önbellekleme Tamamlandı: {time()-t0:.2f} sn")

# --- BACKTEST ENGINE (Fast Numpy) ---
def fast_backtest(closes, signals, exits_long, exits_short):
    pos = 0 # 0:Flat, 1:Long, -1:Short
    entry_price = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    max_dd = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    
    n = len(closes)
    
    # Numpy arrays for speed access
    # Assuming signals etc are already numpy arrays
    
    for i in range(1, n):
        price = closes[i]
        
        if pos == 0:
            sig = signals[i]
            if sig == 1:
                pos = 1
                entry_price = price
                trades += 1
            elif sig == -1:
                pos = -1
                entry_price = price
                trades += 1
                
        elif pos == 1:
            # Exit Check
            if exits_long[i]:
                pnl = price - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                
                current_equity += pnl
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd
                
                pos = 0
                
                # Re-entry check (Reverse) - Optional, here we go Flat first
                
        elif pos == -1:
            # Exit Check
            if exits_short[i]:
                pnl = entry_price - price
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

# --- MAIN OPTIMIZATION LOOP ---
def run_smart_optimization():
    df = load_data()
    if df is None: return
    
    cache = IndicatorCache(df)
    
    # --- PARAMETER RANGES (Geniş Aralık) ---
    ars_emas = [3, 5, 8]
    ars_ks = [0.01, 0.0123, 0.015, 0.02]
    rvi_ps = [10, 14, 21]
    adx_ps = [14, 21, 30]
    qstick_ps = [5, 8, 13]
    
    # Eşikler
    min_scores = [3, 4]
    netlots = [10, 20] # Sadece 2 seçenek yeterli
    
    # Pre-calculate
    cache.precalculate(ars_emas, ars_ks, rvi_ps, adx_ps, qstick_ps)
    
    closes = df['Kapanis'].values
    best_result = {'Score': -999}
    results = []
    
    # --- NESTED LOOPS ---
    print("\n--- MATRIX TARAMA BAŞLIYOR ---")
    start_time = time()
    counter = 0
    total_iters = len(ars_emas)*len(ars_ks)*len(rvi_ps)*len(adx_ps)*len(qstick_ps)*len(min_scores)*len(netlots)
    
    print(f"Toplam Kombinasyon: {total_iters}")
    
    # 1. Loop ARS
    for ars_p in ars_emas:
        for ars_k in ars_ks:
            ars_arr = cache.ars_cache[(ars_p, ars_k)]
            ars_diff = np.diff(ars_arr, prepend=0) != 0
            # Rolling sum is tricky in purely vectorized loop, but cache helps
            # Calculation of 'Yatay Filtre Components' dependent on ARS
            ars_mesafe = np.abs(closes - ars_arr) / np.where(ars_arr!=0, ars_arr, 1) * 100
            
            # Bu kısım biraz maliyetli, ama loop içinde optimize edilebilir
            # ARS Degisti (Rolling)
            ars_degisti = pd.Series(ars_diff).rolling(10).sum().gt(0).values.astype(int)
            
            # 2. Loop ADX
            for adx_p in adx_ps:
                adx_arr = cache.adx_cache[adx_p]
                
                # Yatay Filtre Maskesi (Kısmi)
                # (ARS_Degisti + ARS_Mesafe + ADX)
                # BB Width fixed
                
                # Mask'i her loopta hesaplamak yerine parça parça
                f1 = ars_degisti
                f2 = (ars_mesafe > 0.25).astype(int)
                f3 = (adx_arr > 20.0).astype(int)
                f4 = (cache.bb_width > cache.bb_width_avg * 0.8).astype(int)
                yatay_filtre = (f1 + f2 + f3 + f4) >= 2
                
                # 3. Loop RVI
                for rvi_p in rvi_ps:
                    rvi_val, rvi_sig = cache.rvi_cache[rvi_p]
                    rvi_long = (rvi_val > rvi_sig).astype(int)
                    rvi_short = (rvi_val < rvi_sig).astype(int)
                    
                    # 4. Loop QStick
                    for qs_p in qstick_ps:
                        qs_val = cache.qstick_cache[qs_p]
                        qs_long = (qs_val > 0).astype(int)
                        qs_short = (qs_val < 0).astype(int)
                        
                        # Skor Hesaplama (Vektörel)
                        # Her şey numpy array
                        ars_long = (closes > ars_arr).astype(int)
                        ars_short = (closes < ars_arr).astype(int)
                        
                        base_l = ars_long + rvi_long + qs_long
                        base_s = ars_short + rvi_short + qs_short
                        
                        # 5. Loop NetLot & Min Score
                        for nl_th in netlots:
                            nl_long = (cache.netlot_ma > nl_th).astype(int)
                            nl_short = (cache.netlot_ma < -nl_th).astype(int)
                            
                            # ADX Threshold fixed 25 in score
                            adx_score = (adx_arr > 25.0).astype(int)
                            
                            final_l_score = base_l + nl_long + adx_score
                            final_s_score = base_s + nl_short + adx_score
                            
                            for min_sc in min_scores:
                                # Sinyalleri Oluştur
                                sigs = np.zeros(len(closes), dtype=int)
                                
                                # Entry
                                long_cond = yatay_filtre & (final_l_score >= min_sc) & (final_s_score < 2)
                                short_cond = yatay_filtre & (final_s_score >= min_sc) & (final_l_score < 2)
                                
                                sigs[long_cond] = 1
                                sigs[short_cond] = -1
                                
                                # Exit
                                ex_l = (closes < ars_arr) | (final_s_score >= 4)
                                ex_s = (closes > ars_arr) | (final_l_score >= 4)
                                
                                # Backtest
                                np_val, tr, pf, dd = fast_backtest(closes, sigs, ex_l, ex_s)
                                
                                # --- SMART SCORE ---
                                # Profit Factor, Drawdown ve Net Kar dengesi
                                # Basit Smart Score: Profit * PF / (1 + DD/1000)
                                smart_score = np_val * pf / (1 + dd/5000)
                                
                                if np_val > 1500: # Filtre
                                    res = {
                                        'NetProfit': np_val,
                                        'PF': pf,
                                        'MaxDD': dd,
                                        'Trades': tr,
                                        'SmartScore': smart_score,
                                        'Params': f"ARS({ars_p},{ars_k}) ADX({adx_p}) RVI({rvi_p}) Q({qs_p}) Sc({min_sc}) NL({nl_th})"
                                    }
                                    results.append(res)
                                    
                                    if smart_score > best_result['Score']:
                                        best_result = {'Score': smart_score, 'Res': res}
                                
                                counter += 1
                                if counter % 100 == 0:
                                    print(f"Prog: {counter}/{total_iters} | Best: {best_result.get('Res', {}).get('NetProfit', 0):.0f}", end='\r')

    print(f"\n\nTamamlandı! Süre: {time()-start_time:.2f} sn")
    
    if len(results) > 0:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values('SmartScore', ascending=False)
        print("\n--- EN İYİ 10 'SMART' SONUÇ ---")
        print(res_df.head(10).to_string(index=False))
        res_df.to_csv("d:/Projects/IdealQuant/tests/smart_optimization_results.csv", index=False)
    else:
        print("Sonuç yok.")

if __name__ == "__main__":
    run_smart_optimization()
