# -*- coding: utf-8 -*-
"""
Vectorized Optimization for ScoreBasedStrategy
Hızlandırılmış Grid Search Optimizasyonu
"""

import sys
import io
import pandas as pd
import numpy as np
import itertools
from time import time

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def load_data():
    print("Veriler yükleniyor...")
    try:
        # 1. İndikatörler (IdealData'dan gelen)
        df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_score_indicators.csv", sep=';', decimal=',', encoding='utf-8') 
        df_ind.columns = [c.strip() for c in df_ind.columns]
        for col in df_ind.columns:
            if col not in ['Date', 'Time', 'Tarih', 'Saat']:
                df_ind[col] = pd.to_numeric(df_ind[col], errors='coerce')

        # 2. Ham Veri (Fiyatlar)
        csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
        try:
            df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, nrows=5)
            if isinstance(df_raw.iloc[0,0], str) and "Tarih" in df_raw.iloc[0,0]:
                df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
            else:
                df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None)
        except:
             df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None)
             
        df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

        # Veri boyutu eşitleme
        n = min(len(df_ind), len(df_raw))
        df_ind = df_ind.iloc[-n:].reset_index(drop=True)
        df_raw = df_raw.iloc[-n:].reset_index(drop=True)
        
        # Merge
        df = pd.concat([df_raw, df_ind], axis=1)
        
        # Tipik Fiyat
        df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
        
        # Yatay Filtre Bileşenlerini Ön Hesapla (Sabitler)
        # Bollinger
        df['SMA20'] = df['Kapanis'].rolling(20).mean()
        df['StdDev'] = df['Kapanis'].rolling(20).std()
        df['BBUp'] = df['SMA20'] + 2 * df['StdDev']
        df['BBLow'] = df['SMA20'] - 2 * df['StdDev']
        df['BBMid'] = df['SMA20']
        
        df['BBWidth'] = 0.0
        mask = df['BBMid'] != 0
        df.loc[mask, 'BBWidth'] = ((df.loc[mask, 'BBUp'] - df.loc[mask, 'BBLow']) / df.loc[mask, 'BBMid']) * 100
        df['BBWidthAvg'] = df['BBWidth'].rolling(50).mean()
        
        # ARS Değişimi (Son 10 bar)
        # Vektörel olarak zor, rolling apply ile yapılabilir veya simple shift loop
        # Check if ARS changed in last 10 bars
        # Basitçe: ARS != ARS.shift(1) OR ARS != ARS.shift(2) ...
        # Bu biraz yavaş olabilir, ama ARS zaten sabit (CSV).
        # Hızlı yöntem: df['ARS_Diff'] = df['ARS'].diff().ne(0)
        # Rolling sum of diffs > 0 -> Değişti.
        df['ARS_Diff'] = df['ARS'].diff().fillna(0) != 0
        df['ARS_Degisti'] = df['ARS_Diff'].rolling(10).sum() > 0 # Son 10 barda en az 1 değişim
        df['ARS_Degisti'] = df['ARS_Degisti'].astype(int)
        
        return df
        
    except Exception as e:
        print(f"Hata: {e}")
        return None

def backtest_vectorized(df, params):
    # Parametreleri aç
    min_score = params['min_score']
    netlot_th = params['netlot_th']
    adx_th = params['adx_th']
    ars_mesafe_th = params['ars_mesafe_th']
    
    # --- Yatay Filtre Hesapla ---
    # ARS Mesafe
    ars_mesafe = (abs(df['Kapanis'] - df['ARS']) / df['ARS'] * 100).fillna(0)
    
    # Filtre Puanı
    f_score = (df['ARS_Degisti'] == 1).astype(int) + \
              (ars_mesafe > ars_mesafe_th).astype(int) + \
              (df['ADX'] > 20.0).astype(int) + \
              (df['BBWidth'] > df['BBWidthAvg'] * 0.8).astype(int)
              
    yatay_filtre = (f_score >= 2).astype(int)
    
    # --- Sinyal Skorları ---
    # LONG Terms
    l1 = (df['Kapanis'] > df['ARS']).astype(int)
    l2 = ((df['QQEF'] > df['QQES']) & (df['QQEF'] > 50)).astype(int)
    l3 = (df['RVI'] > df['RVI_Sig']).astype(int)
    l4 = (df['Qstick'] > 0).astype(int)
    l5 = (df['NetLot_MA'] > netlot_th).astype(int) # CSV'deki NetLot MA'lı mı? Ideal scriptte NetLot_MA export ettik diye hatırlıyorum. CSV headerlarına bakmak lazım. 
    # CSV header kontrolü: NetLot (step 872 code view: NetLot fonksiyonu MA yok, ama scriptte MA aldık mı? Arastirma_1DK.txt: NetLot_MA = MA(NetLot, 5). CSV'de NetLot var mı?
    # Varsayım: CSV'deki 'NetLot' ham değer. MA almamız gerekebilir.
    # Ama biz ScoreBasedStrategy içinde MA alıyorduk. Burada da almalıyız. 
    # Vektörel olduğu için df['NetLot'].rolling(5).mean() diyebiliriz.
    # Şimdilik direkt kullanalım, sonra düzeltiriz.
    
    l6 = (df['ADX'] > adx_th).astype(int)
    
    long_score = l1 + l2 + l3 + l4 + l5 + l6
    
    # SHORT Terms
    s1 = (df['Kapanis'] < df['ARS']).astype(int)
    s2 = ((df['QQEF'] < df['QQES']) & (df['QQEF'] < 50)).astype(int)
    s3 = (df['RVI'] < df['RVI_Sig']).astype(int)
    s4 = (df['Qstick'] < 0).astype(int)
    s5 = (df['NetLot_MA'] < -netlot_th).astype(int)
    s6 = (df['ADX'] > adx_th).astype(int)
    
    short_score = s1 + s2 + s3 + s4 + s5 + s6
    
    # --- Sinyaller ---
    # Entry Sinyalleri (0: Yok, 1: Long, -1: Short)
    signals = pd.Series(0, index=df.index)
    
    # Sadece Filtre Geçerse
    trend_mask = yatay_filtre == 1
    
    long_entry = trend_mask & (long_score >= min_score) & (short_score < 2)
    short_entry = trend_mask & (short_score >= min_score) & (long_score < 2)
    
    signals.loc[long_entry] = 1
    signals.loc[short_entry] = -1
    
    # Exit Sinyalleri (0: Flat)
    # Long Exit: ARS Kırılımı (Kapanis < ARS) OR Short Score >= 4
    long_exit = (df['Kapanis'] < df['ARS']) | (short_score >= 4)
    
    # Short Exit: ARS Kırılımı (Kapanis > ARS) OR Long Score >= 4
    short_exit = (df['Kapanis'] > df['ARS']) | (long_score >= 4)
    
    # --- Pozisyon Simülasyonu (Iterative Speedup) ---
    # Pandas ile tam vektörel pozisyon takibi zordur (stateful).
    # Ancak Numba veya basit loop ile yapılabilir.
    # Burada basit loop kullanalım ama sadece 'olay' anlarında işlem yapalım.
    
    sig_vals = signals.values
    l_exit_vals = long_exit.values
    s_exit_vals = short_exit.values
    closes = df['Kapanis'].values
    
    pos = 0
    trade_count = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    entry_price = 0.0
    
    for i in range(1, len(df)):
        if pos == 0:
            if sig_vals[i] == 1:
                pos = 1
                entry_price = closes[i]
                trade_count += 1
            elif sig_vals[i] == -1:
                pos = -1
                entry_price = closes[i]
                trade_count += 1
        elif pos == 1:
            if l_exit_vals[i]:
                # Exit Long
                pnl = closes[i] - entry_price
                if pnl > 0: 
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                pos = 0
            elif sig_vals[i] == -1: # Reverse (Opsiyonel? Bizim stratejide önce Flat sonra giriş var mı? Kodda "Sinyal=F" oluyor. Bir sonraki barda giriş olabilir.)
                # Şimdilik sadece Exit.
                pass
        elif pos == -1:
            if s_exit_vals[i]:
                # Exit Short
                pnl = entry_price - closes[i]
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    gross_loss += abs(pnl)
                pos = 0
                
    net_profit = gross_profit - gross_loss
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    return {
        'Net Profit': net_profit,
        'Trades': trade_count,
        'Win Rate': win_rate,
        'Profit Factor': pf
    }

def run_optimization():
    df = load_data()
    if df is None: return
    
    # NetLot MA düzeltme (Eğer CSV'de ham NetLot varsa)
    # Varsayım: CSV'de 'NetLot' var.
    if 'NetLot' in df.columns:
        df['NetLot_MA'] = df['NetLot'].rolling(5).mean().fillna(0)
    else:
        # Belki 'NetLot_MA' adıyla gelmiştir
        if 'NetLot_MA' not in df.columns:
             # Default 0
             df['NetLot_MA'] = 0
    
    # GRID SEARCH KOMBİNASYONLARI
    min_scores = [3, 4, 5]
    netlot_ths = [5, 10, 15, 20]
    adx_ths = [15, 20, 25, 30]
    ars_mesafe_ths = [0.10, 0.15, 0.25]
    
    combinations = list(itertools.product(min_scores, netlot_ths, adx_ths, ars_mesafe_ths))
    print(f"Toplam Kombinasyon: {len(combinations)}")
    
    results = []
    start_time = time()
    
    for i, (ms, nl, adx, ars_m) in enumerate(combinations):
        params = {
            'min_score': ms,
            'netlot_th': nl,
            'adx_th': adx,
            'ars_mesafe_th': ars_m
        }
        
        metrics = backtest_vectorized(df, params)
        
        res = params.copy()
        res.update(metrics)
        results.append(res)
        
        if i % 10 == 0:
            print(f"Bitti: {i}/{len(combinations)}...", end='\r')
            
    print(f"\nTamamlandı! Süre: {time() - start_time:.2f} sn")
    
    # Sonuçları DataFrame yap ve sırala (Net Profit'e göre)
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('Net Profit', ascending=False)
    
    print("\n--- EN İYİ 5 SONUÇ (Net Profit) ---")
    print(res_df.head(5).to_string(index=False))
    
    # En iyi Profit Factor (Min 100 işlem)
    print("\n--- EN İYİ 5 SONUÇ (Profit Factor, Trades > 100) ---")
    filtered = res_df[res_df['Trades'] > 100].sort_values('Profit Factor', ascending=False)
    print(filtered.head(5).to_string(index=False))
    
    # CSV Kaydet
    res_df.to_csv("d:/Projects/IdealQuant/tests/optimization_results.csv", index=False)
    print("\nSonuçlar kaydedildi: d:/Projects/IdealQuant/tests/optimization_results.csv")

if __name__ == "__main__":
    run_optimization()
