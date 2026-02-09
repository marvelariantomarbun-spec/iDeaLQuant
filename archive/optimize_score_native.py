# -*- coding: utf-8 -*-
import sys
import os
import io
import pandas as pd
import numpy as np
import itertools
from time import time

# Proje kök dizini
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indicators.core import EMA, ATR, ADX, SMA, ARS, RVI, Qstick, NetLot

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def load_price_data():
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None)
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        cols = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
        return df
    except Exception as e:
        print(f"Hata: {e}")
        return None

def calculate_indicators_vectorized(df, params):
    # Parametreler
    rvi_p = params['rvi_p']
    adx_p = params['adx_p']
    ars_k = params['ars_k']
    ars_ema_p = params['ars_ema_p']
    qstick_p = params['qstick_p']
    
    opens = df['Acilis'].values.tolist()
    highs = df['Yuksek'].values.tolist()
    lows = df['Dusuk'].values.tolist()
    closes = df['Kapanis'].values.tolist()
    typical = df['Tipik'].values.tolist()
    
    # 1. ARS (Değişken EMA Period)
    ars = ARS(typical, ars_ema_p, ars_k)
    df['ARS'] = ars
    
    # 2. RVI
    rvi, rvi_sig = RVI(opens, highs, lows, closes, rvi_p)
    df['RVI'] = rvi
    df['RVI_Sig'] = rvi_sig
    
    # 3. Qstick
    qstick = Qstick(opens, closes, qstick_p)
    df['Qstick'] = qstick
    
    # 4. NetLot (Sabit)
    netlot = NetLot(opens, highs, lows, closes)
    df['NetLot'] = netlot
    df['NetLot_MA'] = pd.Series(netlot).rolling(5).mean().fillna(0)
    
    # 5. ADX
    adx = ADX(highs, lows, closes, adx_p)
    df['ADX'] = adx
    
    # Yatay Filtre (Sabit 20/2 BB varsayıyoruz - yoksa o da mı optimize edilsin? Şimdilik sabit)
    sma20 = df['Kapanis'].rolling(20).mean()
    std20 = df['Kapanis'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BBWidth'] = ((upper - lower) / sma20 * 100).fillna(0)
    df['BBWidthAvg'] = df['BBWidth'].rolling(50).mean()
    
    df['ARS_Diff'] = df['ARS'].diff().fillna(0) != 0
    df['ARS_Degisti'] = df['ARS_Diff'].rolling(10).sum().gt(0).astype(int)
    
    return df

def backtest_vectorized(df, thresholds):
    min_score = thresholds['min_score']
    netlot_th = thresholds['netlot_th']
    adx_th = thresholds['adx_th']
    ars_mesafe_th = thresholds['ars_mesafe_th']
    
    ars_mesafe = (abs(df['Kapanis'] - df['ARS']) / df['ARS'] * 100).fillna(0)
    
    f_score = (df['ARS_Degisti'] == 1).astype(int) + \
              (ars_mesafe > ars_mesafe_th).astype(int) + \
              (df['ADX'] > 20.0).astype(int) + \
              (df['BBWidth'] > df['BBWidthAvg'] * 0.8).astype(int)
              
    yatay_filtre = (f_score >= 2).astype(int)
    
    # Skorlama
    l1 = (df['Kapanis'] > df['ARS']).astype(int)
    l2 = (df['RVI'] > df['RVI_Sig']).astype(int)
    l3 = (df['Qstick'] > 0).astype(int)
    l4 = (df['NetLot_MA'] > netlot_th).astype(int)
    l5 = (df['ADX'] > adx_th).astype(int)
    long_score = l1 + l2 + l3 + l4 + l5
    
    s1 = (df['Kapanis'] < df['ARS']).astype(int)
    s2 = (df['RVI'] < df['RVI_Sig']).astype(int)
    s3 = (df['Qstick'] < 0).astype(int)
    s4 = (df['NetLot_MA'] < -netlot_th).astype(int)
    s5 = (df['ADX'] > adx_th).astype(int)
    short_score = s1 + s2 + s3 + s4 + s5
    
    signals = pd.Series(0, index=df.index)
    
    # Filtreyi uygula
    valid_mask = yatay_filtre == 1
    
    long_entry = valid_mask & (long_score >= min_score) & (short_score < 2)
    short_entry = valid_mask & (short_score >= min_score) & (long_score < 2)
    
    signals.loc[long_entry] = 1
    signals.loc[short_entry] = -1
    
    long_exit = (df['Kapanis'] < df['ARS']) | (short_score >= 4)
    short_exit = (df['Kapanis'] > df['ARS']) | (long_score >= 4)
    
    # Loop
    sig_vals = signals.values
    l_exit_vals = long_exit.values
    s_exit_vals = short_exit.values
    closes = df['Kapanis'].values
    
    pos = 0
    trade_count = 0
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
                pnl = closes[i] - entry_price
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0
        elif pos == -1:
            if s_exit_vals[i]:
                pnl = entry_price - closes[i]
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0
                
    net_profit = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
    
    return net_profit, trade_count, pf

def run_native_optimization():
    df_raw = load_price_data()
    if df_raw is None: return
    print(f"Veri Yüklendi: {len(df_raw)} Bar")
    
    # --- FULL GRID ---
    # İndikatör Parametreleri
    ars_ks = [0.01, 0.015, 0.02] 
    ars_emas = [3, 5, 8] # ARS Period
    rvi_ps = [10, 14, 21]
    qstick_ps = [5, 8, 13]
    adx_ps = [14, 21] # ADX 14 genelde standarttır, 21 deneyelim
    
    # Eşik Parametreleri (İkinci Loop)
    min_scores = [3, 4] # Max score 5, min 3 mantıklı
    netlots = [10, 20]
    
    ind_combs = list(itertools.product(ars_ks, ars_emas, rvi_ps, qstick_ps, adx_ps))
    print(f"İndikatör Kombinasyon Sayısı: {len(ind_combs)}")
    print(f"Toplam Test (Eşiklerle): {len(ind_combs) * len(min_scores) * len(netlots)}")
    
    best_overall = {'Net Profit': -99999}
    results = []
    
    start_time = time()
    counter = 0
    
    for (k, ars_p, rvi_p, qs_p, adx_p) in ind_combs:
        p_params = {
            'rvi_p': rvi_p, 
            'adx_p': adx_p, 
            'ars_k': k,
            'ars_ema_p': ars_p,
            'qstick_p': qs_p
        }
        
        # 1. İndikatörleri Hesapla (Bu kombinasyon için)
        df_ind = calculate_indicators_vectorized(df_raw.copy(), p_params)
        
        # 2. Eşik Döngüsü
        for ms in min_scores:
            for nl in netlots:
                thresholds = {
                    'min_score': ms, 
                    'netlot_th': nl, 
                    'adx_th': 25, 
                    'ars_mesafe_th': 0.25
                }
                
                np_val, tr, pf_val = backtest_vectorized(df_ind, thresholds)
                
                res = {
                    'ARS_K': k,
                    'ARS_P': ars_p,
                    'RVI_P': rvi_p,
                    'QSt_P': qs_p,
                    'ADX_P': adx_p,
                    'MinScore': ms,
                    'NetLot': nl,
                    'Net Profit': np_val,
                    'Trades': tr,
                    'PF': pf_val
                }
                
                if np_val > 1000: # Dosyayı şişirmemek için sadece pozitifleri ekle
                    results.append(res)
                    
                if np_val > best_overall['Net Profit']: 
                    best_overall = res
                    
        counter += 1
        if counter % 5 == 0:
            print(f"Kombinasyon: {counter}/{len(ind_combs)} (Best: {best_overall['Net Profit']:.0f})", end='\r')

    print(f"\n\nTamamlandı! Süre: {time() - start_time:.2f} sn")
    
    if len(results) > 0:
        res_df = pd.DataFrame(results).sort_values('Net Profit', ascending=False)
        print("\n--- EN İYİ 10 SONUÇ (Full Grid) ---")
        print(res_df.head(10).to_string(index=False))
        res_df.to_csv("d:/Projects/IdealQuant/tests/native_optimization_full.csv", index=False)
    else:
        print("Kârlı sonuç bulunamadı.")

if __name__ == "__main__":
    run_native_optimization()
