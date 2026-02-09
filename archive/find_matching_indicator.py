# -*- coding: utf-8 -*-
"""
Reverse Engineering: Find Matching Indicator Formula
IdealData ile %100 uyuşan formülü bulmak için varyasyonları dener.
"""

import sys
import io
import pandas as pd
import numpy as np

# Konsol encoding ayarı
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- TEMEL FONKSİYONLAR ---
def SMA(data, period):
    return pd.Series(data).rolling(window=period).mean().fillna(0).tolist()

def EMA(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().fillna(0).tolist()

def WMA(data, period):
    weights = np.arange(1, period + 1)
    return pd.Series(data).rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).fillna(0).tolist()

def RMA(data, period):
    """Wilder's Smoothing"""
    n = len(data)
    result = [0.0] * n
    if n < period: return result
    
    result[period-1] = sum(data[:period]) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period - 1) + data[i]) / period
    return result

def RSI_Wilder(closes, period=14):
    """Wilder's RSI (using RMA)"""
    n = len(closes)
    result = [50.0] * n
    if n <= period: return result
    
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    
    # RMA hesapla (manuel loop ile daha kolay kontrol edilir)
    avg_gain = [0.0] * n
    avg_loss = [0.0] * n
    
    # İlk değer SMA
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period + 1, n):
        idx = i - 1 # gains indexi 0'dan başlar ama n-1 boyundadır
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[idx]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[idx]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            result[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100 - (100 / (1 + rs))
            
    return result

def load_data():
    try:
        # IdealData Results
        df_ideal = pd.read_csv("d:/Projects/IdealQuant/data/ideal_score_indicators.csv", sep=';', decimal=',', encoding='utf-8') 
        df_ideal.columns = [c.strip() for c in df_ideal.columns]
        
        for col in df_ideal.columns:
            if col not in ['Date', 'Time', 'Tarih', 'Saat']:
                df_ideal[col] = pd.to_numeric(df_ideal[col], errors='coerce')

        # Raw Data
        df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', header=None)
        if isinstance(df_raw.iloc[0,0], str) and "Tarih" in df_raw.iloc[0,0]:
            df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254')
            df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        else:
            df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

        highs = df_raw['Yuksek'].values.astype(float).tolist()
        lows = df_raw['Dusuk'].values.astype(float).tolist()
        closes = df_raw['Kapanis'].values.astype(float).tolist()
        
        return df_ideal, highs, lows, closes
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None, None, None, None

def check_match(name, calculated, ideal, tolerance=0.0001):
    # İLK 500 BARI ATLA (Initial value bias için)
    calc_slice = np.array(calculated[500:])
    ideal_slice = ideal[500:len(calculated)]
    
    mask = ~np.isnan(calc_slice) & ~np.isnan(ideal_slice)
    if np.sum(mask) == 0: return False, 9999
    
    diff = np.abs(calc_slice[mask] - ideal_slice[mask])
    max_diff = np.max(diff)
    
    return (max_diff < tolerance), max_diff

# --- VARYASYONLAR ---
def adx_variant_rma(highs, lows, closes, period):
    n = len(closes)
    plus_dm, minus_dm, tr = [0.0]*n, [0.0]*n, [0.0]*n
    
    for i in range(1, n):
        hd = highs[i] - highs[i-1]
        ld = lows[i-1] - lows[i]
        if hd > ld and hd > 0: plus_dm[i] = hd
        if ld > hd and ld > 0: minus_dm[i] = ld
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        
    s_tr = RMA(tr, period)
    s_plus = RMA(plus_dm, period)
    s_minus = RMA(minus_dm, period)
    
    dx = [0.0]*n
    for i in range(period, n):
        if s_tr[i] != 0:
            p = (s_plus[i]/s_tr[i])*100
            m = (s_minus[i]/s_tr[i])*100
            if p+m!=0: dx[i] = abs(p-m)/(p+m)*100
            
    # ADX = RMA(DX)
    return RMA(dx, period)

def qqef_variant_ema(closes, rsi_period, smooth_period):
    rsi = RSI_Wilder(closes, rsi_period)
    return EMA(rsi, smooth_period) # QQEF = EMA(RSI)

def qqes_variant_ema(qqef_data, p): return EMA(qqef_data, p)
def qqes_variant_rma(qqef_data, p): return RMA(qqef_data, p)
def qqes_variant_sma(qqef_data, p): return SMA(qqef_data, p)
def qqes_variant_wma(qqef_data, p): return WMA(qqef_data, p)

def run_search():
    print("Veri yükleniyor...")
    df_ideal, highs, lows, closes = load_data()
    if df_ideal is None: return

    print("\n--- ADX Analizi (500 Bar Warmup) ---")
    ideal_adx = df_ideal['ADX'].values
    res = adx_variant_rma(highs, lows, closes, 14)
    match, diff = check_match("ADX(RMA)", res, ideal_adx)
    print(f"  ADX(RMA): MaxDiff={diff:.6f} {'OK' if match else 'FAIL'}")

    print("\n--- QQES Geniş Kapsamlı Arama ---")
    ideal_qqes = df_ideal['QQES'].values
    
    # QQEF (Fast Line) - Doğrulanmış Veri
    # QQEF = EMA(RSI_Wilder, 5)
    rsi = RSI_Wilder(closes, 14)
    qqef = EMA(rsi, 5)
    
    ma_types = [
        ("EMA", lambda d, p: EMA(d, p)),
        ("SMA", lambda d, p: SMA(d, p)),
        ("RMA", lambda d, p: RMA(d, p)),
        ("WMA", lambda d, p: WMA(d, p)),
    ]
    
    best_diff = 9999.0
    best_name = ""
    
    print(f"  {'Type':<10} | {'Period':<6} | {'MaxDiff':<15}")
    print("-" * 40)
    
    # 1'den 30'a kadar periyotları tara
    for period in range(1, 31):
        for type_name, func in ma_types:
            try:
                # 1. Varyasyon: Signal = MA(QQEF, period)
                res = func(qqef, period)
                match, diff = check_match(f"{type_name}({period})", res, ideal_qqes, tolerance=100) # Tolerance önemsiz, min diff arıyoruz
                
                if diff < best_diff:
                    best_diff = diff
                    best_name = f"{type_name}(QQEF, {period})"
                
                if diff < 1.0: # Umut verici olanları yaz
                    print(f"  {type_name:<10} | {period:<6} | {diff:.6f}")
                    
                # 2. Varyasyon: Signal = MA(RSI, period) (Belki direkt RSI'dan türüyordur?)
                res2 = func(rsi, period)
                match2, diff2 = check_match(f"{type_name}(RSI, {period})", res2, ideal_qqes, tolerance=100)
                
                if diff2 < best_diff:
                    best_diff = diff2
                    best_name = f"{type_name}(RSI, {period})"

                    
            except Exception:
                continue
                
    print("-" * 40)
    print(f"EN İYİ EŞLEŞME: {best_name} -> MaxDiff: {best_diff:.6f}")
    
    # Eğer 2.7 civarındaysa, belki QQEF periyodu ile ilgilidir (14, 5).
    # 4.236 fibo sayısı?
    pass

if __name__ == "__main__":
    run_search()
