# -*- coding: utf-8 -*-
"""
Deep Search: QQES Matching with Advanced Moving Averages
DEMA, TEMA, TMA, etc.
"""

import sys
import io
import pandas as pd
import numpy as np

# Konsol encoding ayarı
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- MA FUNCTIONS ---
def SMA(data, period):
    return pd.Series(data).rolling(window=period).mean().fillna(0).tolist()

def EMA(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().fillna(0).tolist()

def WMA(data, period):
    weights = np.arange(1, period + 1)
    return pd.Series(data).rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).fillna(0).tolist()

def RMA(data, period):
    n = len(data)
    result = [0.0] * n
    if n < period: return result
    result[period-1] = sum(data[:period]) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period - 1) + data[i]) / period
    return result

def DEMA(data, period):
    # Double Exponential Moving Average: 2*EMA - EMA(EMA)
    data_series = pd.Series(data)
    e1 = data_series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    return (2 * e1 - e2).fillna(0).tolist()

def TEMA(data, period):
    # Triple Exponential Moving Average: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    data_series = pd.Series(data)
    e1 = data_series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return (3 * e1 - 3 * e2 + e3).fillna(0).tolist()

def TMA(data, period):
    # Triangular Moving Average: SMA(SMA)
    p1 = int((period + 1) / 2)
    s1 = pd.Series(data).rolling(window=p1).mean()
    return s1.rolling(window=p1).mean().fillna(0).tolist()

def RSI_Wilder(closes, period=14):
    """Wilder's RSI (using RMA)"""
    n = len(closes)
    result = [50.0] * n
    if n <= period: return result
    deltas = np.diff(closes)
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = [0.0] * n
    avg_loss = [0.0] * n
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        idx = i - 1 
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[idx]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[idx]) / period
    for i in range(period, n):
        if avg_loss[i] == 0: result[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100 - (100 / (1 + rs))
    return result

def load_data():
    try:
        df_ideal = pd.read_csv("d:/Projects/IdealQuant/data/ideal_score_indicators.csv", sep=';', decimal=',', encoding='utf-8') 
        df_ideal.columns = [c.strip() for c in df_ideal.columns]
        for col in df_ideal.columns:
            if col not in ['Date', 'Time', 'Tarih', 'Saat']:
                df_ideal[col] = pd.to_numeric(df_ideal[col], errors='coerce')

        df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', header=None)
        if isinstance(df_raw.iloc[0,0], str) and "Tarih" in df_raw.iloc[0,0]:
            df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254')
            df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        else:
            df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

        closes = df_raw['Kapanis'].values.astype(float).tolist()
        return df_ideal, closes
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None, None

def check_match(name, calculated, ideal, tolerance=0.0001):
    calc_slice = np.array(calculated[500:])
    ideal_slice = ideal[500:len(calculated)]
    mask = ~np.isnan(calc_slice) & ~np.isnan(ideal_slice)
    if np.sum(mask) == 0: return False, 9999
    diff = np.abs(calc_slice[mask] - ideal_slice[mask])
    max_diff = np.max(diff)
    return (max_diff < tolerance), max_diff

def run_search():
    print("Veri yükleniyor...")
    df_ideal, closes = load_data()
    if df_ideal is None: return

    print("\n--- QQES Advanced Search ---")
    ideal_qqes = df_ideal['QQES'].values
    
    # QQEF (Fast Line) - Doğrulanmış
    rsi = RSI_Wilder(closes, 14)
    qqef = EMA(rsi, 5) # QQEF = EMA(RSI, 5)
    
    ma_types = [
        ("EMA", lambda d, p: EMA(d, p)),
        ("SMA", lambda d, p: SMA(d, p)),
        ("RMA", lambda d, p: RMA(d, p)),
        ("WMA", lambda d, p: WMA(d, p)),
        ("DEMA", lambda d, p: DEMA(d, p)),
        ("TEMA", lambda d, p: TEMA(d, p)),
        ("TMA",  lambda d, p: TMA(d, p))
    ]
    
    best_diff = 9999.0
    best_name = ""
    
    print(f"  {'Type':<10} | {'Period':<6} | {'MaxDiff':<15}")
    print("-" * 40)
    
    # 1-30 Periyot tara
    for period in range(1, 31):
        for type_name, func in ma_types:
            try:
                # 1. Varyasyon: Signal = MA(QQEF, period)
                res = func(qqef, period)
                match, diff = check_match(f"{type_name}({period})", res, ideal_qqes, tolerance=100)
                
                if diff < best_diff:
                    best_diff = diff
                    best_name = f"{type_name}(QQEF, {period})"
                
                if diff < 1.0:
                    print(f"  {type_name:<10} | {period:<6} | {diff:.6f}")
                    
                # 2. Varyasyon: Signal = MA(RSI, period)
                res2 = func(rsi, period)
                match2, diff2 = check_match(f"{type_name}(RSI, {period})", res2, ideal_qqes, tolerance=100)
                
                if diff2 < best_diff:
                    best_diff = diff2
                    best_name = f"{type_name}(RSI, {period})"
            except: continue
                
    print("-" * 40)
    print(f"EN İYİ EŞLEŞME: {best_name} -> MaxDiff: {best_diff:.6f}")

if __name__ == "__main__":
    run_search()
