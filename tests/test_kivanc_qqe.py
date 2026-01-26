# -*- coding: utf-8 -*-
"""
Test KivancOzbilgic QQE Formula
"""
import sys
import io
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def EMA_calc(data, period):
    return pd.Series(data).ewm(span=period, adjust=False).mean().fillna(0).tolist()

def RSI_Wilder(closes, period=14):
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

def KivancQQE(closes, rsi_period=14, smooth_period=5):
    # 1. RSI
    rsi = RSI_Wilder(closes, rsi_period)
    
    # 2. RSII (Smoothed RSI) = QQEF
    # IdealData QQEF = EMA(RSI, 5) ile uyumluydu.
    rsii = EMA_calc(rsi, smooth_period)
    qqef = rsii
    
    # 3. TR Calculation
    n = len(closes)
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = abs(rsii[i] - rsii[i-1])
        
    # 4. Wilder's Smoothing of TR -> WWMA
    # wwalpha = 1 / length (rsi_period)
    wwalpha = 1.0 / rsi_period
    wwma = [0.0] * n
    
    # Initial value logic? Use simple assignment or 0
    # Pine nz(WWMA[1]) is 0 for first bar
    wwma[0] = tr[0] # or 0
    for i in range(1, n):
        wwma[i] = wwalpha * tr[i] + (1 - wwalpha) * wwma[i-1]
        
    # 5. ATRRSI (Double Smoothed)
    atrrsi = [0.0] * n
    atrrsi[0] = wwma[0]
    for i in range(1, n):
        atrrsi[i] = wwalpha * wwma[i] + (1 - wwalpha) * atrrsi[i-1]
        
    # 6. QUP and QDN
    # Multiplier 4.236
    mult = 4.236
    qqes = [0.0] * n
    
    # Initial value for QQES logic
    qqes[0] = rsii[0] 
    
    # PineScript Logic:
    # QUP=QQEF+ATRRSI*4.236
    # QDN=QQEF-ATRRSI*4.236
    # QQES=0.0
    # QQES:=QUP<nz(QQES[1]) ? QUP : QQEF>nz(QQES[1]) and QQEF[1]<nz(QQES[1]) ? QDN :  QDN>nz(QQES[1]) ? QDN : QQEF<nz(QQES[1]) and QQEF[1]>nz(QQES[1]) ? QUP : nz(QQES[1])
    
    for i in range(1, n):
        qup = qqef[i] + atrrsi[i] * mult
        qdn = qqef[i] - atrrsi[i] * mult
        
        prev_qqes = qqes[i-1]
        prev_qqef = qqef[i-1]
        curr_qqef = qqef[i]
        
        # Logic 1: QUP < PrevQQES -> QUP
        if qup < prev_qqes:
            qqes[i] = qup
        # Logic 2: QQEF > PrevQQES and PrevQQEF < PrevQQES -> QDN (CrossUp -> Switch to QDN?)
        # Pine: QQEF>nz(QQES[1]) and QQEF[1]<nz(QQES[1]) ? QDN
        elif curr_qqef > prev_qqes and prev_qqef < prev_qqes:
            qqes[i] = qdn
        # Logic 3: QDN > PrevQQES -> QDN
        elif qdn > prev_qqes:
            qqes[i] = qdn
        # Logic 4: QQEF < PrevQQES and PrevQQEF > PrevQQES -> QUP (CrossDown -> Switch to QUP?)
        # Pine: QQEF<nz(QQES[1]) and QQEF[1]>nz(QQES[1]) ? QUP
        elif curr_qqef < prev_qqes and prev_qqef > prev_qqes:
            qqes[i] = qup
        # Else: keep same
        else:
            qqes[i] = prev_qqes
            
    return qqef, qqes

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

def check_match(name, calculated, ideal, tolerance=0.1):
    # 500 Bar Warmup
    calc_slice = np.array(calculated[500:])
    ideal_slice = ideal[500:len(calculated)]
    mask = ~np.isnan(calc_slice) & ~np.isnan(ideal_slice)
    if np.sum(mask) == 0: return False, 9999
    diff = np.abs(calc_slice[mask] - ideal_slice[mask])
    max_diff = np.max(diff)
    return (max_diff < tolerance), max_diff

def run_test():
    print("Veri yükleniyor...")
    df_ideal, closes = load_data()
    if df_ideal is None: return
    
    ideal_qqes = df_ideal['QQES'].values
    
    print("\n--- KivancQQE Testi ---")
    
    qqef, qqes = KivancQQE(closes, 14, 5)
    
    match, diff = check_match("KivancQQE", qqes, ideal_qqes)
    print(f"  Max Diff: {diff:.6f} {'✅' if match else '❌'}")
    
    if match:
        print("  SÜPER! Formül Bulundu!")
    else:
        print("  Hala fark var. Trailing logic varyasyonları gerekebilir.")

if __name__ == "__main__":
    run_test()
