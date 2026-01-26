# -*- coding: utf-8 -*-
"""
Arastirma_1DK_Duzeltilmis Strateji Karşılaştırma
IdealData ideal_signals.csv ile Python sinyallerini karşılaştırır
"""

import sys
import io
import pandas as pd
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indicators.core import EMA, SMA, ADX, BollingerBands, QQEF, RVI, Qstick, NetLot

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_price_data():
    """OHLCV fiyat verisini yükle"""
    csv_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    return df


def load_ideal_trades():
    """IdealData işlem listesini yükle"""
    # ideal_signals_optimized.csv (4242 işlem, Python çıktısına (4234) çok yakın)
    csv_path = 'd:/Projects/IdealQuant/data/ideal_signals_optimized.csv'
    
    # Encoding ve ayırıcı dosya formatına göre
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path, sep=';', encoding='cp1254')
        
    df.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 
                  'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
    
    # Tarih formatı: 08.01.2025 21:02 (Saniyesiz olabilir)
    df['Time'] = pd.to_datetime(df['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
    df['Direction'] = df['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')
    
    return df[['Time', 'Direction', 'AcilisFyt', 'KarZarar']].dropna(subset=['Time'])


def calculate_ars(typical, ema_period=3, k=0.0123):
    """ARS hesapla (IdealData uyumlu)"""
    n = len(typical)
    ema = EMA(typical, ema_period)
    ars = [0.0] * n
    
    for i in range(1, n):
        alt_band = ema[i] * (1 - k)
        ust_band = ema[i] * (1 + k)
        
        if alt_band > ars[i-1]:
            ars[i] = alt_band
        elif ust_band < ars[i-1]:
            ars[i] = ust_band
        else:
            ars[i] = ars[i-1]
    
    return ars


def run_arastirma_strategy(opens, highs, lows, closes, typical):
    """
    Arastirma_1DK_Duzeltilmis stratejisi
    6 indikatör skor sistemi + Yatay Filtre
    """
    n = len(closes)
    
    # Parametreler (IdealData'dan)
    MIN_ONAY_SKORU = 5
    QQEF_PERIOD = 14
    QQEF_SMOOTH = 5
    RVI_PERIOD = 10
    QSTICK_PERIOD = 8
    NETLOT_ESIK = 20
    ADX_PERIOD = 14
    ADX_ESIK = 25
    ARS_K = 0.0123
    ARS_EMA = 3
    YATAY_ESIK = 10
    
    # İndikatörler
    ars = calculate_ars(typical, ARS_EMA, ARS_K)
    adx = ADX(highs, lows, closes, ADX_PERIOD)
    qqef, qqes = QQEF(closes, QQEF_PERIOD, QQEF_SMOOTH)
    rvi, rvi_sig = RVI(opens, highs, lows, closes, RVI_PERIOD)
    qstick = Qstick(opens, closes, QSTICK_PERIOD)
    netlot = NetLot(opens, highs, lows, closes)
    netlot_ma = SMA(netlot, 5)
    
    # Bollinger
    bb_up, bb_mid, bb_down = BollingerBands(closes, 20, 2.0)
    bb_width = [0.0] * n
    for i in range(n):
        if bb_mid[i] != 0:
            bb_width[i] = ((bb_up[i] - bb_down[i]) / bb_mid[i]) * 100
    bb_width_avg = SMA(bb_width, 50)
    
    # ARS Değişme Durumu
    ars_degisme = [0] * n
    for i in range(YATAY_ESIK, n):
        ars_sabit = True
        for j in range(1, YATAY_ESIK + 1):
            if ars[i] != ars[i-j]:
                ars_sabit = False
                break
        ars_degisme[i] = 0 if ars_sabit else 1
    
    # ARS Mesafe
    ars_mesafe = [0.0] * n
    for i in range(1, n):
        if ars[i] != 0:
            ars_mesafe[i] = abs(closes[i] - ars[i]) / ars[i] * 100
    
    # Yatay Filtre
    yatay_filtre = [0] * n
    for i in range(50, n):
        skor = 0
        if ars_degisme[i] == 1: skor += 1
        if ars_mesafe[i] > 0.25: skor += 1
        if adx[i] > 20.0: skor += 1
        if bb_width[i] > bb_width_avg[i] * 0.8: skor += 1
        yatay_filtre[i] = 1 if skor >= 2 else 0
    
    # Sinyal üret ve P&L hesapla
    signals = []
    trades = []
    son_yon = ""
    entry_price = 0.0
    entry_index = 0
    total_pnl = 0.0
    
    for i in range(50, n):
        sinyal = ""
        
        # Skorlama
        long_score = 0
        short_score = 0
        
        if closes[i] > ars[i]: long_score += 1
        elif closes[i] < ars[i]: short_score += 1
        
        if qqef[i] > qqes[i] and qqef[i] > 50: long_score += 1
        elif qqef[i] < qqes[i] and qqef[i] < 50: short_score += 1
        
        if rvi[i] > rvi_sig[i]: long_score += 1
        elif rvi[i] < rvi_sig[i]: short_score += 1
        
        if qstick[i] > 0: long_score += 1
        elif qstick[i] < 0: short_score += 1
        
        if netlot_ma[i] > NETLOT_ESIK: long_score += 1
        elif netlot_ma[i] < -NETLOT_ESIK: short_score += 1
        
        if adx[i] > ADX_ESIK:
            long_score += 1
            short_score += 1
        
        # Çıkış mantığı
        if son_yon == "A":
            if closes[i] < ars[i] or short_score >= 4:
                sinyal = "F"
        elif son_yon == "S":
            if closes[i] > ars[i] or long_score >= 4:
                sinyal = "F"
        
        # Giriş mantığı
        if sinyal == "" and son_yon != "A" and son_yon != "S":
            if yatay_filtre[i] == 1:
                if long_score >= MIN_ONAY_SKORU and short_score < 2:
                    sinyal = "A"
                elif short_score >= MIN_ONAY_SKORU and long_score < 2:
                    sinyal = "S"
        
        # Pozisyon güncelle
        if sinyal != "" and son_yon != sinyal:
            # İşlem kapatma (P&L ve Kayıt)
            if son_yon in ["A", "S"] and sinyal in ["F", "A", "S"]:
                pnl = 0.0
                if son_yon == "A":
                    pnl = closes[i] - entry_price
                else: # S
                    pnl = entry_price - closes[i]
                
                total_pnl += pnl
                trades.append({
                    'EntryIndex': entry_index,
                    'ExitIndex': i,
                    'Direction': 'LONG' if son_yon == 'A' else 'SHORT',
                    'EntryPrice': entry_price,
                    'ExitPrice': closes[i],
                    'PnL': pnl
                })
            
            # Yeni giriş
            if sinyal in ["A", "S"]:
                entry_price = closes[i]
                entry_index = i
                signals.append({
                    'BarIndex': i,
                    'Direction': 'LONG' if sinyal == 'A' else 'SHORT',
                    'Price': closes[i]
                })
            
            son_yon = sinyal if sinyal != "F" else ""
            
    return signals, trades, total_pnl


def main():
    print("=" * 70)
    print("Arastirma_1DK Strateji - IdealData Karşılaştırması")
    print("=" * 70)
    
    # Veri yükle
    print("\n[1] Veriler yükleniyor...")
    price_df = load_price_data()
    ideal_trades = load_ideal_trades()
    
    print(f"    Fiyat verisi: {len(price_df)} bar")
    print(f"    IdealData işlem: {len(ideal_trades)}")
    
    # Veri hazırla
    opens = price_df['Acilis'].values.astype(float).tolist()
    highs = price_df['Yuksek'].values.astype(float).tolist()
    lows = price_df['Dusuk'].values.astype(float).tolist()
    closes = price_df['Kapanis'].values.astype(float).tolist()
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    
    # Strateji çalıştır
    print("\n[2] Strateji hesaplanıyor...")
    python_signals, python_trades, python_pnl = run_arastirma_strategy(opens, highs, lows, closes, typical)
    print(f"    Python sinyal sayısı: {len(python_signals)}")
    print(f"    Python Toplam P&L : {python_pnl:.2f} Puan")
    
    # DataFrame oluştur
    py_df = pd.DataFrame(python_signals)
    py_df['Time'] = price_df['DateTime'].iloc[py_df['BarIndex'].values].values
    py_df['TimeDakika'] = pd.to_datetime(py_df['Time']).dt.floor('min')
    
    ideal_trades['TimeDakika'] = ideal_trades['Time'].dt.floor('min')
    
    # Karşılaştırma
    print("\n[3] Sinyal Karşılaştırması...")
    
    start = max(ideal_trades['Time'].min(), py_df['Time'].min())
    end = min(ideal_trades['Time'].max(), py_df['Time'].max())
    
    ideal_subset = ideal_trades[(ideal_trades['Time'] >= start) & (ideal_trades['Time'] <= end)]
    py_subset = py_df[(py_df['Time'] >= start) & (py_df['Time'] <= end)]
    
    print(f"    Aralık: {start} - {end}")
    print(f"    Ideal işlem: {len(ideal_subset)}")
    print(f"    Python sinyal: {len(py_subset)}")
    
    # Eşleşme
    matches = 0
    mismatches = 0
    missing = 0
    
    for _, row in ideal_subset.iterrows():
        t = row['TimeDakika']
        d = row['Direction']
        
        match = py_subset[py_subset['TimeDakika'] == t]
        
        if len(match) > 0:
            if match.iloc[0]['Direction'] == d:
                matches += 1
            else:
                mismatches += 1
        else:
            missing += 1
    
    total = len(ideal_subset)
    accuracy = (matches / total * 100) if total > 0 else 0
    
    print("-" * 50)
    print(f"Tam Eşleşme    : {matches}")
    print(f"Yön Hatası     : {mismatches}")
    print(f"Eksik          : {missing}")
    print(f"Uyumluluk      : %{accuracy:.2f}")
    
    if accuracy >= 80:
        print("✅ Yüksek uyum!")
    else:
        print("❌ Düşük uyum!")
        
    print("-" * 50)
    
    # P&L Analizi
    print("\n[4] P&L Analizi...")
    
    # IdealData P&L toplamını dosyadaki "KarZarar" sütunundan alıyoruz (string olabilir)
    # Format: "1.234,56" -> 1234.56
    try:
        ideal_pnl_raw = ideal_subset['KarZarar'].astype(str)
        ideal_pnl_clean = ideal_pnl_raw.str.replace('.', '').str.replace(',', '.')
        ideal_total_pnl = pd.to_numeric(ideal_pnl_clean, errors='coerce').sum()
    except Exception as e:
        print(f"P&L parse hatası: {e}")
        ideal_total_pnl = 0
        
    print(f"IdealData P&L (Dosya) : {ideal_total_pnl:.2f}")
    print(f"Python P&L (Hesaplanan): {python_pnl:.2f}")
    
    diff_pnl = python_pnl - ideal_total_pnl
    diff_pct = (diff_pnl / ideal_total_pnl * 100) if ideal_total_pnl != 0 else 0
    
    print(f"Fark: {diff_pnl:.2f} ({diff_pct:.2f}%)")
    
    if abs(diff_pct) < 5.0:
        print("✅ P&L Uyumu Başarılı!")
    else:
        print("⚠️ P&L Farkı Var! Komisyon/Slippage veya çıkış zamanlaması farkından olabilir.")


if __name__ == "__main__":
    main()
