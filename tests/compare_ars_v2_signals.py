# -*- coding: utf-8 -*-
"""
ARS Trend v2 - IdealData vs Python Sinyal Karşılaştırması
"""

import sys
import os
sys.path.insert(0, 'd:/Projects/IdealQuant/src')
sys.path.insert(0, 'd:/Projects/IdealQuant')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =====================================================
# 1. IdealData Sinyallerini Yükle
# =====================================================
def load_ideal_signals():
    csv_path = 'd:/Projects/IdealQuant/data/ideal_signals_2_Nolu_Strateji_200000Bar.csv'
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    
    # Kolon isimlerini düzelt
    df.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 
                  'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
    
    # Tarih ve fiyat parse
    df['Time'] = pd.to_datetime(df['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
    df['Direction'] = df['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')
    
    # Fiyat parse (virgül -> nokta)
    df['Price'] = df['AcilisFyt'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    return df[['No', 'Time', 'Direction', 'Price']].dropna(subset=['Time'])


# =====================================================
# 2. Python Stratejisini Çalıştır
# =====================================================
def run_python_strategy():
    from src.strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2, Signal
    
    # Fiyat verisini yükle
    csv_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', low_memory=False)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    
    # Verileri hazırla
    opens = df['Acilis'].values.tolist()
    highs = df['Yuksek'].values.tolist()
    lows = df['Dusuk'].values.tolist()
    closes = df['Kapanis'].values.tolist()
    typical = ((df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3).tolist()
    times = df['DateTime'].tolist()
    volumes = df['Lot'].values.tolist()
    
    # 1DK Konfigürasyonu (IdealData ile aynı)
    config = StrategyConfigV2(
        ars_ema_period=3, ars_atr_period=10, ars_atr_mult=0.5,
        ars_min_band=0.002, ars_max_band=0.015,
        momentum_period=5, breakout_period=10,
        mfi_period=14, mfi_hhv_period=14, mfi_llv_period=14,
        volume_hhv_period=14, volume_llv_period=14,
        kar_al_pct=3.0, iz_stop_pct=1.5, vade_tipi='ENDEKS'
    )
    
    strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, volumes, config)
    
    # Sinyalleri topla
    signals = []
    current_pos = 'FLAT'
    entry_price = 0.0
    extreme_price = 0.0
    
    for i in range(len(closes)):
        if current_pos == 'LONG':
            extreme_price = max(extreme_price, highs[i])
        elif current_pos == 'SHORT':
            extreme_price = min(extreme_price if extreme_price > 0 else highs[i], lows[i])
        
        sig = strategy.get_signal(i, current_pos, entry_price, extreme_price)
        
        if sig != Signal.NONE:
            if sig == Signal.LONG:
                signals.append({'time': times[i], 'direction': 'LONG', 'price': closes[i]})
                current_pos = 'LONG'
                entry_price = closes[i]
                extreme_price = highs[i]
            elif sig == Signal.SHORT:
                signals.append({'time': times[i], 'direction': 'SHORT', 'price': closes[i]})
                current_pos = 'SHORT'
                entry_price = closes[i]
                extreme_price = lows[i]
            elif sig == Signal.FLAT:
                current_pos = 'FLAT'
                entry_price = 0.0
                extreme_price = 0.0
    
    return pd.DataFrame(signals)


# =====================================================
# 3. Karşılaştırma
# =====================================================
def compare_signals(ideal_df, python_df, tolerance_minutes=1):
    """
    İki sinyal listesini karşılaştır
    tolerance_minutes: Zaman toleransı (dakika)
    """
    matches = 0
    mismatches = []
    
    # Sadece giriş sinyallerini karşılaştır (LONG/SHORT)
    ideal_entries = ideal_df[ideal_df['Direction'].isin(['LONG', 'SHORT'])].copy()
    python_entries = python_df[python_df['direction'].isin(['LONG', 'SHORT'])].copy()
    
    print(f"IdealData giriş sinyali: {len(ideal_entries)}")
    print(f"Python giriş sinyali: {len(python_entries)}")
    
    # Her IdealData sinyali için Python'da eşleşme ara
    for idx, ideal_row in ideal_entries.iterrows():
        ideal_time = ideal_row['Time']
        ideal_dir = ideal_row['Direction']
        
        # Tolerans içinde Python sinyali ara
        time_mask = (python_entries['time'] >= ideal_time - timedelta(minutes=tolerance_minutes)) & \
                    (python_entries['time'] <= ideal_time + timedelta(minutes=tolerance_minutes))
        
        matching = python_entries[time_mask & (python_entries['direction'] == ideal_dir)]
        
        if len(matching) > 0:
            matches += 1
        else:
            mismatches.append({
                'ideal_time': ideal_time,
                'ideal_dir': ideal_dir,
                'ideal_price': ideal_row['Price']
            })
    
    return matches, mismatches


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ARS TREND v2 - SINYAL KARŞILAŞTIRMASI")
    print("=" * 60)
    
    # 1. IdealData sinyallerini yükle
    print("\n[1] IdealData sinyalleri yükleniyor...")
    ideal_df = load_ideal_signals()
    print(f"    Toplam: {len(ideal_df)} işlem")
    
    # 2. Python stratejisini çalıştır
    print("\n[2] Python stratejisi çalıştırılıyor...")
    python_df = run_python_strategy()
    print(f"    Toplam: {len(python_df)} giriş sinyali")
    
    # 3. Karşılaştır
    print("\n[3] Sinyaller karşılaştırılıyor...")
    matches, mismatches = compare_signals(ideal_df, python_df, tolerance_minutes=1)
    
    total_ideal = len(ideal_df[ideal_df['Direction'].isin(['LONG', 'SHORT'])])
    match_pct = (matches / total_ideal * 100) if total_ideal > 0 else 0
    
    print("\n" + "=" * 60)
    print("SONUÇ")
    print("=" * 60)
    print(f"IdealData işlem sayısı : {total_ideal}")
    print(f"Python giriş sinyali   : {len(python_df)}")
    print(f"Eşleşen                : {matches}")
    print(f"Eşleşmeyen             : {len(mismatches)}")
    print(f"Eşleşme Oranı          : {match_pct:.2f}%")
    
    # İlk 10 uyumsuzluk
    if mismatches:
        print(f"\n--- İlk 10 Uyumsuzluk ---")
        for m in mismatches[:10]:
            print(f"  {m['ideal_time']} | {m['ideal_dir']:5s} | {m['ideal_price']:.2f}")
