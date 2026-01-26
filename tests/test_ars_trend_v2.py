# -*- coding: utf-8 -*-
"""
ARS Trend v2 Strateji - IdealData Karşılaştırma
ideal_signals.csv (8857 işlem) ile Python ARS Trend v2 sinyallerini karşılaştırır
"""

import sys
import io
import pandas as pd
import numpy as np
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2, Signal

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
    """IdealData işlem listesini yükle (ideal_signals.csv - 8857 işlem)"""
    csv_path = 'd:/Projects/IdealQuant/data/ideal_signals.csv'
    
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='cp1254')
    except:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
        
    df.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 
                  'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
                  
    # Tarih parse
    df['Time'] = pd.to_datetime(df['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
    df['Direction'] = df['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')
    
    return df[['Time', 'Direction', 'AcilisFyt', 'KarZarar']].dropna(subset=['Time'])


def run_strategy_test(df_price):
    """Python stratejisini çalıştır"""
    opens = df_price['Acilis'].values.tolist()
    highs = df_price['Yuksek'].values.tolist()
    lows = df_price['Dusuk'].values.tolist()
    closes = df_price['Kapanis'].values.tolist()
    typical = ((df_price['Yuksek'] + df_price['Dusuk'] + df_price['Kapanis']) / 3).tolist()
    times = df_price['DateTime'].tolist()
    
    # 1 Dakikalık Parametreler (ideal_signals.csv ile uyumlu olması beklenen)
    config = StrategyConfigV2(
        ars_ema_period = 3,
        ars_atr_period = 10,
        ars_atr_mult = 0.5,
        ars_min_band = 0.002,
        ars_max_band = 0.015,
        
        momentum_period = 5,
        breakout_period = 10,
        
        kar_al_pct = 3.0,
        iz_stop_pct = 1.5
    )
    
    strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, config)
    
    signals = []
    current_pos = "FLAT"
    entry_price = 0.0
    extreme_price = 0.0
    
    # P&L Hesaplama
    total_pnl = 0.0
    trades = []
    
    for i in range(len(closes)):
        # Extreme Fiyat Güncelle (İzleyen Stop için)
        if current_pos == "LONG":
            extreme_price = max(extreme_price, highs[i])
        elif current_pos == "SHORT":
            extreme_price = min(extreme_price, lows[i])
            
        sig = strategy.get_signal(i, current_pos, entry_price, extreme_price)
        
        if sig != Signal.NONE and sig != current_pos:
            
            # Kapanış İşlemi
            if current_pos != "FLAT":
                pnl = 0.0
                if current_pos == "LONG":
                    pnl = closes[i] - entry_price
                else:
                    pnl = entry_price - closes[i]
                total_pnl += pnl
            
            # Yeni Pozisyon
            if sig == Signal.LONG:
                current_pos = "LONG"
                entry_price = closes[i]
                extreme_price = closes[i] # Reset extreme
                signals.append({
                    'BarIndex': i,
                    'Direction': 'LONG',
                    'Price': closes[i]
                })
                
            elif sig == Signal.SHORT:
                current_pos = "SHORT"
                entry_price = closes[i]
                extreme_price = closes[i] # Reset extreme
                signals.append({
                    'BarIndex': i,
                    'Direction': 'SHORT',
                    'Price': closes[i]
                })
                
            elif sig == Signal.FLAT:
                current_pos = "FLAT"
                entry_price = 0.0
                extreme_price = 0.0
                
    return pd.DataFrame(signals), total_pnl


def main():
    print("=" * 70)
    print("ARS Trend v2 (Asıl Strateji) - IdealData Karşılaştırması")
    print("=" * 70)
    
    # Veri Yükle
    print("\n[1] Veriler yükleniyor...")
    df_price = load_price_data()
    df_ideal = load_ideal_trades()
    
    print(f"    Fiyat Verisi: {len(df_price)} bar")
    print(f"    IdealData İşlem: {len(df_ideal)} (ideal_signals.csv)")
    
    # Strateji Çalıştır
    print("\n[2] Python stratejisi çalıştırılıyor...")
    df_py_signals, py_pnl = run_strategy_test(df_price)
    
    if len(df_py_signals) == 0:
        print("❌ Sinyal üretilemedi!")
        return
        
    # Zaman ekle
    df_py_signals['Time'] = df_price['DateTime'].iloc[df_py_signals['BarIndex'].values].values
    df_py_signals['TimeDakika'] = pd.to_datetime(df_py_signals['Time']).dt.floor('min')
    
    print(f"    Python Sinyal Sayısı: {len(df_py_signals)}")
    print(f"    Python P&L: {py_pnl:.2f}")
    
    # Karşılaştırma
    print("\n[3] Karşılaştırma yapılıyor...")
    
    # Ideal verisini hazırla
    df_ideal['TimeDakika'] = df_ideal['Time'].dt.floor('min')
    
    start = max(df_ideal['Time'].min(), df_py_signals['Time'].min())
    end = min(df_ideal['Time'].max(), df_py_signals['Time'].max())
    
    ideal_subset = df_ideal[(df_ideal['Time'] >= start) & (df_ideal['Time'] <= end)]
    py_subset = df_py_signals[(df_py_signals['Time'] >= start) & (df_py_signals['Time'] <= end)]
    
    print(f"    Aralık: {start} - {end}")
    print(f"    Ideal İşlem: {len(ideal_subset)}")
    print(f"    Python Sinyal: {len(py_subset)}")
    
    matches = 0
    for _, row in ideal_subset.iterrows():
        match = py_subset[
            (py_subset['TimeDakika'] == row['TimeDakika']) & 
            (py_subset['Direction'] == row['Direction'])
        ]
        if len(match) > 0:
            matches += 1
            
    accuracy = (matches / len(ideal_subset)) * 100
    
    print("-" * 50)
    print(f"Tam Eşleşme (Girişler): {matches}")
    print(f"Uyumluluk             : %{accuracy:.2f}")
    print("-" * 50)
    
    # P&L Kıyas
    ideal_pnl_raw = ideal_subset['KarZarar'].astype(str).str.replace('.', '').str.replace(',', '.')
    ideal_pnl = pd.to_numeric(ideal_pnl_raw, errors='coerce').sum()
    
    print(f"Ideal P&L: {ideal_pnl:.2f}")
    print(f"Python P&L: {py_pnl:.2f}")
    
    if accuracy > 90:
        print("✅ Yüksek Uyum - Asıl strateji doğrulandı!")
    else:
        print("❌ Düşük Uyum - Parametre veya mantık farkı var.")

if __name__ == "__main__":
    main()
