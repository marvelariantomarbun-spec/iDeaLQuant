# -*- coding: utf-8 -*-
"""
ARS Trend Strateji Karşılaştırma Testi
ideal_signals.csv ile Python ARS sinyallerini karşılaştırır
"""

import sys
import io
import pandas as pd
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.ars_trend import ARSTrendStrategy, StrategyConfig, Signal

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_ideal_trades():
    """IdealData işlem listesini yükle"""
    csv_path = 'd:/Projects/IdealQuant/data/ideal_signals.csv'
    
    # Encoding sorununu çöz
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv(csv_path, sep=';', encoding='cp1254')
        except:
            df = pd.read_csv(csv_path, sep=';', encoding='latin1')
    
    # Sütun isimlerini düzelt
    df.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 
                  'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
    
    # Tarih parse
    df['Time'] = pd.to_datetime(df['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
    
    # Yön haritalama
    df['Direction'] = df['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')
    
    return df[['Time', 'Direction', 'AcilisFyt', 'KarZarar']].dropna(subset=['Time'])


def load_price_data():
    """OHLCV fiyat verisini yükle"""
    csv_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    # DateTime oluştur
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    
    return df


def run_ars_comparison():
    print("=" * 70)
    print("ARS Trend - IdealData Sinyal Karşılaştırması")
    print("=" * 70)
    
    # 1. IdealData işlemlerini yükle
    print("\n[1] IdealData işlemleri yükleniyor...")
    ideal_trades = load_ideal_trades()
    print(f"    Toplam işlem: {len(ideal_trades)}")
    print(f"    İlk işlem: {ideal_trades['Time'].min()}")
    print(f"    Son işlem: {ideal_trades['Time'].max()}")
    
    # 2. Fiyat verisini yükle
    print("\n[2] Fiyat verisi yükleniyor...")
    price_df = load_price_data()
    print(f"    Toplam bar: {len(price_df)}")
    
    # 3. ARS stratejisini çalıştır
    print("\n[3] ARS stratejisi hesaplanıyor...")
    
    opens = price_df['Acilis'].values.tolist()
    highs = price_df['Yuksek'].values.tolist()
    lows = price_df['Dusuk'].values.tolist()
    closes = price_df['Kapanis'].values.tolist()
    typical = ((price_df['Yuksek'] + price_df['Dusuk'] + price_df['Kapanis']) / 3).tolist()
    
    # 1dk parametreleri (IdealData varsayılan)
    config = StrategyConfig.for_timeframe(1)
    strategy = ARSTrendStrategy(opens, highs, lows, closes, typical, config)
    
    # Sinyal üret
    python_signals = []
    current_pos = "FLAT"
    entry_price = 0.0
    extreme_price = 0.0
    
    for i in range(len(price_df)):
        sig = strategy.get_signal(i, current_pos, entry_price, extreme_price)
        
        if sig == Signal.LONG:
            python_signals.append({
                'Time': price_df['DateTime'].iloc[i],
                'Direction': 'LONG',
                'Price': closes[i]
            })
            current_pos = "LONG"
            entry_price = closes[i]
            extreme_price = closes[i]
            
        elif sig == Signal.SHORT:
            python_signals.append({
                'Time': price_df['DateTime'].iloc[i],
                'Direction': 'SHORT',
                'Price': closes[i]
            })
            current_pos = "SHORT"
            entry_price = closes[i]
            extreme_price = closes[i]
            
        elif sig == Signal.FLAT:
            current_pos = "FLAT"
            entry_price = 0.0
            extreme_price = 0.0
            
        # Extreme fiyat güncelle
        if current_pos == "LONG":
            extreme_price = max(extreme_price, closes[i])
        elif current_pos == "SHORT":
            extreme_price = min(extreme_price, closes[i])
    
    python_df = pd.DataFrame(python_signals)
    print(f"    Python sinyal sayısı: {len(python_df)}")
    
    # 4. Karşılaştırma
    print("\n[4] Karşılaştırma yapılıyor...")
    
    # Ideal işlemleri dakika hassasiyetine yuvarla
    ideal_trades['TimeDakika'] = ideal_trades['Time'].dt.floor('min')
    python_df['TimeDakika'] = python_df['Time'].dt.floor('min')
    
    # Ortak zaman aralığı
    start = max(ideal_trades['Time'].min(), python_df['Time'].min())
    end = min(ideal_trades['Time'].max(), python_df['Time'].max())
    
    ideal_subset = ideal_trades[(ideal_trades['Time'] >= start) & (ideal_trades['Time'] <= end)]
    python_subset = python_df[(python_df['Time'] >= start) & (python_df['Time'] <= end)]
    
    print(f"    Karşılaştırma aralığı: {start} - {end}")
    print(f"    Aralıktaki Ideal işlem: {len(ideal_subset)}")
    print(f"    Aralıktaki Python sinyal: {len(python_subset)}")
    
    # Eşleşme kontrolü
    matches = 0
    mismatches = 0
    missing = 0
    
    for _, row in ideal_subset.iterrows():
        t = row['TimeDakika']
        d = row['Direction']
        
        match = python_subset[python_subset['TimeDakika'] == t]
        
        if len(match) > 0:
            if match.iloc[0]['Direction'] == d:
                matches += 1
            else:
                mismatches += 1
        else:
            missing += 1
    
    total = len(ideal_subset)
    accuracy = (matches / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("SONUÇLAR")
    print("=" * 70)
    print(f"Tam Eşleşme (Zaman+Yön)  : {matches}")
    print(f"Yön Uyuşmazlığı          : {mismatches}")
    print(f"Python'da Eksik          : {missing}")
    print(f"Uyumluluk Oranı          : %{accuracy:.2f}")
    print("=" * 70)
    
    if accuracy >= 90:
        print("✅ Yüksek uyum!")
    elif accuracy >= 50:
        print("⚠️ Orta uyum - strateji parametrelerini kontrol edin")
    else:
        print("❌ Düşük uyum - strateji mantığını gözden geçirin")
    
    # P&L karşılaştırması
    print("\n[5] P&L Karşılaştırması...")
    
    # IdealData toplam P&L
    ideal_pnl = ideal_trades['KarZarar'].sum()
    print(f"    IdealData Toplam P&L: {ideal_pnl:.2f}")
    
    # Son bakiye
    son_bakiye = ideal_trades['Bakiye'].iloc[-1] if 'Bakiye' in ideal_trades.columns else 0
    # Bakiye formatını temizle
    if isinstance(son_bakiye, str):
        son_bakiye = float(son_bakiye.replace('.', '').replace(',', '.'))
    print(f"    IdealData Son Bakiye: {son_bakiye:.2f}")


if __name__ == "__main__":
    run_ars_comparison()
