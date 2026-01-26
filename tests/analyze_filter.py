# -*- coding: utf-8 -*-
"""
YatayFiltre Kararlılık Analizi
Filtre durumu ne sıklıkla değişiyor?
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
import numpy as np
from datetime import datetime
from filters.yatay_filtre import YatayFiltre


def load_data(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None)
    # İlk satır header olabilir, kontrol et
    if isinstance(df.iloc[0,0], str) and "Tarih" in df.iloc[0,0]:
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
        # Sütun isimlerini düzelt (boşluk vs olabilir)
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    else:
        # Header yoksa manuel ver
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

    opens = df['Acilis'].values.astype(float)
    highs = df['Yuksek'].values.astype(float)
    lows = df['Dusuk'].values.astype(float)
    closes = df['Kapanis'].values.astype(float)
    typical = (highs + lows + closes) / 3
    return closes.tolist(), highs.tolist(), lows.tolist(), typical.tolist()


def analyze_filter_stability():
    print("\n" + "=" * 70)
    print("  YatayFiltre Kararlılık Analizi")
    print("=" * 70)
    
    # Veri yükle
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        closes, highs, lows, typical = load_data(csv_path)
    except FileNotFoundError:
        print("Data dosyası bulunamadı!")
        return

    # Filtre oluştur
    filtre = YatayFiltre(closes, highs, lows, typical)
    
    # Analiz
    changes = 0
    trend_duration = []
    flat_duration = []
    
    current_state = filtre.islem_izni(50)
    current_duration = 0
    
    states = []
    
    for i in range(51, len(closes)):
        state = filtre.islem_izni(i)
        states.append(1 if state else 0)
        
        if state != current_state:
            changes += 1
            if current_state: # Trend idi
                trend_duration.append(current_duration)
            else: # Yatay idi
                flat_duration.append(current_duration)
            
            current_state = state
            current_duration = 1
        else:
            current_duration += 1
            
    # Son durum
    if current_state:
        trend_duration.append(current_duration)
    else:
        flat_duration.append(current_duration)
        
    print(f"  Toplam Bar Sayısı : {len(closes)}")
    print(f"  Toplam Değişim    : {changes}")
    print(f"  Değişim Oranı     : %{changes / len(closes) * 100:.2f}")
    print("-" * 60)
    print(f"  Trend Süresi (Ort): {np.mean(trend_duration):.1f} bar")
    print(f"  Trend Süresi (Min): {np.min(trend_duration)} bar")
    print(f"  Trend Süresi (Max): {np.max(trend_duration)} bar")
    print("-" * 60)
    print(f"  Yatay Süresi (Ort): {np.mean(flat_duration):.1f} bar")
    print(f"  Yatay Süresi (Min): {np.min(flat_duration)} bar")
    print("-" * 60)
    
    # Ardışık kısa süreli değişimler (Gürültü)
    short_changes = sum(1 for d in trend_duration if d < 5) + sum(1 for d in flat_duration if d < 5)
    print(f"  Kısa Süreli (<5 bar) Durumlar: {short_changes}")
    print(f"  Gürültü Oranı                : %{short_changes / (len(trend_duration) + len(flat_duration)) * 100:.2f}")


if __name__ == "__main__":
    analyze_filter_stability()
