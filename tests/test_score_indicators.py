# -*- coding: utf-8 -*-
"""
Score Based İndikatör Karşılaştırma Testi
IdealData vs Python
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
import numpy as np
from indicators.core import ARS, QQEF, RVI, Qstick, NetLot, ADX, SMA

def load_ideal_data(csv_path):
    print(f"IdealData yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='utf-8') # Encoding farklı olabilir
    
    # Kolon isimlerini temizle (boşluk vs)
    df.columns = [c.strip() for c in df.columns]

    # Tüm verileri float'a çevir (hata verirse NaN yap)
    for col in df.columns:
        if col not in ['Date', 'Time', 'Tarih', 'Saat']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def load_raw_data(csv_path):
    print(f"Ham veri yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None)
    
    # Header kontrolü
    if isinstance(df.iloc[0,0], str) and "Tarih" in df.iloc[0,0]:
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    else:
        df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

    opens = df['Acilis'].values.astype(float).tolist()
    highs = df['Yuksek'].values.astype(float).tolist()
    lows = df['Dusuk'].values.astype(float).tolist()
    closes = df['Kapanis'].values.astype(float).tolist()
    typical = ((df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3).values.astype(float).tolist()
    
    return opens, highs, lows, closes, typical

def compare_indicator(name, ideal_vals, python_vals, tolerance=1e-4):
    print(f"\n--- {name} Karşılaştırması ---")
    
    # Uzunluk kontolü
    n = min(len(ideal_vals), len(python_vals))
    ideal_vals = np.array(ideal_vals[:n])
    python_vals = np.array(python_vals[:n])
    
    # NaN temizliği (baştaki ısınma turları) ve ilk 50 barı atla
    valid_mask = ~np.isnan(ideal_vals) & ~np.isnan(python_vals) & (ideal_vals != 0)
    
    # İlk 500 barı maskele (False yap) - ADX convergence testi
    if len(valid_mask) > 500:
        valid_mask[:500] = False
    
    if np.sum(valid_mask) == 0:
        print("Çok fazla NaN veya 0, karşılaştırılamadı.")
        return
    
    diff = np.abs(ideal_vals[valid_mask] - python_vals[valid_mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max Fark: {max_diff:.6f}")
    print(f"Ort Fark: {mean_diff:.6f}")
    
    if max_diff < tolerance:
        print(f"✅ {name}: BAŞARILI")
    else:
        print(f"❌ {name}: BAŞARISIZ (Tolerans: {tolerance})")
        # İlk 5 hatayı göster
        error_indices = np.where(diff > tolerance)[0]
        for i in error_indices[:5]:
             idx = np.where(valid_mask)[0][i] # Orijinal index
             print(f"  Bar {idx}: Ideal={ideal_vals[idx]:.4f}, Python={python_vals[idx]:.4f}, Fark={diff[i]:.4f}")

def main():
    # 1. Ideal Verilerini Yükle (Export edilen)
    ideal_csv = "d:/Projects/IdealQuant/data/ideal_score_indicators.csv"
    try:
        df_ideal = load_ideal_data(ideal_csv)
    except Exception as e:
        print(f"Hata: {e}")
        # Try different encoding or loading method if needed
        try:
             df_ideal = pd.read_csv(ideal_csv, sep=';', decimal=',', encoding='cp1254')
             df_ideal.columns = [c.strip() for c in df_ideal.columns]
        except:
             return

    # 2. Ham Verileri Yükle (Hesaplama için)
    # Not: Bu ham veri dosyası IdealData'daki ile AYNI olmalı (VIP_X030T_1dk_.csv)
    raw_csv = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    opens, highs, lows, closes, typical = load_raw_data(raw_csv)
    
    # Veri uzunluklarını eşitle
    # IdealData exportu ham verinin tamamı olmayabilir veya ham veri daha uzun olabilir.
    # Tarih/Saat üzerinden eşleştirme yapmak en doğrusu ama şimdilik basitçe son N barı alalım.
    # Veya IdealData exportu tüm veriyi kapsıyorsa direkt karşılaştırabiliriz.
    
    # 3. Python ile Hesaplama
    print("\nİndikatörler hesaplanıyor...")
    
    # ARS
    py_ars = ARS(typical, ema_period=3, k=0.0123)
    
    # QQEF
    py_qqef, py_qqes = QQEF(closes, rsi_period=14, smooth_period=5)
    
    # RVI
    py_rvi, py_rvi_sig = RVI(opens, highs, lows, closes, period=10)
    
    # Qstick
    py_qstick = Qstick(opens, closes, period=8)
    
    # NetLot (MA'lı hali)
    py_netlot = NetLot(opens, highs, lows, closes)
    py_netlot_ma = SMA(py_netlot, 5)
    
    # ADX
    py_adx = ADX(highs, lows, closes, period=14)
    
    # 4. Karşılaştırma
    # Not: Veri boyutu farkı olabilir. data/ideal_score_indicators.csv dosyasındaki BarNo ile eşleştirelim.
    # Eğer birebir aynı veri seti kullanıldıysa indexler tutacaktır.
    
    # IdealData export dosyasında muhtemelen tüm barlar var.
    # Ham veri dosyasında da tüm barlar var.
    # Boyut kontrolü:
    print(f"Ideal Veri: {len(df_ideal)} bar")
    print(f"Ham Veri  : {len(closes)} bar")
    
    min_len = min(len(df_ideal), len(closes))
    print(f"Karşılaştırılan: {min_len} bar")
    
    compare_indicator("ARS", df_ideal['ARS'].values, py_ars)
    compare_indicator("QQEF", df_ideal['QQEF'].values, py_qqef)
    compare_indicator("QQES", df_ideal['QQES'].values, py_qqes)
    compare_indicator("RVI", df_ideal['RVI'].values, py_rvi)
    compare_indicator("RVI_Sig", df_ideal['RVI_Sig'].values, py_rvi_sig)
    compare_indicator("Qstick", df_ideal['Qstick'].values, py_qstick)
    compare_indicator("NetLot (MA)", df_ideal['NetLot'].values, py_netlot_ma)
    compare_indicator("ADX", df_ideal['ADX'].values, py_adx)

if __name__ == "__main__":
    main()
