# -*- coding: utf-8 -*-
"""
Final Signal Comparison
IdealData Trade List ("ideal_signals.csv") vs Python Signals ("python_signals.csv")

Matches trades based on Date/Time and Direction.
"""

import pandas as pd
import sys
import io

# Konsol encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def load_ideal_trades():
    csv_path = 'd:/Projects/IdealQuant/data/ideal_signals_optimized.csv'
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig') # Genelde utf-8-sig
    except:
        df = pd.read_csv(csv_path, sep=';', encoding='cp1254') # Excel default
    
    # Sütun isimlerini temizle/belirle
    # Index ile alalım: 3: Acilis Tarihi, 1: Yon
    # Kolon isimleri karışık olabilir
    
    # Yeni DataFrame oluştur
    trades = pd.DataFrame()
    
    # Tarih parse (Format: 8.01.2025 20:40)
    # Pandas to_datetime ile
    # Sütun 3 (0-indexed) -> Açılış Tarihi
    trades['Time'] = pd.to_datetime(df.iloc[:, 3], format='%d.%m.%Y %H:%M', errors='coerce')
    
    # Yön haritalama
    # Alış -> LONG, Satış -> SHORT
    # Sütun 1 -> Yön
    direction_map = {'Alış': 'LONG', 'Satış': 'SHORT', 'Alis': 'LONG', 'Satis': 'SHORT'}
    trades['Direction'] = df.iloc[:, 1].map(direction_map)
    
    # Geçersiz tarihleri temizle
    trades = trades.dropna(subset=['Time'])
    
    return trades.sort_values('Time').reset_index(drop=True)

def load_python_signals():
    csv_path = "d:/Projects/IdealQuant/tests/python_signals.csv"
    df = pd.read_csv(csv_path, sep=';')
    
    # DateTime birleştir
    # Tarih: 23.01.2026, Saat: 21:49:00
    # Saat saniyesiz lazım olabilir, ama to_datetime halleder
    df['DateTimeStr'] = df['Tarih'] + ' ' + df['Saat']
    df['Time'] = pd.to_datetime(df['DateTimeStr'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    
    # Yön haritalama: A -> LONG, S -> SHORT
    dir_map = {'A': 'LONG', 'S': 'SHORT', 'F': 'FLAT'}
    df['Direction'] = df['Sinyal'].map(dir_map)
    
    # Sadece LONG/SHORT sinyalleri (FLAT çıkış işlemidir, Ideal listesinde var mı?)
    # Ideal işlem listesi sadece "Giriş"leri gösterir genelde.
    # Ve "Kapanış Tarihi" ile çıkışı gösterir.
    # Bizim Python sinyallerimiz her bar için durum bildirir.
    # Bizim karşılaştırmamız gereken: "Pozisyon Değişimi" anları.
    
    # Python dosyasında zaten sadece değişim anlarını kaydettik (Sinyal != "")
    # Ama FLAT sinyali (Pozisyon Kapatma) Ideal listesinde "Kapanış Tarihi"ne denk gelir.
    # Biz sadece "Giriş" (Açılış) sinyallerini karşılaştıralım ilk etapta.
    
    trades = df[df['Direction'].isin(['LONG', 'SHORT'])].copy()
    
    return trades[['Time', 'Direction']].sort_values('Time').reset_index(drop=True)

def compare_signals():
    print("IdealData İşlemleri Yükleniyor...")
    ideal = load_ideal_trades()
    print(f"Ideal İşlem Sayısı: {len(ideal)}")
    
    print("\nPython Sinyalleri Yükleniyor...")
    python = load_python_signals()
    print(f"Python Sinyal Sayısı (Girişler): {len(python)}")
    
    # Zaman aralığı kesişimi
    if len(ideal) == 0 or len(python) == 0:
        print("Veri yok!")
        return

    start_date = max(ideal['Time'].min(), python['Time'].min())
    end_date = min(ideal['Time'].max(), python['Time'].max())
    
    print(f"\nKarşılaştırma Aralığı: {start_date} - {end_date}")
    
    ideal_subset = ideal[(ideal['Time'] >= start_date) & (ideal['Time'] <= end_date)].copy()
    python_subset = python[(python['Time'] >= start_date) & (python['Time'] <= end_date)].copy()
    
    print(f"Aralıktaki Ideal İşlem: {len(ideal_subset)}")
    print(f"Aralıktaki Python Sinyal: {len(python_subset)}")
    
    matches = 0
    mismatches = 0
    missing_py = 0
    missing_id = 0
    
    # Tolerans (Dakika)
    tolerance = pd.Timedelta(minutes=0) 
    # CSV'den okuduğumuz için birebir aynı olmalı (Data Feed mantığı)
    
    # Ideal üzerinden döngü
    for idx, row in ideal_subset.iterrows():
        t = row['Time']
        d = row['Direction']
        
        # Python'da aynı zamanda sinyal var mı?
        # Saniye farkını yoksay (Ideal'da saniye yok)
        # Python: 20:40:00, Ideal: 20:40:00 (datetime eşitliği genelde saniye 00 ise tutar)
        
        # Tam eşleşme ara
        match = python_subset[python_subset['Time'] == t]
        
        if len(match) > 0:
            py_d = match.iloc[0]['Direction']
            if py_d == d:
                matches += 1
            else:
                mismatches += 1
                # print(f"Yön Uyuşmazlığı: {t} Ideal:{d} Python:{py_d}")
        else:
            missing_py += 1
            # print(f"Python'da Eksik: {t} {d}")
            
    # Python fazlalıkları
    # (Ters döngü veya set farkı ile bulunabilir, şimdilik skor yeterli)
    
    total_checks = len(ideal_subset)
    accuracy = (matches / total_checks) * 100 if total_checks > 0 else 0
    
    print("\n--- SONUÇLAR ---")
    print(f"Tam Eşleşme (Zaman+Yön): {matches}")
    print(f"Yön Hatası             : {mismatches}")
    print(f"Python'da Eksik (Sinyal Yok): {missing_py}")
    print(f"Uyumluluk Oranı        : %{accuracy:.2f}")
    
    if accuracy > 99.0:
        print("\n✅ MÜKEMMEL UYUM! Sistem canlıya hazır.")
    elif accuracy > 90.0:
        print("\n⚠️ Yüksek Uyum (%90+). Ufak zamanlama farkları olabilir.")
    else:
        print("\n❌ Uyumsuzluk var. İndikatör veya mantık farkı.")

if __name__ == "__main__":
    compare_signals()
