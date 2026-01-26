# -*- coding: utf-8 -*-
"""
IdealQuant - Veri Yükleme Testi
Adım 1.1: IdealData CSV uyumluluğu doğrulama
"""

import sys
import io

# Windows konsolunda UTF-8 desteği
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
from pathlib import Path

# =============================================================================
# TEST 1: CSV Dosyasını Oku
# =============================================================================

def test_csv_read():
    """IdealData CSV formatını oku ve analiz et"""
    
    csv_path = Path("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv")
    
    print("=" * 60)
    print("TEST 1: CSV Dosya Okuma")
    print("=" * 60)
    
    # IdealData formatı: noktalı virgül ayırıcı, virgül ondalık
    df = pd.read_csv(
        csv_path, 
        sep=';', 
        decimal=',',
        encoding='cp1254'  # Türkçe karakterler için
    )
    
    print(f"\n[FILE]: {csv_path.name}")
    print(f"[ROWS]: {len(df):,}")
    print(f"[COLS]: {list(df.columns)}")
    
    print("\n[HEAD]:")
    print(df.head())
    
    print("\n[TAIL]:")
    print(df.tail())
    
    print("\n[TYPES]:")
    print(df.dtypes)
    
    return df


# =============================================================================
# TEST 2: Tarih/Saat Parse
# =============================================================================

def test_datetime_parse(df):
    """Tarih ve saat sütunlarını birleştir"""
    
    print("\n" + "=" * 60)
    print("TEST 2: Tarih/Saat Parse")
    print("=" * 60)
    
    # Sütun isimlerini düzelt (encoding sorunu)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    # Tarih + Saat birleştir
    df['datetime'] = pd.to_datetime(
        df['Tarih'] + ' ' + df['Saat'],
        format='%d.%m.%Y %H:%M:%S'
    )
    
    print(f"\n[START]: {df['datetime'].iloc[0]}")
    print(f"[END]: {df['datetime'].iloc[-1]}")
    
    # Tarih aralığı
    date_range = df['datetime'].iloc[-1] - df['datetime'].iloc[0]
    print(f"[DURATION]: {date_range.days} gün")
    
    # Günlük bar sayısı
    daily_bars = df.groupby(df['datetime'].dt.date).size()
    print(f"[AVG BARS]: {daily_bars.mean():.0f}")
    
    return df


# =============================================================================
# TEST 3: OHLCV Sınıfı ile Yükle
# =============================================================================

def test_ohlcv_class(df):
    """OHLCV sınıfı ile veri yükle"""
    
    print("\n" + "=" * 60)
    print("TEST 3: OHLCV Sınıfı Testi")
    print("=" * 60)
    
    from engine.data import OHLCV
    
    # DataFrame'i OHLCV formatına çevir
    ohlcv_df = pd.DataFrame({
        'datetime': df['datetime'],
        'open': df['Acilis'],
        'high': df['Yuksek'],
        'low': df['Dusuk'],
        'close': df['Kapanis'],
        'volume': df['Hacim']
    })
    
    # OHLCV nesnesi oluştur
    data = OHLCV(ohlcv_df)
    
    print(f"\n[OK] OHLCV nesnesi oluşturuldu")
    print(f"[COUNT]: {data.BarSayisi:,}")
    print(f"[FIRST] - O:{data.O[0]} H:{data.H[0]} L:{data.L[0]} C:{data.C[0]}")
    print(f"[LAST] - O:{data.O[-1]} H:{data.H[-1]} L:{data.L[-1]} C:{data.C[-1]}")
    
    # İndeks erişim testi
    bar = data[0]
    print(f"\n[TEST] data[0] testi:")
    print(f"   datetime: {bar.datetime}")
    print(f"   OHLC: {bar.open}, {bar.high}, {bar.low}, {bar.close}")
    print(f"   typical: {bar.typical:.2f}")
    print(f"   range: {bar.range}")
    
    return data


# =============================================================================
# TEST 4: İndikatör Testi
# =============================================================================

def test_indicators(data):
    """Temel indikatör hesaplama testi"""
    
    print("\n" + "=" * 60)
    print("TEST 4: İndikatör Testi")
    print("=" * 60)
    
    from indicators.core import SMA, EMA, RSI, ATR
    
    # SMA testi
    sma20 = SMA(data.C, 20)
    print(f"\n[SMA] SMA(20) - Son 5 değer:")
    for i in range(-5, 0):
        print(f"   Bar {len(data)+i}: Close={data.C[i]:.0f}, SMA={sma20[i]:.2f}")
    
    # EMA testi
    ema20 = EMA(data.C, 20)
    print(f"\n[EMA] EMA(20) - Son 5 değer:")
    for i in range(-5, 0):
        print(f"   Bar {len(data)+i}: Close={data.C[i]:.0f}, EMA={ema20[i]:.2f}")
    
    # RSI testi
    rsi14 = RSI(data.C, 14)
    print(f"\n[RSI] RSI(14) - Son 5 değer:")
    for i in range(-5, 0):
        print(f"   Bar {len(data)+i}: RSI={rsi14[i]:.2f}")
    
    # ATR testi
    atr14 = ATR(data.H, data.L, data.C, 14)
    print(f"\n[ATR] ATR(14) - Son 5 değer:")
    for i in range(-5, 0):
        print(f"   Bar {len(data)+i}: ATR={atr14[i]:.2f}")
    
    return {
        'sma20': sma20,
        'ema20': ema20,
        'rsi14': rsi14,
        'atr14': atr14
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + ">>> IdealQuant Veri Yukleme Testi Basliyor...\n")
    
    try:
        # Test 1: CSV oku
        df = test_csv_read()
        
        # Test 2: Datetime parse
        df = test_datetime_parse(df)
        
        # Test 3: OHLCV sınıfı
        data = test_ohlcv_class(df)
        
        # Test 4: İndikatörler
        indicators = test_indicators(data)
        
        print("\n" + "=" * 60)
        print("OK: TUM TESTLER BASARILI!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
