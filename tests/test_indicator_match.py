# -*- coding: utf-8 -*-
"""
IdealQuant - İndikatör Uyumu Doğrulama Testi
Faz 1.2: IdealData ile Python hesaplamalarını karşılaştır
"""

import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
import numpy as np
from pathlib import Path
from indicators.core import SMA, EMA, RSI, ATR


def compare_indicator(name: str, ideal_values: np.ndarray, python_values: np.ndarray, 
                      tolerance_pct: float = 1.0) -> dict:
    """İki indikatör serisini karşılaştır."""
    min_len = min(len(ideal_values), len(python_values))
    ideal_values = ideal_values[:min_len]
    python_values = python_values[:min_len]
    
    mask = ~(np.isnan(ideal_values) | np.isnan(python_values))
    ideal_clean = ideal_values[mask]
    python_clean = python_values[mask]
    
    if len(ideal_clean) == 0:
        return {'name': name, 'status': 'ERROR', 'is_match': False, 'max_pct_diff': 100, 'mean_pct_diff': 100, 'sample_count': 0}
    
    abs_diff = np.abs(ideal_clean - python_clean)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diff = np.where(ideal_clean != 0, abs_diff / np.abs(ideal_clean) * 100, 0)
    
    max_pct_diff = np.nanmax(pct_diff)
    mean_pct_diff = np.nanmean(pct_diff)
    is_match = max_pct_diff < tolerance_pct
    
    return {
        'name': name,
        'status': 'OK' if is_match else 'FAIL',
        'max_pct_diff': max_pct_diff,
        'mean_pct_diff': mean_pct_diff,
        'sample_count': len(ideal_clean),
        'is_match': is_match
    }


def test_indicator_match():
    print("\n" + "=" * 70)
    print("IdealQuant - İndikatör Uyumu Doğrulama")
    print("=" * 70)
    
    # 1. IdealData export
    export_path = Path("d:/Projects/IdealQuant/data/ideal_ind.csv")
    if not export_path.exists():
        print(f"\n[HATA] Export dosyası bulunamadı: {export_path}")
        return False
    
    print(f"\n[1] IdealData export: {export_path.name}")
    ideal_df = pd.read_csv(export_path, sep=';', decimal='.')
    print(f"    -> {len(ideal_df)} satır, Bar: {ideal_df['BarNo'].iloc[0]} - {ideal_df['BarNo'].iloc[-1]}")
    
    # 2. Orijinal CSV - TÜM VERİYİ OKU
    csv_path = Path("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv")
    print(f"\n[2] Orijinal CSV: {csv_path.name}")
    
    raw_df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
    raw_df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    print(f"    -> {len(raw_df)} toplam bar")
    
    # 3. TÜM VERİ ÜZERİNDEN İNDİKATÖRLERİ HESAPLA
    print("\n[3] Python indikatörleri (tam veri üzerinden)...")
    
    close_full = raw_df['Kapanis'].values.astype(float)
    high_full = raw_df['Yuksek'].values.astype(float)
    low_full = raw_df['Dusuk'].values.astype(float)
    
    py_sma20_full = np.array(SMA(close_full, 20))
    py_ema20_full = np.array(EMA(close_full, 20))
    py_rsi14_full = np.array(RSI(close_full, 14))
    py_atr14_full = np.array(ATR(high_full, low_full, close_full, 14))
    
    # 4. IdealData bar numaralarına göre slice al
    bar_indices = ideal_df['BarNo'].values.astype(int)
    
    py_sma20 = py_sma20_full[bar_indices]
    py_ema20 = py_ema20_full[bar_indices]
    py_rsi14 = py_rsi14_full[bar_indices]
    py_atr14 = py_atr14_full[bar_indices]
    
    # 5. Karşılaştırma
    print("\n[4] Karşılaştırma sonuçları:")
    print("-" * 70)
    print(f"   {'İndikatör':12} | {'Durum':6} | {'Max Fark %':12} | {'Ort Fark %':12} | {'Örnek':8}")
    print("-" * 70)
    
    results = []
    for name, ideal_col, py_vals in [
        ('SMA20', ideal_df['SMA20'].values, py_sma20),
        ('EMA20', ideal_df['EMA20'].values, py_ema20),
        ('RSI14', ideal_df['RSI14'].values, py_rsi14),
        ('ATR14', ideal_df['ATR14'].values, py_atr14),
    ]:
        result = compare_indicator(name, ideal_col, py_vals)
        results.append(result)
        icon = "✓" if result['is_match'] else "✗"
        print(f"   [{icon}] {result['name']:10} | {result['status']:6} | {result['max_pct_diff']:10.4f}% | {result['mean_pct_diff']:10.4f}% | {result['sample_count']:8}")
    
    # 6. Özet
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r['is_match'])
    total = len(results)
    
    if passed == total:
        print(f"SONUÇ: TÜM TESTLER BAŞARILI ({passed}/{total})")
    else:
        print(f"SONUÇ: {total - passed} TEST BAŞARISIZ ({passed}/{total})")
    print("=" * 70)
    return passed == total


if __name__ == "__main__":
    success = test_indicator_match()
    sys.exit(0 if success else 1)
