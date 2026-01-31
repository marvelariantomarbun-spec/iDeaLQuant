"""
IdealQuant - Indicator Calibration Tests

Kalibre edilecek indikatörler:
- ATR (Average True Range)
- HHV/LLV (Volume için)
- MFI (Money Flow Index) - yeni eklendi

Bu script mevcut verileri kullanarak indikatörleri doğrular.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.core import (
    ATR, HHV, LLV, RSI, Momentum, 
    MoneyFlowIndex, SMA, EMA
)


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
        return {'name': name, 'status': 'ERROR', 'is_match': False, 
                'max_pct_diff': 100, 'mean_pct_diff': 100, 'sample_count': 0}
    
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


def test_ars_indicators():
    """
    Test HHV, LLV, RSI, Momentum indicators against ideal_ars_v2_data.csv
    """
    print("\n" + "=" * 70)
    print("İNDİKATÖR KALİBRASYON TESTİ - ARS Trend v2 Verileri")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    # 1. IdealData export
    export_path = project_root / "data" / "ideal_ars_v2_data.csv"
    if not export_path.exists():
        print(f"\n[HATA] Export dosyası bulunamadı: {export_path}")
        return False
    
    print(f"\n[1] IdealData export: {export_path.name}")
    ideal_df = pd.read_csv(export_path, sep=';', decimal='.')
    print(f"    -> {len(ideal_df)} satır")
    print(f"    -> Kolonlar: {list(ideal_df.columns)}")
    
    # 2. Parametreleri belirle (Export_ARS_v2.txt'den)
    BREAKOUT_PERIOD = 20  # HHV/LLV için
    MOMENTUM_PERIOD = 10  # Momentum için
    RSI_PERIOD = 14       # RSI için
    
    # 3. Orijinal CSV yükle
    csv_path = project_root / "data" / "VIP_X030T_1dk_.csv"
    if not csv_path.exists():
        print(f"\n[HATA] Orijinal CSV bulunamadı: {csv_path}")
        return False
        
    print(f"\n[2] Orijinal CSV: {csv_path.name}")
    raw_df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
    raw_df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    print(f"    -> {len(raw_df)} toplam bar")
    
    # 4. Veriyi hazırla
    close = raw_df['Kapanis'].values.astype(float)
    high = raw_df['Yuksek'].values.astype(float)
    low = raw_df['Dusuk'].values.astype(float)
    
    # 5. Python indikatörleri hesapla (TÜM VERİ)
    print("\n[3] Python indikatörleri hesaplanıyor...")
    
    py_hhv = np.array(HHV(high.tolist(), BREAKOUT_PERIOD))
    py_llv = np.array(LLV(low.tolist(), BREAKOUT_PERIOD))
    py_momentum = np.array(Momentum(close.tolist(), MOMENTUM_PERIOD))
    py_rsi = np.array(RSI(close.tolist(), RSI_PERIOD))
    
    # 6. Tarih-saat bazlı eşleşme
    # Ham veri Tarih ve Saat kolonlarını birleştir
    # Saat formatı: Raw CSV'de HH:MM:SS, IdealData'da HH:MM - saniyeyi kes
    raw_df['Time_Short'] = raw_df['Saat'].str[:5]  # "21:11:00" -> "21:11"
    raw_df['DateTime'] = raw_df['Tarih'] + ' ' + raw_df['Time_Short']
    
    # IdealData Date ve Time kolonlarını birleştir
    ideal_df['DateTime'] = ideal_df['Date'] + ' ' + ideal_df['Time']
    
    # DateTime index oluştur
    raw_dt_to_idx = {dt: i for i, dt in enumerate(raw_df['DateTime'])}
    
    # IdealData'daki her satır için raw_df'deki indeksi bul
    aligned_indices = []
    for dt in ideal_df['DateTime']:
        if dt in raw_dt_to_idx:
            aligned_indices.append(raw_dt_to_idx[dt])
        else:
            aligned_indices.append(-1)
    
    aligned_indices = np.array(aligned_indices)
    valid_mask = aligned_indices >= 0
    
    print(f"\n    -> Eslesen bar sayisi: {valid_mask.sum()}/{len(ideal_df)}")
    
    if valid_mask.sum() == 0:
        print("[HATA] Hicbir bar eslesmedi!")
        return False
    
    # Filter to valid aligned bars
    ideal_hhv = ideal_df['HHV'].values[valid_mask]
    ideal_llv = ideal_df['LLV'].values[valid_mask]
    ideal_momentum = ideal_df['Momentum'].values[valid_mask]
    ideal_rsi = ideal_df['RSI'].values[valid_mask]
    
    valid_indices = aligned_indices[valid_mask]
    py_hhv_slice = py_hhv[valid_indices]
    py_llv_slice = py_llv[valid_indices]
    py_momentum_slice = py_momentum[valid_indices]
    py_rsi_slice = py_rsi[valid_indices]
    
    # 7. Karşılaştırma
    print("\n[4] Karşılaştırma sonuçları:")
    print("-" * 70)
    print(f"   {'İndikatör':12} | {'Durum':6} | {'Max Fark %':12} | {'Ort Fark %':12} | {'Örnek':8}")
    print("-" * 70)
    
    results = []
    tests = [
        ('HHV20', ideal_hhv, py_hhv_slice),
        ('LLV20', ideal_llv, py_llv_slice),
        ('Momentum', ideal_momentum, py_momentum_slice),
        ('RSI14', ideal_rsi, py_rsi_slice),
    ]
    
    for name, ideal_vals, py_vals in tests:
        result = compare_indicator(name, ideal_vals, py_vals, tolerance_pct=3.0)
        results.append(result)
        icon = "OK" if result['is_match'] else "XX"
        print(f"   [{icon:2}] {result['name']:10} | {result['status']:6} | "
              f"{result['max_pct_diff']:10.4f}% | {result['mean_pct_diff']:10.4f}% | "
              f"{result['sample_count']:8}")
        
        # Detaylı fark analizi (başarısız ise)
        if not result['is_match']:
            print(f"\n       [DEBUG] İlk 5 bar karşılaştırması:")
            for i in range(min(5, len(ideal_vals))):
                print(f"       Bar {i}: Ideal={ideal_vals[i]:.4f}, Python={py_vals[i]:.4f}, "
                      f"Diff={abs(ideal_vals[i] - py_vals[i]):.4f}")
    
    # 8. Özet
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r['is_match'])
    total = len(results)
    
    if passed == total:
        print(f"SONUÇ: TÜM TESTLER BAŞARILI ({passed}/{total})")
    else:
        print(f"SONUÇ: {total - passed} TEST BAŞARISIZ ({passed}/{total})")
    print("=" * 70)
    
    return passed == total


def test_atr_calibration():
    """
    ATR kalibrasyonu - ideal_ind.csv kullanarak
    """
    print("\n" + "=" * 70)
    print("İNDİKATÖR KALİBRASYON TESTİ - ATR")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    # ideal_ind.csv ATR içeriyor
    export_path = project_root / "data" / "ideal_ind.csv"
    if not export_path.exists():
        print(f"\n[HATA] Export dosyası bulunamadı: {export_path}")
        return False
    
    print(f"\n[1] IdealData export: {export_path.name}")
    ideal_df = pd.read_csv(export_path, sep=';', decimal='.')
    print(f"    -> {len(ideal_df)} satır")
    
    # Orijinal CSV
    csv_path = project_root / "data" / "VIP_X030T_1dk_.csv"
    raw_df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
    raw_df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    close = raw_df['Kapanis'].values.astype(float)
    high = raw_df['Yuksek'].values.astype(float)
    low = raw_df['Dusuk'].values.astype(float)
    
    # Python ATR hesapla
    print("\n[2] Python ATR hesaplanıyor...")
    py_atr = np.array(ATR(high.tolist(), low.tolist(), close.tolist(), 14))
    
    # BarNo'ya göre slice al
    if 'BarNo' in ideal_df.columns:
        bar_indices = ideal_df['BarNo'].values.astype(int)
        py_atr_slice = py_atr[bar_indices]
        ideal_atr = ideal_df['ATR14'].values
        
        result = compare_indicator('ATR14', ideal_atr, py_atr_slice)
        icon = "OK" if result['is_match'] else "XX"
        
        print("\n[3] Karşılaştırma:")
        print("-" * 70)
        print(f"   [{icon}] {result['name']:10} | {result['status']:6} | "
              f"{result['max_pct_diff']:10.4f}% | {result['mean_pct_diff']:10.4f}% | "
              f"{result['sample_count']:8}")
        
        return result['is_match']
    else:
        print("\n[HATA] BarNo kolonu bulunamadı")
        return False


def test_mfi_sanity():
    """
    MFI mantık testi - IdealData export olmadan temel çalışma kontrolü
    """
    print("\n" + "=" * 70)
    print("MFI TEMEL FONKSİYON TESTİ")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "VIP_X030T_1dk_.csv"
    
    if not csv_path.exists():
        print(f"\n[HATA] CSV bulunamadı: {csv_path}")
        return False
        
    raw_df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
    raw_df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    high = raw_df['Yuksek'].values.astype(float)
    low = raw_df['Dusuk'].values.astype(float)
    close = raw_df['Kapanis'].values.astype(float)
    volume = raw_df['Hacim'].values.astype(float)
    
    print(f"\n[1] Veri: {len(close)} bar")
    
    # MFI hesapla
    print("[2] MFI hesaplanıyor...")
    mfi = MoneyFlowIndex(high.tolist(), low.tolist(), close.tolist(), volume.tolist(), 14)
    mfi = np.array(mfi)
    
    # Temel kontroller
    print("\n[3] Temel kontroller:")
    print(f"    - MFI aralığı: {mfi.min():.2f} - {mfi.max():.2f}")
    print(f"    - İlk 5 değer: {mfi[14:19]}")
    print(f"    - Son 5 değer: {mfi[-5:]}")
    
    # MFI 0-100 aralığında olmalı
    valid_range = (mfi[14:] >= 0).all() and (mfi[14:] <= 100).all()
    print(f"\n    - Gecerli aralik (0-100): {'[OK]' if valid_range else '[XX]'}")
    
    # NaN olmamali
    no_nan = not np.isnan(mfi[14:]).any()
    print(f"    - NaN yok: {'[OK]' if no_nan else '[XX]'}")
    
    success = valid_range and no_nan
    print(f"\n[SONUÇ] MFI Temel Test: {'BAŞARILI' if success else 'BAŞARISIZ'}")
    
    return success


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IdealQuant - İNDİKATÖR KALİBRASYON TESTLERİ")
    print("=" * 70)
    
    results = []
    
    # Test 1: ARS Trend indicators
    results.append(("ARS Trend Indicators", test_ars_indicators()))
    
    # Test 2: ATR
    results.append(("ATR Calibration", test_atr_calibration()))
    
    # Test 3: MFI sanity check
    results.append(("MFI Sanity", test_mfi_sanity()))
    
    # Summary
    print("\n" + "=" * 70)
    print("GENEL ÖZET")
    print("=" * 70)
    
    for name, success in results:
        icon = "OK" if success else "XX"
        print(f"   [{icon}] {name}")
    
    total_success = sum(1 for _, s in results if s)
    print(f"\nToplam: {total_success}/{len(results)} test başarılı")
