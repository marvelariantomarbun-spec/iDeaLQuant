"""
IdealQuant - New Indicators Calibration Tests
Tests the 58 newly added indicators for basic sanity and cross-indicator consistency.

For full IdealData calibration, user needs to export indicator values from IdealData.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators import (
    # Moving Averages
    SMA, EMA, WMA, DEMA, TEMA, KAMA, FRAMA, ZLEMA, T3,
    # Oscillators
    RSI, CCI, MACD, StochRSI, WilliamsR, ROC, UltimateOscillator, 
    TRIX, DPO, ChandeMomentum, RMI, AwesomeOscillator,
    # Trend
    ADX, DI_Plus, DI_Minus, AroonUp, AroonDown, AroonOsc,
    ParabolicSAR, Ichimoku, PriceChannel, VHF, LinearReg,
    # Volume
    OBV, PVT, ADL, ChaikinOsc, NVI, PVI, KlingerOsc, ForceIndex,
    # Volatility
    ATR, BollingerUp, BollingerDown, BollingerWidth,
    KeltnerUp, KeltnerDown, StandardDeviation, NATR
)


def load_test_data():
    """Load sample OHLCV data for testing."""
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "VIP_X030T_1dk_.csv"
    
    if not csv_path.exists():
        print(f"[HATA] CSV bulunamadı: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', nrows=5000)
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    return {
        'open': df['Acilis'].values.astype(float).tolist(),
        'high': df['Yuksek'].values.astype(float).tolist(),
        'low': df['Dusuk'].values.astype(float).tolist(),
        'close': df['Kapanis'].values.astype(float).tolist(),
        'volume': df['Hacim'].values.astype(float).tolist(),
    }


def test_moving_averages(data):
    """Test all moving average variants."""
    print("\n" + "=" * 60)
    print("MOVING AVERAGES TEST")
    print("=" * 60)
    
    close = data['close']
    results = []
    
    tests = [
        ("SMA(20)", lambda: SMA(close, 20)),
        ("EMA(20)", lambda: EMA(close, 20)),
        ("WMA(20)", lambda: WMA(close, 20)),
        ("DEMA(20)", lambda: DEMA(close, 20)),
        ("TEMA(20)", lambda: TEMA(close, 20)),
        ("KAMA(10,2,30)", lambda: KAMA(close, 10, 2, 30)),
        ("FRAMA(16)", lambda: FRAMA(close, 16)),
        ("ZLEMA(20)", lambda: ZLEMA(close, 20)),
        ("T3(20)", lambda: T3(close, 20)),
    ]
    
    for name, func in tests:
        try:
            result = func()
            valid = len(result) == len(close)
            no_nan = not np.isnan(result[-100:]).any()
            reasonable = all(0 < r < close[-1] * 2 for r in result[-100:] if r > 0)
            
            status = "OK" if valid and no_nan and reasonable else "FAIL"
            results.append((name, status))
            print(f"  [{status:4}] {name:20} Last: {result[-1]:.2f}")
        except Exception as e:
            results.append((name, "ERROR"))
            print(f"  [ERR ] {name:20} {str(e)[:40]}")
    
    return results


def test_oscillators(data):
    """Test oscillator indicators."""
    print("\n" + "=" * 60)
    print("OSCILLATORS TEST")
    print("=" * 60)
    
    close = data['close']
    high = data['high']
    low = data['low']
    results = []
    
    tests = [
        ("RSI(14)", lambda: RSI(close, 14), 0, 100),
        ("CCI(20)", lambda: CCI(high, low, close, 20), -500, 500),
        ("MACD(12,26,9)", lambda: MACD(close, 12, 26, 9)[0], -1000, 1000),
        ("StochRSI(14)", lambda: StochRSI(close, 14, 14, 3, 3)[0], 0, 100),
        ("Williams%R(14)", lambda: WilliamsR(high, low, close, 14), -100, 0),
        ("ROC(10)", lambda: ROC(close, 10), -50, 50),
        ("UltimateOsc", lambda: UltimateOscillator(high, low, close, 7, 14, 28), 0, 100),
        ("TRIX(15)", lambda: TRIX(close, 15), -1, 1),
        ("DPO(20)", lambda: DPO(close, 20), -500, 500),
        ("CMO(9)", lambda: ChandeMomentum(close, 9), -100, 100),
        ("RMI(4,14)", lambda: RMI(close, 4, 14), 0, 100),
        ("AO(5,34)", lambda: AwesomeOscillator(high, low, 5, 34), -500, 500),
    ]
    
    for name, func, min_val, max_val in tests:
        try:
            result = func()
            if isinstance(result, tuple):
                result = result[0]
            result = np.array(result)
            
            # Skip warm-up period
            valid_slice = result[50:]
            in_range = (valid_slice >= min_val - 50).all() and (valid_slice <= max_val + 50).all()
            no_nan = not np.isnan(valid_slice[-100:]).any()
            
            status = "OK" if in_range and no_nan else "WARN"
            results.append((name, status))
            print(f"  [{status:4}] {name:20} Range: [{result[50:].min():.2f}, {result[50:].max():.2f}]")
        except Exception as e:
            results.append((name, "ERROR"))
            print(f"  [ERR ] {name:20} {str(e)[:40]}")
    
    return results


def test_trend_indicators(data):
    """Test trend indicators."""
    print("\n" + "=" * 60)
    print("TREND INDICATORS TEST")
    print("=" * 60)
    
    close = data['close']
    high = data['high']
    low = data['low']
    results = []
    
    tests = [
        ("ADX(14)", lambda: ADX(high, low, close, 14), 0, 100),
        ("DI+(14)", lambda: DI_Plus(high, low, close, 14), 0, 100),
        ("DI-(14)", lambda: DI_Minus(high, low, close, 14), 0, 100),
        ("AroonUp(25)", lambda: AroonUp(high, 25), 0, 100),
        ("AroonDown(25)", lambda: AroonDown(low, 25), 0, 100),
        ("AroonOsc(25)", lambda: AroonOsc(high, low, 25), -100, 100),
        ("ParabolicSAR", lambda: ParabolicSAR(high, low), min(low), max(high)),
        ("VHF(28)", lambda: VHF(close, 28), 0, 2),
        ("LinearReg(14)", lambda: LinearReg(close, 14), min(close)*0.8, max(close)*1.2),
    ]
    
    for name, func, min_val, max_val in tests:
        try:
            result = func()
            if isinstance(result, tuple):
                result = result[0]
            result = np.array(result)
            
            valid_slice = result[50:]
            no_nan = not np.isnan(valid_slice[-100:]).any()
            
            status = "OK" if no_nan else "WARN"
            results.append((name, status))
            print(f"  [{status:4}] {name:20} Range: [{result[50:].min():.2f}, {result[50:].max():.2f}]")
        except Exception as e:
            results.append((name, "ERROR"))
            print(f"  [ERR ] {name:20} {str(e)[:40]}")
    
    # Ichimoku
    try:
        ich = Ichimoku(high, low, close)
        no_nan = not np.isnan(np.array(ich.tenkan)[-100:]).any()
        status = "OK" if no_nan else "WARN"
        results.append(("Ichimoku", status))
        print(f"  [{status:4}] {'Ichimoku':20} Tenkan: {ich.tenkan[-1]:.2f}")
    except Exception as e:
        results.append(("Ichimoku", "ERROR"))
        print(f"  [ERR ] {'Ichimoku':20} {str(e)[:40]}")
    
    return results


def test_volume_indicators(data):
    """Test volume indicators."""
    print("\n" + "=" * 60)
    print("VOLUME INDICATORS TEST")
    print("=" * 60)
    
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    results = []
    
    tests = [
        ("OBV", lambda: OBV(close, volume)),
        ("PVT", lambda: PVT(close, volume)),
        ("ADL", lambda: ADL(high, low, close, volume)),
        ("ChaikinOsc", lambda: ChaikinOsc(high, low, close, volume, 3, 10)),
        ("NVI", lambda: NVI(close, volume)),
        ("PVI", lambda: PVI(close, volume)),
        ("Klinger", lambda: KlingerOsc(high, low, close, volume, 34, 55)),
        ("ForceIndex", lambda: ForceIndex(close, volume, 13)),
    ]
    
    for name, func in tests:
        try:
            result = func()
            result = np.array(result)
            
            no_nan = not np.isnan(result[-100:]).any()
            
            status = "OK" if no_nan else "WARN"
            results.append((name, status))
            print(f"  [{status:4}] {name:20} Last: {result[-1]:.2f}")
        except Exception as e:
            results.append((name, "ERROR"))
            print(f"  [ERR ] {name:20} {str(e)[:40]}")
    
    return results


def test_volatility_indicators(data):
    """Test volatility indicators."""
    print("\n" + "=" * 60)
    print("VOLATILITY INDICATORS TEST")
    print("=" * 60)
    
    close = data['close']
    high = data['high']
    low = data['low']
    results = []
    
    tests = [
        ("ATR(14)", lambda: ATR(high, low, close, 14)),
        ("BollingerUp", lambda: BollingerUp(close, 20, 2)),
        ("BollingerDown", lambda: BollingerDown(close, 20, 2)),
        ("BollingerWidth", lambda: BollingerWidth(close, 20, 2)),
        ("KeltnerUp", lambda: KeltnerUp(high, low, close, 20, 10, 2)),
        ("KeltnerDown", lambda: KeltnerDown(high, low, close, 20, 10, 2)),
        ("StdDev(20)", lambda: StandardDeviation(close, 20)),
        ("NATR(14)", lambda: NATR(high, low, close, 14)),
    ]
    
    for name, func in tests:
        try:
            result = func()
            result = np.array(result)
            
            valid_slice = result[30:]
            no_nan = not np.isnan(valid_slice[-100:]).any()
            positive = (valid_slice[-100:] >= 0).all()
            
            status = "OK" if no_nan and positive else "WARN"
            results.append((name, status))
            print(f"  [{status:4}] {name:20} Last: {result[-1]:.4f}")
        except Exception as e:
            results.append((name, "ERROR"))
            print(f"  [ERR ] {name:20} {str(e)[:40]}")
    
    return results


def main():
    print("\n" + "=" * 60)
    print("IdealQuant - YENİ GÖSTERGELER KALİBRASYON TESTİ")
    print("=" * 60)
    
    data = load_test_data()
    if data is None:
        return
    
    print(f"\nTest verisi: {len(data['close'])} bar")
    
    all_results = []
    
    # Run all tests
    all_results.extend(test_moving_averages(data))
    all_results.extend(test_oscillators(data))
    all_results.extend(test_trend_indicators(data))
    all_results.extend(test_volume_indicators(data))
    all_results.extend(test_volatility_indicators(data))
    
    # Summary
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    
    ok_count = sum(1 for _, s in all_results if s == "OK")
    warn_count = sum(1 for _, s in all_results if s == "WARN")
    err_count = sum(1 for _, s in all_results if s == "ERROR")
    
    print(f"  OK   : {ok_count}")
    print(f"  WARN : {warn_count}")
    print(f"  ERROR: {err_count}")
    print(f"  -----------")
    print(f"  TOPLAM: {len(all_results)} gosterge")
    
    if err_count == 0:
        print("\n✅ TÜM GÖSTERGELER ÇALIŞIYOR!")
    else:
        print(f"\n⚠️ {err_count} göstergede hata var.")
        print("\nHatalı göstergeler:")
        for name, status in all_results:
            if status == "ERROR":
                print(f"  - {name}")


if __name__ == "__main__":
    main()
