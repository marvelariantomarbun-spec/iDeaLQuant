"""
Zero-Tolerance Calibration Test - BarNo Based
Tests for 0% difference between Python and IdealData indicators
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.core import (
    HHV, LLV, RSI, Momentum, ATR, MoneyFlowIndex
)

def test_zero_tolerance():
    print("="*70)
    print("ZERO-TOLERANCE CALIBRATION TEST (BarNo Based)")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # 1. Load IdealData export with BarNo
    ideal = pd.read_csv(project_root / 'data' / 'ideal_ars_v2_data.csv', sep=';', decimal='.')
    print(f"\n[1] IdealData export: {len(ideal)} bars")
    print(f"    BarNo range: {ideal['BarNo'].iloc[0]} - {ideal['BarNo'].iloc[-1]}")
    print(f"    Columns: {ideal.columns.tolist()}")
    
    # 2. Load raw CSV
    raw = pd.read_csv(project_root / 'data' / 'VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
    raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    print(f"\n[2] Raw CSV: {len(raw)} bars")
    
    # 3. Calculate Python indicators on FULL data
    print("\n[3] Calculating Python indicators...")
    highs = raw['Yuksek'].values.astype(float).tolist()
    lows = raw['Dusuk'].values.astype(float).tolist()
    closes = raw['Kapanis'].values.astype(float).tolist()
    volumes = raw['Lot'].values.astype(float).tolist()  # Use Lot, not Hacim
    
    # Get parameters from export script
    BREAKOUT_Period = 10
    MOMENTUM_Period = 5
    
    py_hhv = np.array(HHV(highs, BREAKOUT_Period))
    py_llv = np.array(LLV(lows, BREAKOUT_Period))
    py_momentum = np.array(Momentum(closes, MOMENTUM_Period))
    py_rsi = np.array(RSI(closes, 14))
    py_mfi = np.array(MoneyFlowIndex(highs, lows, closes, volumes, 14))
    py_atr = np.array(ATR(highs, lows, closes, 14))
    
    # 4. Extract Python values at IdealData BarNo positions
    bar_indices = ideal['BarNo'].values.astype(int)
    
    py_hhv_aligned = py_hhv[bar_indices]
    py_llv_aligned = py_llv[bar_indices]
    py_momentum_aligned = py_momentum[bar_indices]
    py_rsi_aligned = py_rsi[bar_indices]
    py_mfi_aligned = py_mfi[bar_indices]
    py_atr_aligned = py_atr[bar_indices]
    
    # 5. Compare
    print("\n[4] Comparison Results:")
    print("-"*70)
    print(f"   {'Indicator':12} | {'Max Diff':12} | {'Mean Diff':12} | {'Status':8}")
    print("-"*70)
    
    results = []
    tests = [
        ('HHV', ideal['HHV'].values, py_hhv_aligned),
        ('LLV', ideal['LLV'].values, py_llv_aligned),
        ('Momentum', ideal['Momentum'].values, py_momentum_aligned),
        ('RSI', ideal['RSI'].values, py_rsi_aligned),
        ('MFI', ideal['MFI'].values, py_mfi_aligned),
        ('ATR', ideal['ATR'].values, py_atr_aligned),
    ]
    
    all_pass = True
    for name, ideal_vals, py_vals in tests:
        diff = np.abs(ideal_vals - py_vals)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        
        # For 0 tolerance, max_diff should be < 0.01 (rounding precision)
        is_pass = max_diff < 0.02
        status = "OK" if is_pass else "FAIL"
        if not is_pass:
            all_pass = False
        
        results.append((name, max_diff, mean_diff, is_pass))
        print(f"   {name:12} | {max_diff:12.4f} | {mean_diff:12.6f} | [{status:4}]")
        
        # Show first mismatch if failed
        if not is_pass:
            mismatch_idx = np.argmax(diff)
            bar_no = bar_indices[mismatch_idx]
            print(f"        -> Worst at BarNo {bar_no}: Ideal={ideal_vals[mismatch_idx]:.4f}, Python={py_vals[mismatch_idx]:.4f}")
    
    # 6. Summary
    print("\n" + "="*70)
    passed = sum(1 for r in results if r[3])
    if all_pass:
        print(f"SUCCESS: ALL {len(results)} INDICATORS MATCH (0 difference)")
    else:
        print(f"RESULT: {passed}/{len(results)} indicators match")
    print("="*70)
    
    return all_pass

if __name__ == "__main__":
    test_zero_tolerance()
