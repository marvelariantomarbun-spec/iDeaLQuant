"""
Volume HHV/LLV Calibration Test
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.core import HHV, LLV

def test_volume_hhv_llv():
    print("="*70)
    print("VOLUME HHV/LLV CALIBRATION TEST")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # 1. Load IdealData export
    ideal = pd.read_csv(project_root / 'data' / 'ideal_ars_v2_data.csv', sep=';', decimal='.')
    print(f"\n[1] IdealData export: {len(ideal)} bars")
    print(f"    Columns: {ideal.columns.tolist()}")
    
    if 'VolHHV' not in ideal.columns:
        print("[ERROR] VolHHV column not found in export!")
        return False
    
    # 2. Load raw CSV
    raw = pd.read_csv(project_root / 'data' / 'VIP_X030T_1dk_.csv', sep=';', decimal=',', encoding='cp1254')
    raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    print(f"\n[2] Raw CSV: {len(raw)} bars")
    
    # 3. Calculate Python Volume HHV/LLV using Lot column
    print("\n[3] Calculating Python Volume HHV/LLV...")
    lots = raw['Lot'].values.astype(float).tolist()
    
    py_vol_hhv = np.array(HHV(lots, 14))
    py_vol_llv = np.array(LLV(lots, 14))
    
    # 4. Align by BarNo
    bar_indices = ideal['BarNo'].values.astype(int)
    
    py_vol_hhv_aligned = py_vol_hhv[bar_indices]
    py_vol_llv_aligned = py_vol_llv[bar_indices]
    
    ideal_vol_hhv = ideal['VolHHV'].values
    ideal_vol_llv = ideal['VolLLV'].values
    
    # 5. Compare
    print("\n[4] Comparison Results:")
    print("-"*70)
    
    tests = [
        ('VolHHV', ideal_vol_hhv, py_vol_hhv_aligned),
        ('VolLLV', ideal_vol_llv, py_vol_llv_aligned),
    ]
    
    all_pass = True
    for name, ideal_vals, py_vals in tests:
        diff = np.abs(ideal_vals - py_vals)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        
        is_pass = max_diff < 1.0  # Integer values, should be exactly 0
        status = "OK" if is_pass else "FAIL"
        if not is_pass:
            all_pass = False
        
        print(f"   {name:12} | Max Diff: {max_diff:8.2f} | Mean Diff: {mean_diff:8.4f} | [{status}]")
        
        if not is_pass:
            mismatch_idx = np.argmax(diff)
            bar_no = bar_indices[mismatch_idx]
            print(f"        -> Worst at BarNo {bar_no}: Ideal={ideal_vals[mismatch_idx]}, Python={py_vals[mismatch_idx]}")
    
    # 6. Summary
    print("\n" + "="*70)
    if all_pass:
        print("SUCCESS: Volume HHV/LLV match IdealData!")
    else:
        print("FAIL: Volume HHV/LLV have differences")
    print("="*70)
    
    return all_pass

if __name__ == "__main__":
    test_volume_hhv_llv()
