"""
MFI Volume Test - Hacim vs Lot Karşılaştırması
IdealData'nın MFI hangi volume verisini kullandığını tespit et
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.core import MoneyFlowIndex

def test_mfi_volume():
    """MFI'yi Hacim ve Lot ile karşılaştır"""
    
    # Raw data
    raw_path = PROJECT_ROOT / "data" / "VIP_X030T_1dk_.csv"
    if not raw_path.exists():
        print(f"[HATA] Ham CSV yok: {raw_path}")
        return
    
    # IdealData export (eğer varsa)
    mfi_test_path = PROJECT_ROOT / "data" / "mfi_volume_test.csv"
    
    # Raw data oku
    raw_df = pd.read_csv(raw_path, sep=";", decimal=",", encoding="cp1254")
    raw_df.columns = ["Tarih", "Saat", "Acilis", "Yuksek", "Dusuk", "Kapanis", "Ortalama", "Hacim", "Lot"]
    
    h = raw_df["Yuksek"].values.astype(float)
    l = raw_df["Dusuk"].values.astype(float)
    c = raw_df["Kapanis"].values.astype(float)
    hacim = raw_df["Hacim"].values.astype(float)
    lot = raw_df["Lot"].values.astype(float)
    
    # Python MFI hesapla (Hacim ile)
    mfi_hacim = MoneyFlowIndex(h.tolist(), l.tolist(), c.tolist(), hacim.tolist(), 14)
    
    # Python MFI hesapla (Lot ile)
    mfi_lot = MoneyFlowIndex(h.tolist(), l.tolist(), c.tolist(), lot.tolist(), 14)
    
    print("\n=== MFI VOLUME TEST ===")
    print(f"Data rows: {len(raw_df)}")
    
    # İlk 100 bar karşılaştır
    print("\n--- İlk 100 Bar Karşılaştırması ---")
    for i in range(20, 30):  # 20-30 arası (MFI period=14 sonrası)
        print(f"Bar {i:3d}: MFI(Hacim)={mfi_hacim[i]:6.2f}  MFI(Lot)={mfi_lot[i]:6.2f}  "
              f"Fark={abs(mfi_hacim[i] - mfi_lot[i]):.4f}")
    
    # Fark istatistikleri
    diff = np.array([abs(mfi_hacim[i] - mfi_lot[i]) for i in range(14, len(mfi_hacim))])
    print(f"\n--- İstatistikler ---")
    print(f"Ortalama fark: {diff.mean():.4f}")
    print(f"Max fark: {diff.max():.4f}")
    print(f"Min fark: {diff.min():.4f}")
    
    # Eğer IdealData export varsa karşılaştır
    if mfi_test_path.exists():
        print(f"\n--- IdealData Export Karşılaştırması ---")
        ideal_df = pd.read_csv(mfi_test_path, sep=";", decimal=".")
        
        # Son 100 bar karşılaştır
        for i in range(len(ideal_df) - 10, len(ideal_df)):
            bar_no = ideal_df.iloc[i]["BarNo"]
            ideal_mfi = ideal_df.iloc[i]["MFI_Auto"]
            
            if bar_no < len(mfi_hacim):
                py_hacim = mfi_hacim[bar_no]
                py_lot = mfi_lot[bar_no]
                
                diff_hacim = abs(ideal_mfi - py_hacim)
                diff_lot = abs(ideal_mfi - py_lot)
                
                winner = "HACIM" if diff_hacim < diff_lot else "LOT"
                
                print(f"Bar {bar_no:4d}: IdealData={ideal_mfi:6.2f}  "
                      f"Py(Hacim)={py_hacim:6.2f} Δ={diff_hacim:.3f}  "
                      f"Py(Lot)={py_lot:6.2f} Δ={diff_lot:.3f}  → {winner}")
        
        print("\n✓ IdealData'nın hangi volume kullandığı tespit edildi!")
    else:
        print(f"\n[BİLGİ] IdealData export yok: {mfi_test_path}")
        print("Test_MFI_Volume.cs script'ini IdealData'da çalıştırın.")
    
    # Hacim vs Lot özellikleri
    print("\n--- Hacim vs Lot İstatistikleri ---")
    print(f"Hacim: Ortalama={hacim.mean():.0f}, Max={hacim.max():.0f}")
    print(f"Lot:   Ortalama={lot.mean():.0f}, Max={lot.max():.0f}")
    print(f"Hacim/Lot oranı: {(hacim.mean() / lot.mean()):.0f}x")


if __name__ == "__main__":
    test_mfi_volume()
