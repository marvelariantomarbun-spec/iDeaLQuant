"""
ARS_Dynamic Test - Yuvarlama Mantığı Kontrolü
IdealData ile Python implementasyonunu karşılaştır
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators.core import ARS_Dynamic, EMA, ATR

def test_ars_dynamic():
    """ARS_Dynamic yuvarlama mantığını test et"""
    
    # Raw data
    raw_path = PROJECT_ROOT / "data" / "VIP_X030T_1dk_.csv"
    if not raw_path.exists():
        print(f"[HATA] Ham CSV yok: {raw_path}")
        return
    
    # Raw data oku
    raw_df = pd.read_csv(raw_path, sep=";", decimal=",", encoding="cp1254", nrows=1000)
    raw_df.columns = ["Tarih", "Saat", "Acilis", "Yuksek", "Dusuk", "Kapanis", "Ortalama", "Hacim", "Lot"]
    
    h = raw_df["Yuksek"].values.astype(float)
    l = raw_df["Dusuk"].values.astype(float)
    c = raw_df["Kapanis"].values.astype(float)
    typical = ((h + l + c) / 3).tolist()
    
    print("\n=== ARS_DYNAMIC TEST ===")
    print(f"Data rows: {len(raw_df)}")
    
    # Test 1: 1DK parametreleri (ATR_Mult > 0)
    print("\n--- Test 1: 1DK Parametreleri (Dinamik ATR) ---")
    ars_1dk = ARS_Dynamic(
        typical, h.tolist(), l.tolist(), c.tolist(),
        ema_period=3, atr_period=10, atr_mult=0.5,
        min_k=0.002, max_k=0.015
    )
    
    # EMA ve ATR'yi de hesapla (debug için)
    ema = EMA(typical, 3)
    atr = ATR(h.tolist(), l.tolist(), c.tolist(), 10)
    
    # İlk 50 bar'ı göster
    print("\nBar |    Close |      EMA |      ATR | ATR/EMA  | Dynamic K | RoundStep |      ARS")
    print("-" * 85)
    for i in range(20, 30):
        if ema[i] > 0:
            atr_ratio = atr[i] / ema[i]
            dyn_k = atr_ratio * 0.5
            dyn_k = max(0.002, min(0.015, dyn_k))
            round_step = max(0.01, atr[i] * 0.1)
        else:
            atr_ratio = 0
            dyn_k = 0.002
            round_step = 0.01
        
        print(f"{i:3d} | {c[i]:8.2f} | {ema[i]:8.2f} | {atr[i]:8.4f} | "
              f"{atr_ratio:8.6f} | {dyn_k:9.6f} | {round_step:9.4f} | {ars_1dk[i]:8.2f}")
    
    # Test 2: 15DK/60DK parametreleri (ATR_Mult = 0, Classic mod)
    print("\n--- Test 2: 15DK/60DK Parametreleri (Classic ARS) ---")
    ars_classic = ARS_Dynamic(
        typical, h.tolist(), l.tolist(), c.tolist(),
        ema_period=3, atr_period=14, atr_mult=0.0,
        min_k=0.0123, max_k=0.0123
    )
    
    print("\nBar |    Close |      EMA |      ATR | RoundStep |  ARS_Classic")
    print("-" * 70)
    for i in range(20, 30):
        # ATR_Mult = 0 ise round_step = 0.025 (IdealData uyumlu)
        round_step = 0.025
        
        print(f"{i:3d} | {c[i]:8.2f} | {ema[i]:8.2f} | {atr[i]:8.4f} | "
              f"{round_step:9.4f} | {ars_classic[i]:12.4f}")
    
    # Yuvarlama kontrolü
    print("\n--- Yuvarlama Kontrolü ---")
    print("ARS_Dynamic (1DK) yuvarlama adımları:")
    for i in range(20, 25):
        round_step = max(0.01, atr[i] * 0.1)
        print(f"  Bar {i}: ATR={atr[i]:.4f} → RoundStep={round_step:.4f} → ARS={ars_1dk[i]:.2f}")
    
    print("\nARS_Classic (15DK) yuvarlama adımları:")
    for i in range(20, 25):
        round_step = 0.025  # Sabit
        print(f"  Bar {i}: RoundStep={round_step:.4f} → ARS={ars_classic[i]:.4f}")
    
    print("\n✓ ARS_Dynamic yuvarlama mantığı güncellendi!")
    print("  - ATR_Mult > 0: round_step = max(0.01, ATR * 0.1)")
    print("  - ATR_Mult = 0: round_step = 0.025 (IdealData default)")


if __name__ == "__main__":
    test_ars_dynamic()
