
import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Strategies
try:
    from src.optimization.strategy1_optimizer import fast_backtest_score
    from src.optimization.strategy2_optimizer import fast_backtest_strategy2
    from src.optimization.strategy3_optimizer import fast_backtest_paradise
    from src.optimization.strategy4_optimizer import fast_backtest_strategy4
    print("Imports Successful")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        # Create Dummy Data
        self.n = 200
        self.closes = np.random.uniform(100, 110, self.n).astype(np.float64)
        self.opens = self.closes * 1.0
        self.highs = self.closes * 1.01
        self.lows = self.closes * 0.99
        self.volumes = np.random.uniform(1000, 5000, self.n).astype(np.float64)
        
        # Indicator Arrays (Dummy)
        self.ars_arr = self.closes.copy()
        self.hhv = self.highs.copy()
        self.llv = self.lows.copy()
        self.mom = np.random.uniform(98, 102, self.n)
        self.rsi = np.random.uniform(30, 70, self.n)
        self.mfi = np.random.uniform(20, 80, self.n)
        self.adx = np.random.uniform(10, 40, self.n)
        self.macd = np.random.uniform(-1, 1, self.n)
        self.sig = np.random.uniform(-1, 1, self.n)
        self.netlot = np.random.uniform(-50, 50, self.n)
        self.atr = np.random.uniform(0.5, 2.0, self.n)
        self.ema = self.closes.copy()
        self.dsma = self.closes * 1.001
        self.sma = self.closes * 0.999
        
        # TOMA arrays
        self.toma_trend = np.ones(self.n, dtype=np.int32)
        self.trix = np.zeros(self.n)
        
        # MASK (All True)
        self.mask = np.ones(self.n, dtype=bool)
        
    def test_s1_score(self):
        print("\nTesting S1 (Score)...")
        # fast_backtest_score arguments found in file:
        # closes, ars_arr, adx_arr, macd_arr, sig_arr, netlot_ma_arr,
        # bb_u, bb_m, bb_l, mask_arr, ... params
        
        bb_u = self.closes * 1.02
        bb_m = self.closes
        bb_l = self.closes * 0.98
        
        res = fast_backtest_score(
            self.closes, self.ars_arr, self.adx, self.macd, self.sig, self.netlot,
            bb_u, bb_m, bb_l, self.mask,
            3, 3, 2, # scores
            25.0, 0.0, 20.0, # thr
            20.0, 2, 0.8, 0.25, 10
        )
        print(f"S1 Result: {res}")
        self.assertTrue(len(res) == 4)

    def test_s2_ars(self):
        print("\nTesting S2 (ARS)...")
        # fast_backtest_strategy2 args:
        # closes, highs, lows, volumes, ars_arr, hhv, llv, mom, rsi,
        # mfi_arr, mfi_hhv, mfi_llv, vol_hhv, mask_arr, ...
        
        res = fast_backtest_strategy2(
            self.closes, self.highs, self.lows, self.volumes,
            self.ars_arr, self.hhv, self.llv, self.mom, self.rsi,
            self.mfi, self.mfi, self.mfi, self.volumes, self.mask,
            5, 10, 14, 2.0, 1.0, # params
            70, 30, True, True
        )
        print(f"S2 Result: {res}")
        self.assertTrue(len(res) == 4)

    def test_s3_paradise(self):
        print("\nTesting S3 (Paradise)...")
        # fast_backtest_paradise args:
        # closes, highs, lows, volumes, ema, dsma, sma, mom, hh, ll, atr, vol_hhv, mask
        
        res = fast_backtest_paradise(
            self.closes, self.highs, self.lows, self.volumes,
            self.ema, self.dsma, self.sma, self.mom,
            self.hhv, self.llv, self.atr, self.volumes,
            self.mask,
            98.0, 102.0, 2.0, 4.0, 2.5, True
        )
        print(f"S3 Result: {res}")
        self.assertTrue(len(res) == 4)

    def test_s4_toma(self):
        print("\nTesting S4 (TOMA)...")
        
        # Prepare Inputs for new signature
        toma_val = self.closes * 0.95
        
        res = fast_backtest_strategy4(
            self.closes, 
            self.toma_trend, toma_val,
            self.hhv, self.llv, # hhv1, llv1
            self.hhv, self.llv, # hhv2, llv2
            self.hhv, self.llv, # hhv3, llv3
            self.mom, self.trix,
            self.mask,
            98.0, 101.5, # limits
            110, 140,    # trix lookbacks
            0.0, 0.0     # exits
        )
        print(f"S4 Result: {res}")
        self.assertTrue(len(res) == 4)

if __name__ == '__main__':
    unittest.main()
