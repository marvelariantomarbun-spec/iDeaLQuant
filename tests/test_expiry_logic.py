
import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.data import OHLCV
from src.strategies.holidays import vade_sonu_is_gunu

class TestExpiryLogic(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset
        # Covers: 
        # 1. Normal Day (Jan 5 2024 - Friday)
        # 2. Weekend (Jan 6-7 2024)
        # 3. Expiry Day (Feb 29 2024 - Thursday)
        
        dates = []
        
        # 1. Normal Day: Jan 5 (Friday)
        t = datetime(2024, 1, 5, 9, 30)
        while t.time() <= time(18, 15):
            dates.append(t)
            t += timedelta(minutes=60)
            
        # 2. Expiry Day: Feb 29 (Thursday)
        # Vade Sonu for Feb 2024
        t = datetime(2024, 2, 29, 9, 30)
        while t.time() <= time(18, 15):
            dates.append(t)
            t += timedelta(minutes=60)
            
        self.df = pd.DataFrame({'datetime': dates})
        # Dummy columns
        self.df['open'] = 100.0
        self.df['high'] = 101.0
        self.df['low'] = 99.0
        self.df['close'] = 100.0
        self.df['volume'] = 1000
        
        self.ohlcv = OHLCV(self.df)
        
    def test_get_trading_mask(self):
        mask = self.ohlcv.get_trading_mask("ENDEKS")
        
        # Verify Normal Day (Jan 5)
        # All bars should be True (Tradable)
        jan5_mask = mask[self.df['datetime'].dt.date == datetime(2024, 1, 5).date()]
        self.assertTrue(np.all(jan5_mask), "Normal day should be all tradable")
        
        # Verify Expiry Day (Feb 29)
        # Bars after 12:00 should be False
        feb29_mask = mask[self.df['datetime'].dt.date == datetime(2024, 2, 29).date()]
        feb29_times = self.df.loc[self.df['datetime'].dt.date == datetime(2024, 2, 29).date(), 'datetime'].dt.time
        
        print("\nExpiry Day Check (Feb 29):")
        for m, t in zip(feb29_mask, feb29_times):
            print(f"Time: {t}, Tradable: {m}")
            if t >= time(12, 0):
                self.assertFalse(m, f"Bar at {t} on Expiry Day should be False")
            else:
                self.assertTrue(m, f"Bar at {t} on Expiry Day (Morning) should be True")

if __name__ == '__main__':
    unittest.main()
