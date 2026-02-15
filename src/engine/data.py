"""
IdealQuant - OHLCV Data Structures and Loader
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
from datetime import datetime, time, date

# Import holiday logic
try:
    from src.strategies.holidays import (
        is_tatil_gunu, is_arefe, vade_sonu_is_gunu
    )
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


@dataclass
class Bar:
    """Single OHLCV bar"""
    datetime: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    @property
    def typical(self) -> float:
        """Typical price (H+L+C)/3"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """Bar range (H-L)"""
        return self.high - self.low


class OHLCV:
    """OHLCV data container with IdealData-compatible access patterns"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize from DataFrame with columns: datetime, open, high, low, close, volume
        """
        self.df = df.copy()
        self._ensure_columns()
        
        # Pre-compute lists for fast access (IdealData style)
        self.O = self.df['open'].tolist()
        self.H = self.df['high'].tolist()
        self.L = self.df['low'].tolist()
        self.C = self.df['close'].tolist()
        self.V = self.df['volume'].tolist()
        self.T = ((self.df['high'] + self.df['low'] + self.df['close']) / 3).tolist()
        self.DT = self.df['datetime'].tolist()
        
    def _ensure_columns(self):
        """Ensure required columns exist"""
        required = ['datetime', 'open', 'high', 'low', 'close']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'volume' not in self.df.columns:
            self.df['volume'] = 0.0
    
    @property
    def BarSayisi(self) -> int:
        """IdealData compatible bar count"""
        return len(self.df)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> Bar:
        """Get bar by index"""
        row = self.df.iloc[idx]
        return Bar(
            datetime=row['datetime'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row.get('volume', 0.0)
        )
    
    @classmethod
    def from_csv(cls, filepath: str, 
                 datetime_col: str = 'datetime',
                 datetime_format: str = None,
                 separator: str = ',') -> 'OHLCV':
        """
        Load OHLCV data from CSV file
        
        Args:
            filepath: Path to CSV file
            datetime_col: Name of datetime column
            datetime_format: Optional datetime format string
            separator: CSV separator
            
        Returns:
            OHLCV instance
        """
        df = pd.read_csv(filepath, sep=separator)
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Parse datetime
        if datetime_format:
            df['datetime'] = pd.to_datetime(df[datetime_col.lower()], format=datetime_format)
        else:
            df['datetime'] = pd.to_datetime(df[datetime_col.lower()])
        
        # Rename columns if needed
        column_mapping = {
            'date': 'datetime',
            'time': 'datetime',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vol': 'volume',
            'acilis': 'open',
            'yuksek': 'high',
            'dusuk': 'low',
            'kapanis': 'close',
            'hacim': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return cls(df)
    
    @classmethod
    def from_ideal_export(cls, filepath: str) -> 'OHLCV':
        """
        Load from IdealData CSV export format
        Expected columns: Tarih;Saat;Acilis;Yuksek;Dusuk;Kapanis;Hacim
        """
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='cp1254')
        
        # Rename by index immediately to avoid encoding/BOM issues
        try:
            mapping = {
                df.columns[0]: 'date_str',
                df.columns[1]: 'time_str',
                df.columns[2]: 'open',
                df.columns[3]: 'high',
                df.columns[4]: 'low',
                df.columns[5]: 'close',
                df.columns[7]: 'volume'
            }
            df = df.rename(columns=mapping)
        except IndexError:
            # Fallback (maybe fewer columns?)
            pass
            
        # create datetime from renamed columns
        try:
            df['datetime'] = pd.to_datetime(
                df['date_str'] + ' ' + df['time_str'], 
                format='%d.%m.%Y %H:%M:%S'
            )
        except KeyError:
             # Fallback if rename failed, try original names if compatible
             pass
        except Exception as e:
             print(f"Date Parse Error: {e}")
             
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return cls(df)
    
    def get_trading_mask(self, vade_tipi: str = "ENDEKS") -> np.ndarray:
        """
        Generate a boolean mask for tradable bars.
        False = Do not trade / Close position (Holiday, Weekend, Expiry Afternoon, Half-day Eve)
        True = Tradable
        
        Args:
            vade_tipi: "ENDEKS" or "SPOT"
            
        Returns:
            np.ndarray (bool): Mask array of same length as data
        """
        if not HOLIDAYS_AVAILABLE:
            # Fallback if holidays module is missing
            return np.ones(len(self.df), dtype=bool)
            
        n = len(self.df)
        mask = np.ones(n, dtype=bool)
        
        # Pre-calculate unique dates to minimize function calls
        # We assume self.DT contains datetime objects
        # If not, convert once
        if len(self.DT) > 0 and not isinstance(self.DT[0], (datetime, pd.Timestamp)):
             times = pd.to_datetime(self.df['datetime']).tolist()
        else:
             times = self.DT
             
        # Extract date and time components efficiently if possible
        # For simplicity and correctness with the existing messy date formats:
        
        # Cache for expensive valid/invalid days
        valid_days = {}
        expiry_days = set()
        
        # Identify Expiry Dates
        # Get unique months to check for expiry
        # This is faster than checking every single day
        if hasattr(self.df['datetime'], 'dt'):
            unique_dates = self.df['datetime'].dt.date.unique()
        else:
            # Fallback
            unique_dates = set([t.date() for t in times])
            
        for d in unique_dates:
            # Check Expiry
            # Monthly check for candidate
            is_expiry = False
            # Optimization: vade_sonu_is_gunu is a bit heavy, call it sparsely
            # Logic: Check if d is a vade_sonu
            # We can use the helper from holidays.py which returns the date
            # But here we need to know if 'd' IS the expiry date.
            
            # Better approach: 
            # 1. Check if it is a holiday/weekend -> Mark False
            if is_tatil_gunu(d):
                valid_days[d] = False
                continue
            
            # 2. Check Expiry
            # ENDEKS: Even months only. SPOT: Every month.
            m = d.month
            if vade_tipi == "ENDEKS" and m % 2 != 0:
                pass # Not an expiry month
            else:
                 # Calculate expiry for this month and check if d matches
                 # vade_sonu_is_gunu returns the DATE of expiry for that month
                 actual_expiry = vade_sonu_is_gunu(d, vade_tipi) 
                 if d == actual_expiry:
                     expiry_days.add(d)
            
            # 3. Check Arefe (Half day)
            if is_arefe(d):
                 # Arefe is consistent, keep valid but mark specific times later
                 pass
            
            valid_days[d] = True

        # Now iterate bars and apply mask
        # Vectorized approach would be better but requires aligning with holiday logic structure
        # Loop is fine for creating mask once (it's not inner loop)
        
        time_12_00 = time(12, 0)
        time_12_30 = time(12, 30)
        time_18_15 = time(18, 15)
        
        for i in range(n):
            t_obj = times[i]
            d = t_obj.date()
            t = t_obj.time()
            
            # 1. Day Check (Weekend/Holiday)
            if not valid_days.get(d, True):
                mask[i] = False
                continue
                
            # 2. Arefe Check (Half Day)
            # Arefe: Close after 12:00 (Logic says 12:30 usually empty but let's be safe)
            if is_arefe(d):
                if t > time_12_30:
                    mask[i] = False
                    continue
            
            # 3. Expiry Check (Vade Sonu)
            # Close positions in the afternoon of expiry day
            if d in expiry_days:
                # Vade sonu günü 12:00'den sonra işlem yapma / kapat
                # IdealData logic: 12:00 - 18:15 arası FLAT
                if t >= time_12_00:
                    mask[i] = False
                    continue
                    
        return mask


def Liste(value: float, count: int = None, data: OHLCV = None) -> List[float]:
    """
    IdealData compatible Sistem.Liste() function
    Creates a list filled with a constant value
    """
    if count is None and data is not None:
        count = len(data)
    elif count is None:
        count = 0
    
    return [float(value)] * count
