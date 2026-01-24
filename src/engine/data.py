"""
IdealQuant - OHLCV Data Structures and Loader
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


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
        df = pd.read_csv(filepath, sep=';', decimal=',')
        
        # Combine date and time
        df['datetime'] = pd.to_datetime(
            df['Tarih'] + ' ' + df['Saat'], 
            format='%d.%m.%Y %H:%M'
        )
        
        # Rename columns
        df = df.rename(columns={
            'Acilis': 'open',
            'Yuksek': 'high',
            'Dusuk': 'low',
            'Kapanis': 'close',
            'Hacim': 'volume'
        })
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return cls(df)


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
