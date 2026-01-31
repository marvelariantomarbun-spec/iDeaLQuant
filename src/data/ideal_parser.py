# -*- coding: utf-8 -*-
"""
IdealData Binary Parser
-----------------------
IdealData .01 dosyalarını okur ve pandas DataFrame'e çevirir.

Format (32 byte per record):
- int32: dakika sayısı (base date'den itibaren)
- float32: Open
- float32: High  
- float32: Low
- float32: Close
- float32: Volume (Lot)
- float32: Amount (Hacim TL)
- int32: Flags

Base Date: 1988-02-28
"""

import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd


BASE_DATE = datetime(1988, 2, 28)
RECORD_SIZE = 32


def read_ideal_data(file_path: str) -> pd.DataFrame:
    """
    IdealData .01 dosyasını okur.
    
    Args:
        file_path: .01 dosya yolu
        
    Returns:
        pandas DataFrame with columns: Date, Open, High, Low, Close, Volume, Amount, Flags
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    records = []
    
    with open(path, 'rb') as f:
        data = f.read()
    
    num_records = len(data) // RECORD_SIZE
    
    for i in range(num_records):
        offset = i * RECORD_SIZE
        chunk = data[offset:offset + RECORD_SIZE]
        
        if len(chunk) < RECORD_SIZE:
            break
            
        try:
            # Little-endian: int32, float32 x6, int32
            time_minutes, o, h, l, c, volume, amount, flags = struct.unpack('<i6fi', chunk)
            
            # Calculate date from minutes since base date
            bar_date = BASE_DATE + timedelta(minutes=time_minutes)
            
            records.append({
                'Date': bar_date,
                'Open': o,
                'High': h,
                'Low': l,
                'Close': c,
                'Volume': volume,  # Lot
                'Amount': amount,  # TL
                'Flags': flags
            })
        except struct.error:
            continue
    
    df = pd.DataFrame(records)
    return df


def list_ideal_symbols(chart_data_path: str, market: str = "VIP", period: str = "01") -> List[str]:
    """
    ChartData klasöründeki sembolleri listeler.
    
    Args:
        chart_data_path: ChartData klasör yolu (örn: D:\\iDeal\\ChartData)
        market: Pazar (VIP, IMKBH, FX vb.)
        period: Periyot (01=1dk, 05=5dk, G=günlük vb.)
        
    Returns:
        Sembol listesi
    """
    path = Path(chart_data_path) / market / period
    if not path.exists():
        return []
    
    symbols = []
    for f in path.glob("*.01"):
        # VIP'VIP-X030.01 -> X030
        name = f.stem
        if "'" in name:
            parts = name.split("'")
            if len(parts) >= 2:
                symbol = parts[1].replace("VIP-", "").replace("F_", "")
                symbols.append(symbol)
    
    return sorted(set(symbols))


if __name__ == "__main__":
    # Test
    import sys
    
    # Test file path
    test_file = r"D:\iDeal\ChartData\VIP\01\VIP'VIP-X030.01"
    
    print(f"Reading: {test_file}")
    df = read_ideal_data(test_file)
    
    print(f"\nTotal bars: {len(df)}")
    print(f"\nDate range: {df['Date'].min()} -> {df['Date'].max()}")
    print(f"\nFirst 5 bars:")
    print(df.head())
    print(f"\nLast 5 bars:")
    print(df.tail())
    
    # Test symbol listing
    print(f"\n\nAvailable symbols:")
    symbols = list_ideal_symbols(r"D:\iDeal\ChartData")
    print(symbols[:20])
