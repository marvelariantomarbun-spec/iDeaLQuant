
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig
from src.strategies.common import Signal

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, sep=';')
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('DateTime').reset_index(drop=True)
    return df

def load_ideal_signals(path):
    try:
        df = pd.read_csv(path, sep=';', thousands='.', decimal=',')
    except:
        df = pd.read_csv(path, sep=';')
    
    df.columns = [c.strip() for c in df.columns]
    
    # Try multiple date formats
    for fmt in ['%d.%m.%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
        try:
            df['OpenTime'] = pd.to_datetime(df['Açılış Tarihi'], format=fmt)
            break
        except:
            continue
            
    return df

def run_fast_backtest():
    # 1. Load Data
    data_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    full_df = load_data(data_path)
    
    # 2. Filter for relevant period (Dec 2024 - Jan 2025 based on user image)
    # Start slightly before for warmup
    start_date = pd.to_datetime("2024-12-01")
    mask = full_df['DateTime'] >= start_date
    df = full_df[mask].reset_index(drop=True)
    
    print(f"Filtered Data: {len(df)} rows starting from {df['DateTime'].min()}")
    
    # 3. Prepare Inputs
    opens = df['Açılış'].values.astype(float).tolist()
    highs = df['Yüksek'].values.astype(float).tolist()
    lows = df['Düşük'].values.astype(float).tolist()
    closes = df['Kapanış'].values.astype(float).tolist()
    # Typical Price
    typical = ((df['Yüksek'] + df['Düşük'] + df['Kapanış']) / 3.0).values.astype(float).tolist()
    
    # 4. Init Strategy
    st = ScoreBasedStrategy(opens, highs, lows, closes, typical)
    
    # 5. Run Loop
    trades = []
    position = "FLAT"
    entry_price = 0.0
    entry_time = None
    
    print("Starting Loop...")
    for i in range(50, len(df)):
        # Signal Check
        sig = st.get_signal(i, position)
        
        current_time = df.loc[i, 'DateTime']
        price = closes[i]
        
        # Debug critical trade: 25.12.2024 around 21:00
        # User Image Row 2: 25.12.2024 21:06 Alış 10969
        if current_time.day == 25 and current_time.month == 12 and current_time.hour == 21:
             pass # Breakpoint opportunity
             
        if sig == "LONG" and position != "LONG":
            if position == "SHORT": # Close Short
                trades.append({'Type': 'Short Exit', 'Time': current_time, 'Price': price})
            position = "LONG"
            entry_price = price
            entry_time = current_time
            trades.append({'Type': 'Long Entry', 'Time': current_time, 'Price': price})
            
        elif sig == "SHORT" and position != "SHORT":
            if position == "LONG": # Close Long
                trades.append({'Type': 'Long Exit', 'Time': current_time, 'Price': price})
            position = "SHORT"
            entry_price = price
            entry_time = current_time
            trades.append({'Type': 'Short Entry', 'Time': current_time, 'Price': price})
            
        elif sig == "FLAT" and position != "FLAT":
            trades.append({'Type': f'{position} Exit', 'Time': current_time, 'Price': price})
            position = "FLAT"
            
    # 6. Compare
    py_trades = pd.DataFrame(trades)
    print(f"\nPython Trades Found: {len(py_trades)}")
    if not py_trades.empty:
        print(py_trades.head(10))
        
    # Load Ideal Signals
    s1_path = 'd:/Projects/IdealQuant/data/ideal_signals_1_Nolu_Strateji_200000Bar.csv'
    if os.path.exists(s1_path):
        ideal_df = load_ideal_signals(s1_path)
        print(f"\nIdeal Trades Loaded: {len(ideal_df)}")
        print(ideal_df[['Açılış Tarihi', 'Açılış Fyt', 'Yön']].head(5))
    else:
        print("Ideal signals file not found.")

if __name__ == "__main__":
    run_fast_backtest()
