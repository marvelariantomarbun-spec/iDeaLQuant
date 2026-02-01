
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strategies.ars_trend_v2 import ARSTrendStrategyV2, Signal, StrategyConfigV2
from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig

def load_data(path):
    print(f"Loading data from {path}...")
    # VIP_X030T format: Tarih;Saat;Açılış;Yüksek;Düşük;Kapanış;Ortalama;Hacim;Lot
    df = pd.read_csv(path, sep=';')
    
    # Parse Date and Time
    df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    
    # Sort
    df = df.sort_values('DateTime').reset_index(drop=True)
    print(f"Data Loaded: {len(df)} rows. Range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"Sample Row: {df.iloc[0]}")
    return df

def load_ideal_signals(path):
    print(f"Loading signals from {path}...")
    # IdealData format: No;Yön;Lot;Açılış Tarihi;Açılış Fyt;Kapanış Tarihi;Kapanış Fyt;Kar / Zarar;Bakiye
    # Decimal separator is likely ',' based on viewing the file (e.g., 11205,00) but int values seen in snippet were 11205.
    # Let's check format again. Snippet showed: "11514". But maybe float columns have commas.
    # We'll try dynamic parsing.
    
    try:
        df = pd.read_csv(path, sep=';', thousands='.', decimal=',')
    except:
        df = pd.read_csv(path, sep=';')
        
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Parse dates
    # Format seen: 8.01.2025 20:40 (d.m.Y H:M) or dd.mm.yyyy
    try:
        df['OpenTime'] = pd.to_datetime(df['Açılış Tarihi'], format='%d.%m.%Y %H:%M')
        df['CloseTime'] = pd.to_datetime(df['Kapanış Tarihi'], format='%d.%m.%Y %H:%M')
    except:
        # Try Auto
        df['OpenTime'] = pd.to_datetime(df['Açılış Tarihi'])
        df['CloseTime'] = pd.to_datetime(df['Kapanış Tarihi'])
        
    return df

def run_strategy_backtest(strategy_obj, df, strategy_name="Strategy"):
    print(f"Running backtest for {strategy_name}...")
    
    signals = []
    trades = []
    
    position = "FLAT"
    entry_price = 0.0
    entry_index = 0
    entry_time = None
    
    extreme_price = 0.0 # For trailing stop/SAR logic in strategy
    
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    # Warmup
    start_index = 50 # Minimum warmup
    
    # Strategy specific state
    is_v2 = isinstance(strategy_obj, ARSTrendStrategyV2)
    
    for i in range(start_index, len(df)):
        if i % 50000 == 0: print(f"Processing bar {i}/{len(df)} - Date: {df.loc[i, 'DateTime']}")
        
        # Debug: Targeted Trace for Strategy 1 Reference Trade
        current_time = df.loc[i, 'DateTime']
        
        # Check Strategy 1 Reference: 8.01.2025 20:40
        if current_time.month == 1 and current_time.day == 8 and current_time.hour == 20 and current_time.minute == 40 and current_time.year == 2025:
            print(f"\n--- DEBUG TARGET {current_time} ---")
            print(f"Price: O={df.loc[i,'Open']} H={df.loc[i,'High']} L={df.loc[i,'Low']} C={df.loc[i,'Close']}")
            
            if not is_v2:
                # Access Strategy 1 internals
                idx = i
                print(f"ARS: {strategy_obj.ars[idx]}")
                print(f"Yatay Filtre: {strategy_obj.yatay_filtre[idx]}")
                print(f"Scores:")
                
                # Re-calc scores manually to see components
                cfg = strategy_obj.config
                long_score = 0
                short_score = 0
                
                # 1. Price vs ARS
                if strategy_obj.closes[idx] > strategy_obj.ars[idx]: 
                    print("  Vote: Close > ARS (+1 Long)")
                    long_score += 1
                elif strategy_obj.closes[idx] < strategy_obj.ars[idx]: 
                    print("  Vote: Close < ARS (+1 Short)")
                    short_score += 1
                
                # 2. MACD-V (Replaces QQE)
                if strategy_obj.macdv[idx] > strategy_obj.macdv_sig[idx]: 
                    print(f"  Vote: MACD-V({strategy_obj.macdv[idx]:.2f}) > Sig({strategy_obj.macdv_sig[idx]:.2f}) (+1 Long)")
                    long_score += 1
                elif strategy_obj.macdv[idx] < strategy_obj.macdv_sig[idx]: 
                    print(f"  Vote: MACD-V({strategy_obj.macdv[idx]:.2f}) < Sig({strategy_obj.macdv_sig[idx]:.2f}) (+1 Short)")
                    short_score += 1
                
                # 3. NetLot
                if strategy_obj.netlot_ma[idx] > cfg.netlot_threshold: 
                    print(f"  Vote: NetLot({strategy_obj.netlot_ma[idx]:.2f}) > {cfg.netlot_threshold} (+1 Long)")
                    long_score += 1
                elif strategy_obj.netlot_ma[idx] < -cfg.netlot_threshold: 
                    print(f"  Vote: NetLot({strategy_obj.netlot_ma[idx]:.2f}) < -{cfg.netlot_threshold} (+1 Short)")
                    short_score += 1
                
                # 4. ADX (Votes for both if strong trend)
                if strategy_obj.adx[idx] > 25.0: # Hardcoded 25.0 in strategy logic
                    print(f"  Vote: ADX({strategy_obj.adx[idx]:.2f}) > 25.0 (Both +1)")
                    long_score += 1
                    short_score += 1
                
                print(f"Total Long Score: {long_score} (Need {cfg.min_score})")
                print(f"Total Short Score: {short_score}") 
                if hasattr(strategy_obj, 'yatay_filtre'):
                     print(f"Yatay Filtre: {strategy_obj.yatay_filtre[idx]} (Need 1)")
                
                sig = strategy_obj.get_signal(i, position)
                print(f"RESULT SIGNAL: {sig}")

        # Update state for trailing logic
        if position == "LONG":
            extreme_price = max(extreme_price, highs[i])
        elif position == "SHORT":
            extreme_price = min(extreme_price, lows[i])
            
        # Get Signal
        if is_v2:
            sig = strategy_obj.get_signal(i, position, entry_price, extreme_price)
        else:
            # ScoreBasedStrategy also expects (i, position, entry_price, extreme_price) signature based on file view
            sig = strategy_obj.get_signal(i, position, entry_price, extreme_price)
            
        # Process Signal
        # IdealData execution assumes Close price of signal bar or Open of next?
        # Usually IdealData SystemTester executes on "Close" of the bar where conditions are met, 
        # specifically if Sequential. If not, it might be Next Open.
        # Looking at Ideal logs:
        # Row 2: Open 8.01.2025 20:40 Price 11514. 
        # This matches the bar time. So execution is likely on Close or "On Update" treated as Close.
        # Let's assume execution price is Close[i].
        
        exec_price = closes[i]
        timestamp = df.loc[i, 'DateTime']
        
        if sig == "LONG" and position != "LONG":
            # Close Short if exists
            if position == "SHORT":
                trades.append({
                    'Type': 'Short Exit',
                    'ExitTime': timestamp,
                    'ExitPrice': exec_price,
                    'PnL': (entry_price - exec_price), # Simple PnL
                    'EntryTime': entry_time,
                    'EntryPrice': entry_price
                })
            
            # Open Long
            position = "LONG"
            entry_price = exec_price
            entry_index = i
            entry_time = timestamp
            extreme_price = exec_price # Reset extreme
            
        elif sig == "SHORT" and position != "SHORT":
            # Close Long if exists
            if position == "LONG":
                trades.append({
                    'Type': 'Long Exit',
                    'ExitTime': timestamp,
                    'ExitPrice': exec_price,
                    'PnL': (exec_price - entry_price),
                    'EntryTime': entry_time,
                    'EntryPrice': entry_price
                })
                
            # Open Short
            position = "SHORT"
            entry_price = exec_price
            entry_index = i
            entry_time = timestamp
            extreme_price = exec_price
            
        elif sig == "FLAT" and position != "FLAT":
            pnl = (exec_price - entry_price) if position == "LONG" else (entry_price - exec_price)
            trades.append({
                'Type': f'{position.capitalize()} Exit',
                'ExitTime': timestamp,
                'ExitPrice': exec_price,
                'PnL': pnl,
                'EntryTime': entry_time,
                'EntryPrice': entry_price
            })
            position = "FLAT"
            entry_price = 0.0
            
    return pd.DataFrame(trades)

def compare_results(py_trades, ideal_trades, strategy_name):
    print(f"\n--- Comparison for {strategy_name} ---")
    
    # Filter Ideal Trades by date range available in Py Trades
    if py_trades.empty:
        print("Python backtest produced NO trades!")
        return
        
    start_date = py_trades['EntryTime'].min()
    end_date = py_trades['ExitTime'].max()
    
    print(f"Time Range: {start_date} to {end_date}")
    
    # Filter ideal trades within this range purely for count comparison
    # Ideal trades 'OpenTime'
    mask = (ideal_trades['OpenTime'] >= start_date) & (ideal_trades['OpenTime'] <= end_date)
    ideal_subset = ideal_trades[mask]
    
    print(f"Total Trades (Python): {len(py_trades)}")
    print(f"Total Trades (Ideal) : {len(ideal_subset)} (approx match in range)")
    
    print(f"Net PnL (Python): {py_trades['PnL'].sum():.2f}")
    # Ideal PnL might be in 'Kar / Zarar' column
    try:
        ideal_pnl = ideal_subset['Kar / Zarar'].sum()
        print(f"Net PnL (Ideal) : {ideal_pnl:.2f}")
    except:
        print("Could not calc Ideal PnL")

    # Entry Logic Match Check
    # Check first 5 trades match exactly
    print("\nFirst 5 Python Trades:")
    print(py_trades[['EntryTime', 'EntryPrice', 'ExitTime', 'ExitPrice', 'PnL']].head(5))
    
    print("\nFirst 5 Ideal Trades (in range):")
    cols = ['Açılış Tarihi', 'Açılış Fyt', 'Kapanış Tarihi', 'Kapanış Fyt', 'Kar / Zarar']
    print(ideal_subset[cols].head(5))

def main():
    # Load Data
    # Assuming standard path
    data_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    df = load_data(data_path)
    
    # Convert cols to float/list for strategy
    opens = df['Açılış'].values.astype(float).tolist()
    highs = df['Yüksek'].values.astype(float).tolist()
    lows = df['Düşük'].values.astype(float).tolist()
    closes = df['Kapanış'].values.astype(float).tolist()
    typical = ((df['Yüksek'] + df['Düşük'] + df['Kapanış']) / 3.0).values.astype(float).tolist()
    times = df['DateTime'].tolist()
    volumes = df['Lot'].values.astype(float).tolist()
    
    # Prepare DF for backtest engine (needs Close, High, Low for logic)
    # The backtest loop uses df['Close'] etc.
    df['Close'] = closes
    df['High'] = highs
    df['Low'] = lows
    df['Open'] = opens
    
    # --- STRATEGY 2 (ARS Trend v2) ---
    ars_v2_path = 'd:/Projects/IdealQuant/data/ideal_signals_ars_v2.csv'
    if os.path.exists(ars_v2_path):
        ideal_v2 = load_ideal_signals(ars_v2_path)
        
        # Instantiate V2 Strategy
        # Params need to match IdealData defaults
        # ARS v2 defaults in python: ema=3, atr=10, mult=0.5
        # Ideal usage likely matches defaults or we need to find "2_Nolu_Strateji.txt" content
        # Assuming defaults based on python file
        st2 = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, volumes)
        
        res_v2 = run_strategy_backtest(st2, df, "ARS Trend v2")
        compare_results(res_v2, ideal_v2, "ARS Trend v2")
    else:
        print(f"Signals file not found: {ars_v2_path}")

    # --- STRATEGY 1 (ScoreBased) ---
    s1_path = 'd:/Projects/IdealQuant/data/ideal_signals_1_Nolu_Strateji_200000Bar.csv'
    if os.path.exists(s1_path):
        ideal_s1 = load_ideal_signals(s1_path)
        
        # ScoreBasedStrategy
        # Params: opens, highs, lows, closes, typical
        st1 = ScoreBasedStrategy(opens, highs, lows, closes, typical)
        
        res_s1 = run_strategy_backtest(st1, df, "Score Based Strategy")
        compare_results(res_s1, ideal_s1, "Score Based Strategy")
    else:
        print(f"Signals file not found: {s1_path}")

if __name__ == "__main__":
    main()
