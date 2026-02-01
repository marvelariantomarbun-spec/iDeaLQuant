
"""
IdealQuant - Full Indicator Calibration
Compares Python implementations against IdealData exported CSV.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style, init

# Init colorama
init()

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators import (
    # MA
    SMA, EMA, DEMA, TEMA, HullMA, FRAMA, KAMA,
    # Oscillators
    RSI, CCI, Momentum, ROC, ChandeMomentum, WilliamsR, 
    StochasticFast, StochasticSlow, Qstick, RVI,
    MACD, TRIX, ElliotWaveOscillator,
    # Trend
    ADX, DI_Plus, DI_Minus, AroonUp, AroonDown, AroonOsc, ParabolicSAR,
    LinearReg, LinearRegSlope, PriceChannelUp, PriceChannelDown,
    # Volatility
    ATR, BollingerUp, BollingerMid, BollingerDown, BollingerWidth,
    KeltnerUp, KeltnerDown, ChaikinVolatility,
    # Volume
    MoneyFlowIndex, OBV, PVT, ChaikinMoneyFlow, ChaikinOsc,
    ADL, EaseOfMovement, KlingerOsc, MassIndex,
    # Other
    HHV, LLV, QQEF
)

# Custom/Special implementations if needed (e.g. wrapper for multi-output)
def get_macd_line(close): return MACD(close, 12, 26, 9)[0]
def get_stoch_fast(high, low, close): return StochasticFast(high, low, close, k_period=14)
def get_stoch_slow(high, low, close): return StochasticSlow(high, low, close, k_period=14, smooth=3)
def get_rvi_main(close, high, low, open_): return RVI(opens=open_, highs=high, lows=low, closes=close, period=10)[0]
def get_rvi_signal(close, high, low, open_): return RVI(opens=open_, highs=high, lows=low, closes=close, period=10)[1]
def get_boll_up(close): return BollingerUp(close, period=20, deviation=2)
def get_boll_mid(close): return BollingerMid(close, period=20)
def get_boll_down(close): return BollingerDown(close, period=20, deviation=2)
def get_boll_width(close): return BollingerWidth(close, period=20, deviation=2)
def get_frama(high, low, close): return FRAMA(high, low, close)
def get_qqef(close): return QQEF(close, 14, 5)[0]
def get_qqes(close): return QQEF(close, 14, 5)[1]

def check_close(series1, series2, name, tolerance=1e-4):
    """Compare two congruous series."""
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    # Trim to valid length (indicators have warmup)
    valid_mask = ~np.isnan(s1) & ~np.isnan(s2)
    # Skip first 100 bars for warmup stability
    valid_mask[:100] = False
    
    if not np.any(valid_mask):
        print(f"{Fore.RED}[FAIL] {name}: No valid overlapping data points.{Style.RESET_ALL}")
        return False

    v1 = s1[valid_mask]
    v2 = s2[valid_mask]
    
    diff = np.abs(v1 - v2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Correlation
    if len(v1) > 1 and np.std(v1) > 1e-9 and np.std(v2) > 1e-9:
        corr = np.corrcoef(v1, v2)[0, 1]
    else:
        corr = 0.0
        
    is_ok = max_diff < tolerance
    
    status_color = Fore.GREEN if is_ok else Fore.RED
    status_text = "PASS" if is_ok else "FAIL"
    
    print(f"{status_color}[{status_text}] {name:<20} | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f} | Corr: {corr:.4f}{Style.RESET_ALL}")
    
    if not is_ok:
        # Show sample mismatch
        idx = np.argmax(diff)
        print(f"    Sample Mismatch at index {idx} (Original {np.where(valid_mask)[0][idx]}): Ideal={v1[idx]:.4f} vs Py={v2[idx]:.4f}")
        
    return is_ok

def main():
    csv_path = Path(__file__).parent.parent / "data" / "ideal_ind_export.csv"
    if not csv_path.exists():
        print("Export file not found!")
        return

    print("Loading data...")
    # Load with ; separator and . decimal (as per our export script)
    df = pd.read_csv(csv_path, sep=';', decimal='.')
    
    # Map basic data
    # IdealData Export Headers: 
    # BarNo;Date;Time;Close;Hacim;Lot;...
    
    # Ensure numerical types
    closes = df['Close'].values.astype(float)
    volumes = df['Hacim'].values.astype(float)
    # Ideally High/Low/Open should be exported too for full calib (ATR, etc need them)
    # Wait, the export script ONLY exported Close, Hacim, Lot...
    # BUT many indicators need High/Low (ATR, Stochastic, etc.)
    # ERROR IN EXPORT SCRIPT: The script exported 'Close' but mostly used 'C' in calculations.
    # However, Python functions NEED High/Low/Open arrays.
    # We must load the SOURCE data (VIP_X030T_1dk_.csv) and sync it, OR
    # UPDATE THE EXPORT SCRIPT TO INCLUDE O/H/L. 
    
    # CRITICAL CHECK: The export script `ideal_ind_export.csv` has `Close`, `Hacim`, `Lot`.
    # It DOES NOT have Open, High, Low.
    # Most indicators (ATR, Stochastic, Ichimoku, ADX) WILL FAIL or be approximate if we use Close instead of H/L.
    
    # Let's check if we can align with `VIP_X030T_1dk_.csv`.
    # IdealData export has Date/Time. We can merge.
    
    source_df = pd.read_csv(Path(__file__).parent.parent / "data" / "VIP_X030T_1dk_.csv", 
                           sep=';', decimal=',', encoding='cp1254', header=0,
                           names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'WAP', 'Vol', 'Lot'],
                           dtype={'Date': str, 'Time': str})
    
    # Align simply by length or Date/Time? 
    # IdealData export usually processes the SAME file.
    # Let's try to match by tail.
    
    print(f"Export rows: {len(df)}")
    print(f"Source rows: {len(source_df)}")
    
    # Create valid OHL arrays
    # We will slice source_df to match df based on Date/Time
    
    # Fix Time format in source (remove :00 seconds)
    source_df['Time'] = source_df['Time'].apply(lambda x: x[:5] if isinstance(x, str) and len(x) >= 5 else x)

    # Create composite key
    df['DateTime'] = df['Date'] + ' ' + df['Time']
    source_df['DateTime'] = source_df['Date'] + ' ' + source_df['Time']
    
    # Merge to get O/H/L onto the Export DF
    merged = pd.merge(df, source_df[['DateTime', 'Open', 'High', 'Low']], on='DateTime', how='inner')
    
    print(f"Aligned rows: {len(merged)}")
    
    if len(merged) < 100:
        print("CRITICAL: Alignment failed. Not enough matching rows.")
        return

    # Now use Merged data for calculation
    O = merged['Open'].values
    H = merged['High'].values
    L = merged['Low'].values
    C = merged['Close'].values # Exported Close
    V = merged['Hacim'].values
    
    # --- COMPARISONS ---
    print("\n--- Moving Averages ---")
    check_close(merged['SMA20'], SMA(C, 20), "SMA(20)")
    check_close(merged['EMA20'], EMA(C, 20), "EMA(20)")
    check_close(merged['DEMA20'], DEMA(C, 20), "DEMA(20)")
    check_close(merged['TEMA20'], TEMA(C, 20), "TEMA(20)")
    check_close(merged['HullMA20'], HullMA(C, 20), "HullMA(20)")
    check_close(merged['FRAMA'], get_frama(H, L, C), "FRAMA") # Changed to use get_frama
    check_close(merged['KAMA1030'], KAMA(C, 10, 2, 30), "KAMA")

    print("\n--- Oscillators ---")
    check_close(merged['RSI14'], RSI(C, 14), "RSI(14)")
    check_close(merged['CCI20'], CCI(H, L, C, 20), "CCI(20)")
    check_close(merged['Momentum10'], Momentum(C, 10), "Momentum(10)")
    check_close(merged['ROC10'], ROC(C, 10), "ROC(10)")
    check_close(merged['CMO9'], ChandeMomentum(C, 9), "CMO(9)")
    check_close(merged['WilliamsR14'], WilliamsR(H, L, C, 14), "S.WilliamsR(14)")
    check_close(merged['StochFast'], get_stoch_fast(H, L, C), "StochFast")
    check_close(merged['StochSlow'], get_stoch_slow(H, L, C), "StochSlow")
    check_close(merged['Qstick10'], Qstick(O, C, 10), "Qstick(10)")
    check_close(merged['RVI10'], get_rvi_main(C, H, L, O), "RVI(10)")
    check_close(merged['RVI10Sig'], get_rvi_signal(C, H, L, O), "RVI(10) Sig")
    check_close(merged['MACD1226'], get_macd_line(C), "MACD(12,26)")
    check_close(merged['TRIX15'], TRIX(C, 15), "TRIX(15)")

    print("\n--- Trend ---")
    check_close(merged['HHV20'], HHV(H, 20), "HHV(20)")
    check_close(merged['LLV20'], LLV(L, 20), "LLV(20)")
    check_close(merged['ADX14'], ADX(H, L, C, 14), "ADX(14)")
    check_close(merged['DIPlus14'], DI_Plus(H, L, C, 14), "DI+(14)")
    check_close(merged['DIMinus14'], DI_Minus(H, L, C, 14), "DI-(14)")
    check_close(merged['AroonUp14'], AroonUp(H, 14), "AroonUp(14)")
    check_close(merged['AroonDown14'], AroonDown(L, 14), "AroonDown(14)")
    check_close(merged['AroonOsc14'], AroonOsc(H, L, 14), "AroonOsc(14)")
    check_close(merged['Parabolic'], ParabolicSAR(H, L, 0.02, 0.02, 0.2), "Parabolic")
    check_close(merged['LinearReg14'], LinearReg(C, 14), "LinReg(14)")
    check_close(merged['LinearRegSlope14'], LinearRegSlope(C, 14), "LinRegSlope")
    check_close(merged['PriceChannelUp20'], PriceChannelUp(H, 20), "PriceChannelUp")
    check_close(merged['PriceChannelDown20'], PriceChannelDown(L, 20), "PriceChDn")

    print("\n--- Volatility ---")
    check_close(merged['ATR14'], ATR(H, L, C, 14), "ATR(14)")
    check_close(merged['BollUp'], get_boll_up(C), "BollUp")
    check_close(merged['BollMid'], get_boll_mid(C), "BollMid")
    check_close(merged['BollDown'], get_boll_down(C), "BollDown")
    check_close(merged['BollWidth'], get_boll_width(C), "BollWidth")
    check_close(merged['KeltnerUp20'], KeltnerUp(H, L, C, 20), "KeltnerUp")
    check_close(merged['KeltnerDown20'], KeltnerDown(H, L, C, 20), "KeltnerDown")
    check_close(merged['ChaikinVol10_10'], ChaikinVolatility(H, L, 10, 10), "ChaikinVol")

    print("\n--- Volume ---")
    check_close(merged['MFI14'], MoneyFlowIndex(H, L, C, V, 14), "MFI(14)")
    check_close(merged['OBV'], OBV(C, V), "OBV")
    check_close(merged['PVT'], PVT(C, V), "PVT")
    check_close(merged['ChaikinMF20'], ChaikinMoneyFlow(H, L, C, V, 20), "CMF(20)")
    check_close(merged['ChaikinOsc'], ChaikinOsc(H, L, C, V), "ChaikinOsc")
    check_close(merged['ADL'], ADL(H, L, C, V), "ADL")
    check_close(merged['KlingerOsc34'], KlingerOsc(H, L, C, V, 34, 55), "Klinger")
    check_close(merged['MassIndex9'], MassIndex(H, L, 9, 25), "MassIndex")
    
    print("\n--- Special / New ---")
    check_close(merged['QQEF14_5'], get_qqef(C), "QQEF")
    check_close(merged['QQES14_5'], get_qqes(C), "QQES")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
