# -*- coding: utf-8 -*-
"""
IdealData vs Python Sinyal Karşılaştırma
1_Nolu_Strateji (Gatekeeper) için
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime

from src.strategies.score_based import ScoreBasedStrategy, ScoreConfig

def load_ohlc_data(filepath: str) -> pd.DataFrame:
    """OHLC verisini yükle"""
    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8')
    df.columns = ['Tarih', 'Saat', 'Open', 'High', 'Low', 'Close', 'Avg', 'Volume', 'Lot']
    df['Datetime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

def load_ideal_signals(filepath: str) -> pd.DataFrame:
    """IdealData sinyallerini yükle"""
    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8')
    # "Açılış Tarihi" formatı: "25.12.2024 21:06"
    df['Entry_DT'] = pd.to_datetime(df['Açılış Tarihi'], format='%d.%m.%Y %H:%M')
    df['Exit_DT'] = pd.to_datetime(df['Kapanış Tarihi'], format='%d.%m.%Y %H:%M')
    df['Direction'] = df['Yön'].map({'Alış': 'LONG', 'Satış': 'SHORT'})
    return df

def run_python_strategy(ohlc: pd.DataFrame) -> list:
    """Python stratejisini çalıştır"""
    opens = ohlc['Open'].tolist()
    highs = ohlc['High'].tolist()
    lows = ohlc['Low'].tolist()
    closes = ohlc['Close'].tolist()
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    
    config = ScoreConfig()
    strategy = ScoreBasedStrategy(opens, highs, lows, closes, typical, config)
    
    signals = []
    position = "FLAT"
    
    for i in range(len(closes)):
        signal = strategy.get_signal(i, position)
        
        if signal.name == "LONG":
            if position == "FLAT":
                signals.append({
                    'bar_idx': i,
                    'datetime': ohlc['Datetime'].iloc[i],
                    'direction': 'LONG',
                    'type': 'ENTRY',
                    'price': closes[i]
                })
                position = "LONG"
        elif signal.name == "SHORT":
            if position == "FLAT":
                signals.append({
                    'bar_idx': i,
                    'datetime': ohlc['Datetime'].iloc[i],
                    'direction': 'SHORT',
                    'type': 'ENTRY',
                    'price': closes[i]
                })
                position = "SHORT"
        elif signal.name == "FLAT":
            if position != "FLAT":
                signals.append({
                    'bar_idx': i,
                    'datetime': ohlc['Datetime'].iloc[i],
                    'direction': position,
                    'type': 'EXIT',
                    'price': closes[i]
                })
                position = "FLAT"
    
    return signals

def compare_signals(ideal_df: pd.DataFrame, python_signals: list, ohlc: pd.DataFrame) -> dict:
    """Sinyalleri karşılaştır"""
    # Python sinyallerini trade'lere dönüştür
    python_trades = []
    i = 0
    while i < len(python_signals):
        if python_signals[i]['type'] == 'ENTRY':
            entry = python_signals[i]
            exit_sig = None
            if i + 1 < len(python_signals) and python_signals[i + 1]['type'] == 'EXIT':
                exit_sig = python_signals[i + 1]
                i += 1
            python_trades.append({
                'entry_dt': entry['datetime'],
                'direction': entry['direction'],
                'entry_price': entry['price'],
                'exit_dt': exit_sig['datetime'] if exit_sig else None,
                'exit_price': exit_sig['price'] if exit_sig else None
            })
        i += 1
    
    # Karşılaştırma - Optimize edilmiş (dict lookup)
    results = {
        'ideal_trades': len(ideal_df),
        'python_trades': len(python_trades),
        'matched': 0,
        'direction_match': 0,
        'time_diff_avg': 0,
        'mismatches': []
    }
    
    # Python trades'i datetime'a göre indexle
    py_by_time = {}
    for pt in python_trades:
        key = pt['entry_dt'].floor('min')
        if key not in py_by_time:
            py_by_time[key] = []
        py_by_time[key].append(pt)
    
    time_diffs = []
    
    for idx, ideal_trade in ideal_df.iterrows():
        ideal_entry = ideal_trade['Entry_DT']
        ideal_dir = ideal_trade['Direction']
        
        # ±5 dakika içinde ara
        best_match = None
        best_diff = pd.Timedelta(minutes=10)
        
        for offset in range(-5, 6):
            check_time = ideal_entry + pd.Timedelta(minutes=offset)
            check_key = check_time.floor('min')
            if check_key in py_by_time:
                for py_trade in py_by_time[check_key]:
                    diff = abs(py_trade['entry_dt'] - ideal_entry)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = py_trade
        
        if best_match and best_diff <= pd.Timedelta(minutes=5):
            results['matched'] += 1
            time_diffs.append(best_diff.total_seconds())
            if best_match['direction'] == ideal_dir:
                results['direction_match'] += 1
            else:
                if len(results['mismatches']) < 50:
                    results['mismatches'].append({
                        'ideal_time': str(ideal_entry),
                        'ideal_dir': ideal_dir,
                        'python_time': str(best_match['entry_dt']),
                        'python_dir': best_match['direction'],
                        'issue': 'DIRECTION_MISMATCH'
                    })
        else:
            if len(results['mismatches']) < 50:
                results['mismatches'].append({
                    'ideal_time': str(ideal_entry),
                    'ideal_dir': ideal_dir,
                    'python_time': str(best_match['entry_dt']) if best_match else 'N/A',
                    'python_dir': best_match['direction'] if best_match else 'N/A',
                    'issue': 'NO_MATCH'
                })
    
    if time_diffs:
        results['time_diff_avg'] = sum(time_diffs) / len(time_diffs)
    
    return results, python_trades

def main():
    print("=" * 80)
    print("1_NOLU_STRATEJI (GATEKEEPER) - IDEAL vs PYTHON KARŞILAŞTIRMASI")
    print("=" * 80)
    
    # Veri yükle
    ohlc = load_ohlc_data('data/VIP_X030T_1dk_.csv')
    print(f"\nOHLC Veri: {len(ohlc):,} bar ({ohlc['Datetime'].iloc[0]} - {ohlc['Datetime'].iloc[-1]})")
    
    ideal_df = load_ideal_signals('data/ideal_signals_1_Nolu_Strateji_200000Bar.csv')
    print(f"Ideal Sinyaller: {len(ideal_df)} trade")
    
    # Python stratejisini çalıştır
    print("\nPython stratejisi çalıştırılıyor...")
    python_signals = run_python_strategy(ohlc)
    print(f"Python Sinyaller: {len(python_signals)} sinyal")
    
    # Karşılaştır
    results, python_trades = compare_signals(ideal_df, python_signals, ohlc)
    
    print("\n" + "=" * 80)
    print("SONUÇLAR")
    print("=" * 80)
    print(f"IdealData Trade Sayısı: {results['ideal_trades']}")
    print(f"Python Trade Sayısı:    {results['python_trades']}")
    print(f"Eşleşen Trade:          {results['matched']} ({100*results['matched']/results['ideal_trades']:.1f}%)")
    print(f"Yön Eşleşme:            {results['direction_match']} ({100*results['direction_match']/max(1,results['matched']):.1f}%)")
    print(f"Ortalama Zaman Farkı:   {results['time_diff_avg']:.1f} saniye")
    
    # İlk 10 eşleşmeyen trade
    if results['mismatches']:
        print(f"\n--- İLK 10 UYUŞMAZLIK ---")
        for m in results['mismatches'][:10]:
            print(f"  {m['issue']}: Ideal={m['ideal_time']} {m['ideal_dir']} | Python={m['python_time']} {m['python_dir']}")
    
    # İlk birkaç Python trade
    print(f"\n--- İLK 10 PYTHON TRADE ---")
    for t in python_trades[:10]:
        print(f"  {t['entry_dt']} - {t['direction']} @ {t['entry_price']}")
    
    # İlk birkaç Ideal trade
    print(f"\n--- İLK 10 IDEAL TRADE ---")
    for idx, row in ideal_df.head(10).iterrows():
        print(f"  {row['Entry_DT']} - {row['Direction']} @ {row['Açılış Fyt']}")
    
    return results

if __name__ == "__main__":
    main()
