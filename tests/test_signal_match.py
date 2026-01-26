# -*- coding: utf-8 -*-
"""
Test Signal Match
IdealData'dan gelen indikatör verilerini (ideal_score_indicators.csv) kullanarak
Python tarafında Sinyal (Al/Sat) üretir ve bunları 'python_signals.csv' dosyasına kaydeder.

Amaç: IdealData işlem listesiyle %100 uyumu doğrulamak.
"""

import sys
import io
import pandas as pd
import numpy as np
import os

# Proje kök dizinini ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.score_based import ScoreBasedStrategy, ScoreConfig
from strategies.ars_trend import Signal

# Konsol encoding ayarı
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def load_data():
    try:
        # 1. İndikatörler (IdealData'dan gelen)
        df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_score_indicators.csv", sep=';', decimal=',', encoding='utf-8') 
        df_ind.columns = [c.strip() for c in df_ind.columns]
        for col in df_ind.columns:
            if col not in ['Date', 'Time', 'Tarih', 'Saat']:
                df_ind[col] = pd.to_numeric(df_ind[col], errors='coerce')

        # 2. Ham Veri (Fiyatlar)
        csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
        # Header kontrolü
        try:
            df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, nrows=5)
            if isinstance(df_raw.iloc[0,0], str) and "Tarih" in df_raw.iloc[0,0]:
                df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
                df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
            else:
                df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
                df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        except:
             df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None)
             df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']

        return df_ind, df_raw
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None, None

def run_signal_test():
    print("Veriler yükleniyor...")
    df_ind, df_raw = load_data()
    if df_ind is None or df_raw is None:
        return

    print(f"İndikatör Verisi: {len(df_ind)} bar")
    print(f"Fiyat Verisi    : {len(df_raw)} bar")
    
    n = min(len(df_ind), len(df_raw))
    df_ind = df_ind.iloc[-n:].reset_index(drop=True)
    df_raw = df_raw.iloc[-n:].reset_index(drop=True)
    
    opens = df_raw['Acilis'].values.tolist()
    highs = df_raw['Yuksek'].values.tolist()
    lows = df_raw['Dusuk'].values.tolist()
    closes = df_raw['Kapanis'].values.tolist()
    typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
    
    config = ScoreConfig()
    algo = ScoreBasedStrategy(opens, highs, lows, closes, typical, config, indicators_df=df_ind)
    
    print("Sinyaller üretiliyor...")
    signals = []
    positions = [] 
    current_pos_str = "FLAT" 
    
    # PnL Takibi
    entry_price = 0.0
    total_pnl = 0.0
    win_count = 0
    loss_count = 0
    trade_count = 0
    
    for i in range(n):
        enum_sig = algo.get_signal(i, current_pos_str)
        
        sig_code = ""
        current_price = closes[i]
        
        # Pozisyon Değişimi Varsa
        if enum_sig != Signal.NONE:
            # ÇIKIŞ (FLAT)
            if enum_sig == Signal.FLAT:
                profit = 0
                if current_pos_str == "LONG":
                    profit = current_price - entry_price
                elif current_pos_str == "SHORT":
                    profit = entry_price - current_price
                
                total_pnl += profit
                trade_count += 1
                if profit > 0: win_count += 1
                else: loss_count += 1
                
                sig_code = "F"
                current_pos_str = "FLAT"
                
            # GİRİŞ (LONG/SHORT)
            elif enum_sig == Signal.LONG:
                entry_price = current_price
                sig_code = "A"
                current_pos_str = "LONG"
                
            elif enum_sig == Signal.SHORT:
                entry_price = current_price
                sig_code = "S"
                current_pos_str = "SHORT"
            
        signals.append(sig_code)
        
        pos_val = 0
        if current_pos_str == "LONG": pos_val = 1
        elif current_pos_str == "SHORT": pos_val = -1
        positions.append(pos_val)
        
    df_res = pd.DataFrame({
        'Tarih': df_raw['Tarih'],
        'Saat': df_raw['Saat'],
        'Kapanis': closes,
        'Sinyal': signals,
        'Pozisyon': positions
    })
    
    df_trades = df_res[df_res['Sinyal'] != ""].copy()
    
    print(f"Toplam Sinyal (Event) Sayısı: {len(df_trades)}")
    print(f"Gerçekleşen İşlem (Trade) Sayısı: {trade_count}")
    print(f"Net PnL Hesaplanan: {total_pnl:.2f} Puan")
    print(f"Kazanma Oranı: %{(win_count/trade_count*100):.2f}" if trade_count > 0 else "0")
    
    print("\nSon 10 Sinyal:")
    print(df_trades.tail(10))
    
    output_path = "d:/Projects/IdealQuant/tests/python_signals.csv"
    df_trades.to_csv(output_path, index=False, sep=';')
    print(f"\nSinyaller kaydedildi: {output_path}")
    print("Bu dosyayı IdealData işlem listesiyle karşılaştırabilirsiniz.")

if __name__ == "__main__":
    run_signal_test()
