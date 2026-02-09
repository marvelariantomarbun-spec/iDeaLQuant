# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

# Konsol encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_prices():
    # 1. Ham Veri
    print("Fiyat Verisi Yükleniyor...")
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        df_raw = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', header=None, nrows=100000) 
        df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    except:
        print("Data okuma hatası")
        return

    # DateTime
    df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    
    # Numeric convert
    cols_to_num = ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis']
    for c in cols_to_num:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
        
    print(f"Python Veri Başlangıcı: {df_raw['DateTime'].min()}")
    print(f"Python Veri Bitişi    : {df_raw['DateTime'].max()}")
    print("-" * 30)
    
    # 2. İşlem Listesi
    print("IdealData İşlem Listesi Yükleniyor...")
    trade_path = "d:/Projects/IdealQuant/data/ideal_signals_optimized.csv"
    try:
        df_trades = pd.read_csv(trade_path, sep=';', encoding='utf-8-sig', nrows=5)
    except:
        df_trades = pd.read_csv(trade_path, sep=';', encoding='cp1254', nrows=5)
        
    # Tarih parse
    trade_date_str = df_trades.iloc[0, 3] # İlk işlem tarihi
    trade_date = pd.to_datetime(trade_date_str, format='%d.%m.%Y %H:%M', errors='coerce')
    
    print(f"IdealData İlk İşlem  : {trade_date}")
    
    # Fiyat Analizi (İlk İşlem İçin)
    trade_price = float(str(df_trades.iloc[0, 4]).replace(',', '.'))
    print(f"IdealData İşlem Fiyatı: {trade_price}")
    
    row = df_raw[df_raw['DateTime'] == trade_date]
    if len(row) > 0:
        bar_open = row.iloc[0]['Acilis']
        bar_close = row.iloc[0]['Kapanis']
        print(f"Bar ({trade_date}) -> Open: {bar_open}, Close: {bar_close}")
        
        matches = []
        if abs(trade_price - bar_close) < 0.1: matches.append("THIS_CLOSE")
        if abs(trade_price - bar_open) < 0.1: matches.append("THIS_OPEN")
        
        # Next Open
        idx = row.index[0]
        if idx + 1 < len(df_raw):
            next_open = df_raw.iloc[idx+1]['Acilis']
            print(f"Next Bar Open: {next_open}")
            if abs(trade_price - next_open) < 0.1: matches.append("NEXT_OPEN")
            
        print(f"EŞLEŞME TİPİ: {matches}")
    else:
        print("İlk işlem tarihi fiyat verisinde bulunamadı.")

if __name__ == "__main__":
    check_prices()
