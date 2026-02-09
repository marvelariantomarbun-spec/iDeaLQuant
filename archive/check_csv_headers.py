# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

# Konsol encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

csv_path = 'd:/Projects/IdealQuant/data/ideal_signals.csv'

try:
    # Önce cp1254 dene
    df = pd.read_csv(csv_path, sep=';', encoding='cp1254', nrows=5)
except:
    try:
        # Sonra utf-8-sig dene
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig', nrows=5)
    except:
        print("CSV okunamadı (Encoding hatası)")
        sys.exit(1)

print("Sütunlar:", df.columns.tolist())
print("-" * 30)

# Sütun isimlerinden bağımsız index ile erişim (Garanti olsun)
# Genelde: No, Yön, Lot, Açılış Tarihi (3), Açılış Fyt, Kapanış Tarihi (5) ...
try:
    col_acilis_tarih = df.columns[3]
    col_kapanis_tarih = df.columns[5]
    
    val_acilis = df.iloc[0, 3]
    val_kapanis = df.iloc[0, 5]
    
    print(f"Col[{col_acilis_tarih}] İlk Değer: '{val_acilis}'")
    print(f"Col[{col_kapanis_tarih}] İlk Değer: '{val_kapanis}'")
    
    # Saat kontrolü
    has_time_acilis = ":" in str(val_acilis)
    has_time_kapanis = ":" in str(val_kapanis)
    
    print(f"Açılış Saati Var mı? {'EVET' if has_time_acilis else 'HAYIR'}")
    print(f"Kapanış Saati Var mı? {'EVET' if has_time_kapanis else 'HAYIR'}")
    
except Exception as e:
    print(f"Erişim hatası: {e}")
