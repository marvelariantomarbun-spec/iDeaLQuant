# -*- coding: utf-8 -*-
"""
P&L (Kar/Zarar) Uyumu Testi
IdealData işlem listesi ile Python backtest sonuçlarını karşılaştırır.
"""

import sys, io, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2, Signal

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("P&L UYUMU TESTİ - ARS Trend v2")
print("="*80)

# =============================================================================
# 1. IdealData İşlem Listesini Yükle
# =============================================================================
print("\n[1] IdealData işlem listesi yükleniyor...")

ref_path = "d:/Projects/IdealQuant/data/ideal_signals_ars_v2.csv"
try:
    df_ideal = pd.read_csv(ref_path, sep=';', encoding='utf-8')
except:
    df_ideal = pd.read_csv(ref_path, sep=';', encoding='cp1254')

# Kolon isimlerini index ile al (encoding sorunlarını önlemek için)
col_no = df_ideal.columns[0]
col_yon = df_ideal.columns[1]
col_lot = df_ideal.columns[2]
col_acilis_tarih = df_ideal.columns[3]
col_acilis_fiyat = df_ideal.columns[4]
col_kapanis_tarih = df_ideal.columns[5]
col_kapanis_fiyat = df_ideal.columns[6]
col_kar_zarar = df_ideal.columns[7]
col_bakiye = df_ideal.columns[8]

# Parse
df_ideal['OpenTime'] = pd.to_datetime(df_ideal[col_acilis_tarih], format='%d.%m.%Y %H:%M', errors='coerce')
df_ideal['CloseTime'] = pd.to_datetime(df_ideal[col_kapanis_tarih], format='%d.%m.%Y %H:%M', errors='coerce')
df_ideal['Direction'] = df_ideal[col_yon].map({'Alış': 'LONG', 'Satış': 'SHORT'})

# Fiyatları parse et (Türkçe format: 11.609,00 -> 11609.00)
def parse_price(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace('.', '').replace(',', '.')
    try:
        return float(s)
    except:
        return np.nan

df_ideal['EntryPrice'] = df_ideal[col_acilis_fiyat].apply(parse_price)
df_ideal['ExitPrice'] = df_ideal[col_kapanis_fiyat].apply(parse_price)
df_ideal['PnL'] = df_ideal[col_kar_zarar].apply(parse_price)

# Geçerli işlemler
df_ideal = df_ideal.dropna(subset=['OpenTime', 'Direction', 'EntryPrice'])

print(f"   Toplam işlem: {len(df_ideal)}")
print(f"   İlk işlem: {df_ideal.iloc[0]['OpenTime']} - {df_ideal.iloc[0]['Direction']}")
print(f"   Son işlem: {df_ideal.iloc[-1]['OpenTime']} - {df_ideal.iloc[-1]['Direction']}")

# IdealData toplamları
ideal_total_pnl = df_ideal['PnL'].sum()
ideal_winning = len(df_ideal[df_ideal['PnL'] > 0])
ideal_losing = len(df_ideal[df_ideal['PnL'] < 0])
ideal_breakeven = len(df_ideal[df_ideal['PnL'] == 0])

print(f"\n   IdealData Özet:")
print(f"   Toplam P&L: {ideal_total_pnl:,.0f} puan")
print(f"   Kazanan: {ideal_winning} | Kaybeden: {ideal_losing} | Başabaş: {ideal_breakeven}")

# =============================================================================
# 2. Market Verisini Yükle
# =============================================================================
print("\n[2] Market verisi yükleniyor...")

data_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
df_raw = pd.read_csv(data_path, sep=';', decimal=',', encoding='cp1254', low_memory=False)
df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
df_raw = df_raw.sort_values('DateTime').reset_index(drop=True)

opens = df_raw['Acilis'].values.tolist()
highs = df_raw['Yuksek'].values.tolist()
lows = df_raw['Dusuk'].values.tolist()
closes = df_raw['Kapanis'].values.tolist()
typical = ((df_raw['Yuksek'] + df_raw['Dusuk'] + df_raw['Kapanis']) / 3).tolist()
times = df_raw['DateTime'].tolist()

print(f"   Toplam bar: {len(df_raw)}")

# =============================================================================
# 3. Python Backtest Çalıştır
# =============================================================================
print("\n[3] Python backtest çalıştırılıyor...")

config = StrategyConfigV2(
    ars_ema_period = 3,
    ars_atr_period = 10,
    ars_atr_mult = 0.5,
    ars_min_band = 0.002,
    ars_max_band = 0.015,
    momentum_period = 5,
    breakout_period = 10,
    rsi_period = 14,
    kar_al_pct = 3.0,
    iz_stop_pct = 1.5,
    vade_tipi = "ENDEKS"
)

strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, config)

# Backtest State
current_position = "FLAT"
entry_price = 0.0
entry_time = None
extreme_price = 0.0
python_trades = []

# İlk işlemden biraz önce başla
first_trade_date = df_ideal.iloc[0]['OpenTime']
start_idx = 0
for i, t in enumerate(times):
    if t >= first_trade_date - timedelta(days=5):
        start_idx = i
        break

for i in range(start_idx, len(closes)):
    signal = strategy.get_signal(i, current_position, entry_price, extreme_price)
    
    # Trailing stop için extreme güncelle
    if current_position == "LONG":
        extreme_price = max(extreme_price, highs[i])
    elif current_position == "SHORT":
        extreme_price = min(extreme_price, lows[i])
        
    # Sinyal işle
    if signal == Signal.LONG and current_position != "LONG":
        if current_position == "SHORT":
            # Short kapat
            pnl = entry_price - closes[i]  # Short P&L
            python_trades.append({
                'Direction': 'SHORT',
                'OpenTime': entry_time,
                'CloseTime': times[i],
                'EntryPrice': entry_price,
                'ExitPrice': closes[i],
                'PnL': pnl
            })
        # Long aç
        current_position = "LONG"
        entry_price = closes[i]
        entry_time = times[i]
        extreme_price = closes[i]
        
    elif signal == Signal.SHORT and current_position != "SHORT":
        if current_position == "LONG":
            # Long kapat
            pnl = closes[i] - entry_price  # Long P&L
            python_trades.append({
                'Direction': 'LONG',
                'OpenTime': entry_time,
                'CloseTime': times[i],
                'EntryPrice': entry_price,
                'ExitPrice': closes[i],
                'PnL': pnl
            })
        # Short aç
        current_position = "SHORT"
        entry_price = closes[i]
        entry_time = times[i]
        extreme_price = closes[i]
        
    elif signal == Signal.FLAT and current_position != "FLAT":
        if current_position == "LONG":
            pnl = closes[i] - entry_price
        else:
            pnl = entry_price - closes[i]
            
        python_trades.append({
            'Direction': current_position,
            'OpenTime': entry_time,
            'CloseTime': times[i],
            'EntryPrice': entry_price,
            'ExitPrice': closes[i],
            'PnL': pnl
        })
        current_position = "FLAT"
        entry_price = 0
        entry_time = None

df_python = pd.DataFrame(python_trades)

print(f"   Python işlem sayısı: {len(df_python)}")

# Python toplamları
python_total_pnl = df_python['PnL'].sum()
python_winning = len(df_python[df_python['PnL'] > 0])
python_losing = len(df_python[df_python['PnL'] < 0])
python_breakeven = len(df_python[df_python['PnL'] == 0])

print(f"\n   Python Özet:")
print(f"   Toplam P&L: {python_total_pnl:,.0f} puan")
print(f"   Kazanan: {python_winning} | Kaybeden: {python_losing} | Başabaş: {python_breakeven}")

# =============================================================================
# 4. Karşılaştırma
# =============================================================================
print("\n" + "="*80)
print("KARŞILAŞTIRMA SONUÇLARI")
print("="*80)

print(f"\n{'Metrik':<25} {'IdealData':<15} {'Python':<15} {'Fark':<15} {'Uyum'}")
print("-"*80)

# İşlem sayısı
trade_diff = abs(len(df_ideal) - len(df_python))
trade_pct = (1 - trade_diff / len(df_ideal)) * 100
trade_status = "✅" if trade_pct > 95 else "❌"
print(f"{'İşlem Sayısı':<25} {len(df_ideal):<15} {len(df_python):<15} {trade_diff:<15} {trade_status} {trade_pct:.1f}%")

# Toplam P&L
pnl_diff = abs(ideal_total_pnl - python_total_pnl)
pnl_pct_diff = abs(pnl_diff / abs(ideal_total_pnl)) * 100 if ideal_total_pnl != 0 else 0
pnl_status = "✅" if pnl_pct_diff < 5 else ("⚠️" if pnl_pct_diff < 10 else "❌")
print(f"{'Toplam P&L':<25} {ideal_total_pnl:,.0f}{'':<7} {python_total_pnl:,.0f}{'':<7} {pnl_diff:,.0f}{'':<7} {pnl_status} {pnl_pct_diff:.1f}%")

# Kazanan sayısı
win_diff = abs(ideal_winning - python_winning)
print(f"{'Kazanan İşlem':<25} {ideal_winning:<15} {python_winning:<15} {win_diff:<15}")

# Kaybeden sayısı
lose_diff = abs(ideal_losing - python_losing)
print(f"{'Kaybeden İşlem':<25} {ideal_losing:<15} {python_losing:<15} {lose_diff:<15}")

# =============================================================================
# 5. İşlem Bazlı Eşleştirme
# =============================================================================
print("\n" + "="*80)
print("İŞLEM BAZLI EŞLEŞTİRME (İlk 20 + Farklı olanlar)")
print("="*80)

matched_pnl_diff = []
unmatched_count = 0

print(f"\n{'#':<5} {'Py_Open':<20} {'Dir':<6} {'Py_PnL':<10} {'ID_PnL':<10} {'Diff':<10} {'Match'}")
print("-"*80)

for i, py_row in df_python.iterrows():
    # En yakın IdealData işlemini bul
    window = timedelta(minutes=60)
    candidates = df_ideal[
        (df_ideal['Direction'] == py_row['Direction']) &
        (df_ideal['OpenTime'] >= py_row['OpenTime'] - window) &
        (df_ideal['OpenTime'] <= py_row['OpenTime'] + window)
    ]
    
    if not candidates.empty:
        candidates['TimeDiff'] = (candidates['OpenTime'] - py_row['OpenTime']).abs()
        best = candidates.sort_values('TimeDiff').iloc[0]
        
        py_pnl = py_row['PnL']
        id_pnl = best['PnL'] if not pd.isna(best['PnL']) else 0
        diff = abs(py_pnl - id_pnl)
        matched_pnl_diff.append(diff)
        
        if diff < 5:
            status = "✅"
        elif diff < 20:
            status = "⚠️"
        else:
            status = "❌"
            
        # İlk 20 ve farklı olanları göster
        if i < 20 or diff >= 20:
            print(f"{i:<5} {str(py_row['OpenTime']):<20} {py_row['Direction']:<6} {py_pnl:<10.0f} {id_pnl:<10.0f} {diff:<10.0f} {status}")
    else:
        unmatched_count += 1
        if i < 20:
            print(f"{i:<5} {str(py_row['OpenTime']):<20} {py_row['Direction']:<6} {py_row['PnL']:<10.0f} {'---':<10} {'---':<10} ❌ NoMatch")

# =============================================================================
# 6. Özet
# =============================================================================
print("\n" + "="*80)
print("ÖZET")
print("="*80)

if matched_pnl_diff:
    avg_pnl_diff = np.mean(matched_pnl_diff)
    max_pnl_diff = np.max(matched_pnl_diff)
    perfect_match = len([d for d in matched_pnl_diff if d < 1])
    good_match = len([d for d in matched_pnl_diff if d < 5])
    
    print(f"\nP&L Eşleştirme:")
    print(f"  Eşleştirilen işlem: {len(matched_pnl_diff)}")
    print(f"  Eşleştirilemeyen: {unmatched_count}")
    print(f"  Ortalama P&L farkı: {avg_pnl_diff:.1f} puan")
    print(f"  Maksimum P&L farkı: {max_pnl_diff:.0f} puan")
    print(f"  Mükemmel eşleşen (<1 puan): {perfect_match} ({perfect_match/len(matched_pnl_diff)*100:.1f}%)")
    print(f"  İyi eşleşen (<5 puan): {good_match} ({good_match/len(matched_pnl_diff)*100:.1f}%)")

# Genel sonuç
print(f"\n{'='*80}")
if trade_pct > 95 and pnl_pct_diff < 10:
    print("✅ P&L UYUMU TESTİ BAŞARILI")
    print(f"   İşlem uyumu: {trade_pct:.1f}%")
    print(f"   Toplam P&L farkı: {pnl_pct_diff:.1f}%")
else:
    print("⚠️ P&L UYUMU TESTİ TAMAMLANDI - FARKLAR VAR")
    print(f"   İşlem uyumu: {trade_pct:.1f}%")
    print(f"   Toplam P&L farkı: {pnl_pct_diff:.1f}%")
print("="*80)
