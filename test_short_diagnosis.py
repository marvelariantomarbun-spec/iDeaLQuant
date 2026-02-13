# -*- coding: utf-8 -*-
"""
S2 Short Sinyal Diagnostik Scripti
Her koşulu ayrı ayrı test eder ve hangisinin darboğaz olduğunu bulur.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.data.ideal_parser import load_ideal_data
from src.strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2
from src.strategies.common import Signal
import numpy as np

# Veriyi yükle
chart_data = r"D:\iDeal\ChartData"
df = load_ideal_data(chart_data, "VIP", "X030-T", "1")
if df is None:
    print("Veri bulunamadı!")
    sys.exit(1)

# DataFrame'i Strategy uyumlu hale getir
df['Tipik'] = (df['High'] + df['Low'] + df['Close']) / 3
print(f"Veri: {len(df)} bar, {df['DateTime'].iloc[0]} -> {df['DateTime'].iloc[-1]}")

# Varsayılan parametrelerle strateji oluştur
config = StrategyConfigV2()  # default params
strategy = ARSTrendStrategyV2(
    opens=df['Open'].tolist(),
    highs=df['High'].tolist(),
    lows=df['Low'].tolist(),
    closes=df['Close'].tolist(),
    typical=df['Tipik'].tolist(),
    times=df['DateTime'].tolist(),
    volumes=df['Volume'].tolist(),
    config=config,
)

n = strategy.n
warmup = strategy.warmup_bars

print(f"\n{'='*60}")
print(f"KOŞUL ANALİZİ (warmup={warmup} sonrası, {n - warmup} bar)")
print(f"{'='*60}")

# 1. Trend Yönü
trend_up = sum(1 for i in range(warmup, n) if strategy.trend_yonu[i] == 1)
trend_down = sum(1 for i in range(warmup, n) if strategy.trend_yonu[i] == -1)
trend_flat = sum(1 for i in range(warmup, n) if strategy.trend_yonu[i] == 0)
total = n - warmup
print(f"\n--- TREND YÖNÜ ---")
print(f"  Yukarı (trend=1):  {trend_up:>7} ({trend_up/total*100:.1f}%)")
print(f"  Aşağı  (trend=-1): {trend_down:>7} ({trend_down/total*100:.1f}%)")
print(f"  Nötr   (trend=0):  {trend_flat:>7} ({trend_flat/total*100:.1f}%)")

# 2. Her koşul ayrı ayrı
cfg = config
yeni_dip_count = 0
neg_mom_count = 0
mfi_ok_count = 0
vol_ok_count = 0
all_ok_count = 0

yeni_zirve_count = 0
pos_mom_count = 0
mfi_ok_long_count = 0
vol_ok_long_count = 0
all_ok_long_count = 0

for i in range(warmup, n):
    # --- SHORT koşulları (trend=-1 barlar) ---
    if strategy.trend_yonu[i] == -1 and i > 0:
        yeni_dip = strategy.lows[i] <= strategy.llv[i-1] and strategy.llv[i] < strategy.llv[i-1]
        neg_mom = strategy.momentum[i] < (cfg.momentum_base - cfg.momentum_threshold)
        mfi_ok = strategy.mfi[i] <= strategy.mfi_llv[i-1]
        vol_ok = strategy.volumes[i] >= strategy.volume_hhv[i-1] * cfg.volume_mult
        
        if yeni_dip: yeni_dip_count += 1
        if neg_mom: neg_mom_count += 1
        if mfi_ok: mfi_ok_count += 1
        if vol_ok: vol_ok_count += 1
        if yeni_dip and neg_mom and mfi_ok and vol_ok: all_ok_count += 1
    
    # --- LONG koşulları (trend=1 barlar) ---
    if strategy.trend_yonu[i] == 1 and i > 0:
        yeni_zirve = strategy.highs[i] >= strategy.hhv[i-1] and strategy.hhv[i] > strategy.hhv[i-1]
        pos_mom = strategy.momentum[i] > cfg.momentum_threshold
        mfi_ok_l = strategy.mfi[i] >= strategy.mfi_hhv[i-1]
        vol_ok_l = strategy.volumes[i] >= strategy.volume_hhv[i-1] * cfg.volume_mult
        
        if yeni_zirve: yeni_zirve_count += 1
        if pos_mom: pos_mom_count += 1
        if mfi_ok_l: mfi_ok_long_count += 1
        if vol_ok_l: vol_ok_long_count += 1
        if yeni_zirve and pos_mom and mfi_ok_l and vol_ok_l: all_ok_long_count += 1

print(f"\n--- SHORT KOŞULLARİ (trend=-1 olan {trend_down} bar içinde) ---")
print(f"  yeni_dip (LLV breakout):     {yeni_dip_count:>7} ({yeni_dip_count/max(1,trend_down)*100:.1f}%)")
print(f"  negatif_momentum:            {neg_mom_count:>7} ({neg_mom_count/max(1,trend_down)*100:.1f}%)")
print(f"  mfi_onay (MFI<=LLV):         {mfi_ok_count:>7} ({mfi_ok_count/max(1,trend_down)*100:.1f}%)")
print(f"  volume_onay:                 {vol_ok_count:>7} ({vol_ok_count/max(1,trend_down)*100:.1f}%)")
print(f"  [OK] TUMU BIRDEN:              {all_ok_count:>7} ({all_ok_count/max(1,trend_down)*100:.1f}%)")

print(f"\n--- LONG KOŞULLARİ (trend=1 olan {trend_up} bar içinde) ---")
print(f"  yeni_zirve (HHV breakout):   {yeni_zirve_count:>7} ({yeni_zirve_count/max(1,trend_up)*100:.1f}%)")
print(f"  pozitif_momentum:            {pos_mom_count:>7} ({pos_mom_count/max(1,trend_up)*100:.1f}%)")
print(f"  mfi_onay (MFI>=HHV):         {mfi_ok_long_count:>7} ({mfi_ok_long_count/max(1,trend_up)*100:.1f}%)")
print(f"  volume_onay:                 {vol_ok_long_count:>7} ({vol_ok_long_count/max(1,trend_up)*100:.1f}%)")
print(f"  [OK] TUMU BIRDEN:              {all_ok_long_count:>7} ({all_ok_long_count/max(1,trend_up)*100:.1f}%)")

# 3. Momentum değer dağılımı
mom_vals = [strategy.momentum[i] for i in range(warmup, n)]
print(f"\n--- MOMENTUM DAĞILIMI ---")
print(f"  Min: {min(mom_vals):.2f}, Max: {max(mom_vals):.2f}")
print(f"  Ortalama: {np.mean(mom_vals):.2f}, Medyan: {np.median(mom_vals):.2f}")
print(f"  threshold={cfg.momentum_threshold}, base={cfg.momentum_base}")
print(f"  Long koşulu: mom > {cfg.momentum_threshold}")
print(f"  Short koşulu: mom < {cfg.momentum_base - cfg.momentum_threshold} = {cfg.momentum_base - cfg.momentum_threshold}")

# 4. generate_all_signals sonucu
signals, exits_long, exits_short = strategy.generate_all_signals()
long_entries = np.sum(signals == 1)
short_entries = np.sum(signals == -1)
print(f"\n--- GERÇEK SİNYALLER (generate_all_signals) ---")
print(f"  Long giriş:  {long_entries}")
print(f"  Short giriş: {short_entries}")
print(f"  Long çıkış:  {np.sum(exits_long)}")
print(f"  Short çıkış: {np.sum(exits_short)}")

# 5. Koşul overlap analizi (trend=-1 barlar için)
print(f"\n--- KOŞULLARİN İKİLİ KESİŞİMLERİ (trend=-1 barlar) ---")
dip_and_mom = 0
dip_and_mfi = 0
dip_and_vol = 0
mom_and_mfi = 0
mom_and_vol = 0
mfi_and_vol = 0
dip_mom_mfi = 0
dip_mom_vol = 0

for i in range(warmup, n):
    if strategy.trend_yonu[i] == -1 and i > 0:
        yeni_dip = strategy.lows[i] <= strategy.llv[i-1] and strategy.llv[i] < strategy.llv[i-1]
        neg_mom = strategy.momentum[i] < (cfg.momentum_base - cfg.momentum_threshold)
        mfi_ok = strategy.mfi[i] <= strategy.mfi_llv[i-1]
        vol_ok = strategy.volumes[i] >= strategy.volume_hhv[i-1] * cfg.volume_mult
        
        if yeni_dip and neg_mom: dip_and_mom += 1
        if yeni_dip and mfi_ok: dip_and_mfi += 1
        if yeni_dip and vol_ok: dip_and_vol += 1
        if neg_mom and mfi_ok: mom_and_mfi += 1
        if neg_mom and vol_ok: mom_and_vol += 1
        if mfi_ok and vol_ok: mfi_and_vol += 1
        if yeni_dip and neg_mom and mfi_ok: dip_mom_mfi += 1
        if yeni_dip and neg_mom and vol_ok: dip_mom_vol += 1

print(f"  dip ∩ mom:     {dip_and_mom:>6}")
print(f"  dip ∩ mfi:     {dip_and_mfi:>6}")
print(f"  dip ∩ vol:     {dip_and_vol:>6}")
print(f"  mom ∩ mfi:     {mom_and_mfi:>6}")
print(f"  mom ∩ vol:     {mom_and_vol:>6}")
print(f"  mfi ∩ vol:     {mfi_and_vol:>6}")
print(f"  dip ∩ mom ∩ mfi: {dip_mom_mfi:>4}")
print(f"  dip ∩ mom ∩ vol: {dip_mom_vol:>4}")
print(f"  TÜMÜ:            {all_ok_count:>4}")
