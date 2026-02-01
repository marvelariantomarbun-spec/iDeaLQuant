# -*- coding: utf-8 -*-
"""
ARS Trend v2 - Uyumsuz Sinyallerin Derin Analizi
"""

import sys
import os
sys.path.insert(0, 'd:/Projects/IdealQuant/src')
sys.path.insert(0, 'd:/Projects/IdealQuant')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date

from src.strategies.ars_trend_v2 import (
    ARSTrendStrategyV2, StrategyConfigV2, Signal,
    is_arefe, is_bayram_tatili, is_resmi_tatil, vade_sonu_is_gunu, is_seans_icinde
)

# =====================================================
# 1. Veri Yükleme
# =====================================================
def load_data():
    # IdealData sinyalleri
    csv_path = 'd:/Projects/IdealQuant/data/ideal_signals_2_Nolu_Strateji_200000Bar.csv'
    ideal_df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    ideal_df.columns = ['No', 'Yon', 'Lot', 'AcilisTarihi', 'AcilisFyt', 
                        'KapanisTarihi', 'KapanisFyt', 'KarZarar', 'Bakiye']
    ideal_df['Time'] = pd.to_datetime(ideal_df['AcilisTarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
    ideal_df['Direction'] = ideal_df['Yon'].apply(lambda x: 'LONG' if 'Al' in str(x) else 'SHORT')
    ideal_df['Price'] = ideal_df['AcilisFyt'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    # Fiyat verisi
    price_path = 'd:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv'
    price_df = pd.read_csv(price_path, sep=';', decimal=',', encoding='cp1254', low_memory=False)
    price_df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    price_df['DateTime'] = pd.to_datetime(price_df['Tarih'] + ' ' + price_df['Saat'], format='%d.%m.%Y %H:%M:%S')
    
    return ideal_df, price_df


def run_python_strategy_with_debug(price_df):
    """Python stratejisini çalıştır ve tüm sinyalleri + debug bilgilerini döndür"""
    
    opens = price_df['Acilis'].values.tolist()
    highs = price_df['Yuksek'].values.tolist()
    lows = price_df['Dusuk'].values.tolist()
    closes = price_df['Kapanis'].values.tolist()
    typical = ((price_df['Yuksek'] + price_df['Dusuk'] + price_df['Kapanis']) / 3).tolist()
    times = price_df['DateTime'].tolist()
    volumes = price_df['Lot'].values.tolist()
    
    config = StrategyConfigV2(
        ars_ema_period=3, ars_atr_period=10, ars_atr_mult=0.5,
        ars_min_band=0.002, ars_max_band=0.015,
        momentum_period=5, breakout_period=10,
        mfi_period=14, mfi_hhv_period=14, mfi_llv_period=14,
        volume_hhv_period=14, volume_llv_period=14,
        kar_al_pct=3.0, iz_stop_pct=1.5, vade_tipi='ENDEKS'
    )
    
    strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, volumes, config)
    
    # Tüm sinyal bilgilerini topla
    all_signals = []
    current_pos = 'FLAT'
    entry_price = 0.0
    extreme_price = 0.0
    
    for i in range(len(closes)):
        if current_pos == 'LONG':
            extreme_price = max(extreme_price, highs[i])
        elif current_pos == 'SHORT':
            extreme_price = min(extreme_price if extreme_price > 0 else highs[i], lows[i])
        
        sig = strategy.get_signal(i, current_pos, entry_price, extreme_price)
        
        if sig != Signal.NONE:
            signal_info = {
                'bar': i,
                'time': times[i],
                'signal': sig.name,
                'price': closes[i],
                'trend': strategy.trend_yonu[i] if i < len(strategy.trend_yonu) else None,
                'momentum': strategy.momentum[i] if i < len(strategy.momentum) else None,
                'mfi': strategy.mfi[i] if i < len(strategy.mfi) else None,
                'volume': volumes[i],
                'in_warmup': strategy._is_in_warmup(i),
                'is_vade_sonu': times[i].date() in strategy.vade_sonu_gunleri if hasattr(strategy, 'vade_sonu_gunleri') else False,
            }
            all_signals.append(signal_info)
            
            if sig == Signal.LONG:
                current_pos = 'LONG'
                entry_price = closes[i]
                extreme_price = highs[i]
            elif sig == Signal.SHORT:
                current_pos = 'SHORT'
                entry_price = closes[i]
                extreme_price = lows[i]
            elif sig == Signal.FLAT:
                current_pos = 'FLAT'
                entry_price = 0.0
                extreme_price = 0.0
    
    return pd.DataFrame(all_signals), strategy


def analyze_mismatch(ideal_row, price_df, strategy, python_signals_df, tolerance_minutes=5):
    """Tek bir uyumsuzluğu detaylı analiz et"""
    
    ideal_time = ideal_row['Time']
    ideal_dir = ideal_row['Direction']
    ideal_price = ideal_row['Price']
    
    analysis = {
        'ideal_time': ideal_time,
        'ideal_dir': ideal_dir,
        'ideal_price': ideal_price,
        'reason': 'UNKNOWN',
        'details': {}
    }
    
    # 1. O zaman diliminde Python'da ne var?
    time_window_start = ideal_time - timedelta(minutes=tolerance_minutes)
    time_window_end = ideal_time + timedelta(minutes=tolerance_minutes)
    
    nearby_python = python_signals_df[
        (python_signals_df['time'] >= time_window_start) & 
        (python_signals_df['time'] <= time_window_end)
    ]
    
    analysis['details']['nearby_python_signals'] = nearby_python.to_dict('records') if len(nearby_python) > 0 else []
    
    # 2. Fiyat verisinden o bar'ı bul
    bar_match = price_df[price_df['DateTime'] == ideal_time]
    if len(bar_match) == 0:
        # 1 dakika tolerans
        bar_match = price_df[
            (price_df['DateTime'] >= ideal_time - timedelta(minutes=1)) & 
            (price_df['DateTime'] <= ideal_time + timedelta(minutes=1))
        ]
    
    if len(bar_match) > 0:
        bar_idx = bar_match.index[0]
        bar_time = price_df.loc[bar_idx, 'DateTime']
        
        # 3. Strateji değerlerini kontrol et
        if bar_idx < len(strategy.trend_yonu):
            analysis['details']['bar_idx'] = int(bar_idx)
            analysis['details']['trend'] = int(strategy.trend_yonu[bar_idx])
            analysis['details']['momentum'] = float(strategy.momentum[bar_idx])
            analysis['details']['mfi'] = float(strategy.mfi[bar_idx])
            analysis['details']['volume'] = float(price_df.loc[bar_idx, 'Lot'])
            analysis['details']['close'] = float(price_df.loc[bar_idx, 'Kapanis'])
            analysis['details']['high'] = float(price_df.loc[bar_idx, 'Yuksek'])
            analysis['details']['low'] = float(price_df.loc[bar_idx, 'Dusuk'])
            
            # HHV/LLV değerleri
            if bar_idx > 0:
                analysis['details']['hhv_prev'] = float(strategy.hhv[bar_idx-1])
                analysis['details']['llv_prev'] = float(strategy.llv[bar_idx-1])
                analysis['details']['mfi_hhv_prev'] = float(strategy.mfi_hhv[bar_idx-1])
                analysis['details']['mfi_llv_prev'] = float(strategy.mfi_llv[bar_idx-1])
                analysis['details']['vol_hhv_prev'] = float(strategy.volume_hhv[bar_idx-1])
            
            # Isınma kontrolü
            analysis['details']['in_warmup'] = strategy._is_in_warmup(bar_idx)
            
            # Vade sonu kontrolü
            current_date = bar_time.date()
            analysis['details']['is_vade_sonu'] = current_date in strategy.vade_sonu_gunleri
            analysis['details']['is_arefe'] = is_arefe(current_date)
            analysis['details']['is_bayram'] = is_bayram_tatili(current_date)
            analysis['details']['is_resmi_tatil'] = is_resmi_tatil(current_date)
            
            # Seans kontrolü
            current_t = bar_time.time()
            analysis['details']['is_seans_icinde'] = is_seans_icinde(current_t)
            analysis['details']['time_of_day'] = str(current_t)
            
            # Vade geçişi kontrolü
            analysis['details']['is_vade_gecisi'] = bar_idx in strategy.vade_gecis_barlari
            
            # 4. Neden eşleşmedi?
            if analysis['details']['in_warmup']:
                analysis['reason'] = 'WARMUP_PERIOD'
            elif not analysis['details']['is_seans_icinde']:
                analysis['reason'] = 'SEANS_DISI'
            elif analysis['details']['is_vade_gecisi']:
                analysis['reason'] = 'VADE_GECISI_COOLDOWN'
            elif analysis['details']['is_vade_sonu']:
                analysis['reason'] = 'VADE_SONU_KAPANISI'
            elif analysis['details']['is_arefe']:
                analysis['reason'] = 'AREFE_GUNU'
            elif analysis['details']['is_bayram']:
                analysis['reason'] = 'BAYRAM_TATILI'
            else:
                # Koşulları kontrol et
                trend = analysis['details']['trend']
                momentum = analysis['details']['momentum']
                mfi = analysis['details']['mfi']
                volume = analysis['details']['volume']
                
                if ideal_dir == 'LONG':
                    expected_trend = 1
                    expected_momentum = momentum > 100
                    if bar_idx > 0:
                        expected_mfi = mfi >= analysis['details']['mfi_hhv_prev']
                        expected_vol = volume >= analysis['details']['vol_hhv_prev'] * 0.8
                        
                        analysis['details']['expected_trend'] = expected_trend
                        analysis['details']['trend_ok'] = trend == expected_trend
                        analysis['details']['momentum_ok'] = expected_momentum
                        analysis['details']['mfi_ok'] = expected_mfi
                        analysis['details']['volume_ok'] = expected_vol
                        
                        if trend != expected_trend:
                            analysis['reason'] = 'TREND_MISMATCH'
                        elif not expected_momentum:
                            analysis['reason'] = 'MOMENTUM_MISMATCH'
                        elif not expected_mfi:
                            analysis['reason'] = 'MFI_MISMATCH'
                        elif not expected_vol:
                            analysis['reason'] = 'VOLUME_MISMATCH'
                        else:
                            analysis['reason'] = 'BREAKOUT_MISMATCH'
                            
                elif ideal_dir == 'SHORT':
                    expected_trend = -1
                    expected_momentum = momentum < 100
                    if bar_idx > 0:
                        expected_mfi = mfi <= analysis['details']['mfi_llv_prev']
                        expected_vol = volume >= analysis['details']['vol_hhv_prev'] * 0.8
                        
                        analysis['details']['expected_trend'] = expected_trend
                        analysis['details']['trend_ok'] = trend == expected_trend
                        analysis['details']['momentum_ok'] = expected_momentum
                        analysis['details']['mfi_ok'] = expected_mfi
                        analysis['details']['volume_ok'] = expected_vol
                        
                        if trend != expected_trend:
                            analysis['reason'] = 'TREND_MISMATCH'
                        elif not expected_momentum:
                            analysis['reason'] = 'MOMENTUM_MISMATCH'
                        elif not expected_mfi:
                            analysis['reason'] = 'MFI_MISMATCH'
                        elif not expected_vol:
                            analysis['reason'] = 'VOLUME_MISMATCH'
                        else:
                            analysis['reason'] = 'BREAKOUT_MISMATCH'
    else:
        analysis['reason'] = 'BAR_NOT_FOUND'
    
    return analysis


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ARS TREND v2 - UYUMSUZ SİNYALLERİN DERİN ANALİZİ")
    print("=" * 80)
    
    # 1. Veri yükle
    print("\n[1] Veriler yükleniyor...")
    ideal_df, price_df = load_data()
    print(f"    IdealData: {len(ideal_df)} işlem")
    print(f"    Fiyat verisi: {len(price_df)} bar")
    
    # 2. Python stratejisini çalıştır
    print("\n[2] Python stratejisi çalıştırılıyor...")
    python_df, strategy = run_python_strategy_with_debug(price_df)
    print(f"    Python sinyalleri: {len(python_df)}")
    
    # 3. Uyumsuzlukları bul
    print("\n[3] Uyumsuzluklar tespit ediliyor...")
    
    mismatches = []
    ideal_entries = ideal_df[ideal_df['Direction'].isin(['LONG', 'SHORT'])].copy()
    python_entries = python_df[python_df['signal'].isin(['LONG', 'SHORT'])].copy()
    
    for idx, ideal_row in ideal_entries.iterrows():
        ideal_time = ideal_row['Time']
        ideal_dir = ideal_row['Direction']
        
        # 1 dakika tolerans ile ara
        time_mask = (python_entries['time'] >= ideal_time - timedelta(minutes=1)) & \
                    (python_entries['time'] <= ideal_time + timedelta(minutes=1))
        
        matching = python_entries[time_mask & (python_entries['signal'] == ideal_dir)]
        
        if len(matching) == 0:
            mismatches.append(ideal_row)
    
    print(f"    Toplam uyumsuzluk: {len(mismatches)}")
    
    # 4. Her uyumsuzluğu analiz et
    print("\n[4] Uyumsuzluklar analiz ediliyor...")
    print("=" * 80)
    
    reason_counts = {}
    detailed_analyses = []
    
    for i, mismatch in enumerate(mismatches):
        analysis = analyze_mismatch(mismatch, price_df, strategy, python_df)
        detailed_analyses.append(analysis)
        
        reason = analysis['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    # 5. Özet
    print("\n" + "=" * 80)
    print("UYUMSUZLUK NEDENLERİ ÖZETİ")
    print("=" * 80)
    
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / len(mismatches) * 100
        print(f"  {reason:30s} : {count:3d} ({pct:5.1f}%)")
    
    # 6. Detaylı analiz
    print("\n" + "=" * 80)
    print("DETAYLI ANALİZ (Tüm Uyumsuzluklar)")
    print("=" * 80)
    
    for i, analysis in enumerate(detailed_analyses):
        print(f"\n--- Uyumsuzluk #{i+1} ---")
        print(f"  Zaman     : {analysis['ideal_time']}")
        print(f"  Yön       : {analysis['ideal_dir']}")
        print(f"  Fiyat     : {analysis['ideal_price']:.2f}")
        print(f"  NEDEN     : {analysis['reason']}")
        
        details = analysis['details']
        
        if 'bar_idx' in details:
            print(f"  Bar Index : {details['bar_idx']}")
            print(f"  Trend     : {details.get('trend', 'N/A')} (beklenen: {details.get('expected_trend', 'N/A')})")
            print(f"  Momentum  : {details.get('momentum', 0):.2f} (>100: {details.get('momentum_ok', 'N/A')})")
            print(f"  MFI       : {details.get('mfi', 0):.2f} (OK: {details.get('mfi_ok', 'N/A')})")
            print(f"  Volume    : {details.get('volume', 0):.0f} (OK: {details.get('volume_ok', 'N/A')})")
            print(f"  Seans     : {details.get('time_of_day', 'N/A')} (içinde: {details.get('is_seans_icinde', 'N/A')})")
            print(f"  Warmup    : {details.get('in_warmup', 'N/A')}")
            print(f"  Vade Sonu : {details.get('is_vade_sonu', 'N/A')}")
            print(f"  Arefe     : {details.get('is_arefe', 'N/A')}")
        
        if details.get('nearby_python_signals'):
            print(f"  Yakın Python Sinyalleri:")
            for ps in details['nearby_python_signals'][:3]:
                print(f"    - {ps['time']} | {ps['signal']} | {ps.get('price', 0):.2f}")
