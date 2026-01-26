# -*- coding: utf-8 -*-
"""
ARS Trend v2 - İndikatör Doğrulama Testi
ideal_ars_v2_data.csv dosyası ile Python hesaplamalarını karşılaştırır.
"""

import sys
import io
import pandas as pd
import numpy as np
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategies.ars_trend_v2 import ARSTrendStrategyV2, StrategyConfigV2

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def load_data():
    # 1. Referans İndikatör Verisi (IdealData'dan export)
    try:
        df_ind = pd.read_csv("d:/Projects/IdealQuant/data/ideal_ars_v2_data.csv", sep=';')
        # Sütun isimleri: Date;Time;Close;ARS;Momentum;HHV;LLV;RSI
        # Bazı CSV'lerde boşluk olabilir
        df_ind.columns = [c.strip() for c in df_ind.columns]
        
        # Sayısal dönüştürme (Virgül/Nokta)
        for col in ['Close', 'ARS', 'Momentum', 'HHV', 'LLV', 'RSI']:
            if col in df_ind.columns:
                if df_ind[col].dtype == object:
                    df_ind[col] = df_ind[col].str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
        
        # DateTime oluştur
        df_ind['DateTime'] = pd.to_datetime(df_ind['Date'] + ' ' + df_ind['Time'], format='%d.%m.%Y %H:%M', errors='coerce')
        df_ind = df_ind.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
        
    except Exception as e:
        print(f"Referans veri yükleme hatası: {e}")
        return None, None

    # 2. Ham Fiyat Verisi (Tüm geçmiş)
    try:
        df_raw = pd.read_csv("d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv", sep=';', decimal=',', encoding='cp1254', low_memory=False)
        df_raw.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
        df_raw['DateTime'] = pd.to_datetime(df_raw['Tarih'] + ' ' + df_raw['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    except Exception as e:
        print(f"Fiyat verisi yükleme hatası: {e}")
        return None, None
        
    return df_ind, df_raw

def check_metric(name, py_series, id_series, tolerance=0.01):
    # NaN değerleri temizle
    mask = ~np.isnan(py_series) & ~np.isnan(id_series)
    
    if np.sum(mask) == 0:
        print(f"{name}: Karşılaştırılacak veri yok (Hepsi NaN)")
        return False
        
    diff = np.abs(py_series[mask] - id_series[mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    status = "✅" if max_diff < tolerance else "❌"
    print(f"{name:<10} | Max Fark: {max_diff:.6f} | Ort Fark: {mean_diff:.6f} | {status}")
    
    if max_diff >= tolerance:
        # En büyük farkın olduğu yeri bul
        idx = np.argmax(diff)
        print(f"    -> En büyük fark index {idx}: Py={py_series[mask][idx]:.4f} vs ID={id_series[mask][idx]:.4f}")
        return False
    return True

def run_test():
    print("=" * 60)
    print("ARS Trend v2 - İndikatör Uyumu Testi")
    print("=" * 60)
    
    df_ind, df_raw = load_data()
    if df_ind is None or df_raw is None: return
    
    print(f"Referans Veri : {len(df_ind)} bar (Son 5000)")
    print(f"Fiyat Verisi  : {len(df_raw)} bar (Tamamı)")
    
    # Eşleştirme (Zaman damgasına göre)
    # df_ind'deki zamanlar df_raw'da nerede?
    common_dates = df_ind['DateTime']
    
    # df_raw'ı filtrele
    df_raw_subset = df_raw[df_raw['DateTime'].isin(common_dates)].copy()
    
    # İndikatör hesaplaması için GERİYE DÖNÜK veriye ihtiyaç var
    # Bu yüzden df_raw'ın tamamını kullanıp, sonradan kesişimi almalıyız
    # Ancak referans veri sadece son 5000 bar. 
    # Python stratejisi tüm veriyi işlerse 190.000 bar sürer.
    # Stratejiyi sadece son 6000 bar için çalıştıralım (warmup dahil)
    
    last_date = df_ind['DateTime'].max()
    first_date = df_ind['DateTime'].min()
    
    # Tüm veriyi kullan (EMA için uzun geçmiş gerekli)
    df_calc = df_raw.copy()
    
    print(f"Hesaplama Verisi: {len(df_calc)} bar (Tüm veri)")
    
    # Strateji sınıfını hazırla
    # Parametreler: Export scripti ile AYNI OLMALI
    # ARS: 3, 10, 0.5, 0.002, 0.015
    # Mom: 5, Breakout: 10
    config = StrategyConfigV2(
        ars_ema_period = 3,
        ars_atr_period = 10,
        ars_atr_mult = 0.5,
        ars_min_band = 0.002,
        ars_max_band = 0.015,
        momentum_period = 5,
        breakout_period = 10,
        rsi_period = 14
    )
    
    opens = df_calc['Acilis'].values.tolist()
    highs = df_calc['Yuksek'].values.tolist()
    lows = df_calc['Dusuk'].values.tolist()
    closes = df_calc['Kapanis'].values.tolist()
    typical = ((df_calc['Yuksek'] + df_calc['Dusuk'] + df_calc['Kapanis']) / 3).tolist()
    times = df_calc['DateTime'].tolist()
    
    # Strateji nesnesi (Init sırasında indikatörleri hesaplar)
    strategy = ARSTrendStrategyV2(opens, highs, lows, closes, typical, times, config)
    
    # Karşılaştırma için verileri hizala
    # df_calc['DateTime'] ile df_ind['DateTime'] eşleşmeli
    
    # df_ind'deki her satır için strategy'deki karşılığı bul
    calc_map = {t: i for i, t in enumerate(times)}
    
    py_ars = []
    py_mom = []
    py_hhv = []
    py_llv = []
    py_rsi = []
    
    id_ars = []
    id_mom = []
    id_hhv = []
    id_llv = []
    id_rsi = []
    
    for i, row in df_ind.iterrows():
        dt = row['DateTime']
        if dt in calc_map:
            idx = calc_map[dt]
            
            # Python Değerleri
            py_ars.append(strategy.ars[idx])
            py_mom.append(strategy.momentum[idx])
            py_hhv.append(strategy.hhv[idx])
            py_llv.append(strategy.llv[idx])
            py_rsi.append(strategy.rsi[idx])
            
            # Ideal Değerleri
            id_ars.append(row['ARS'])
            id_mom.append(row['Momentum'])
            id_hhv.append(row['HHV'])
            id_llv.append(row['LLV'])
            id_rsi.append(row['RSI'])
            
    # NumPy array dönüşümü
    py_ars = np.array(py_ars)
    py_mom = np.array(py_mom)
    py_hhv = np.array(py_hhv)
    py_llv = np.array(py_llv)
    py_rsi = np.array(py_rsi)
    
    id_ars = np.array(id_ars)
    id_mom = np.array(id_mom)
    id_hhv = np.array(id_hhv)
    id_llv = np.array(id_llv)
    id_rsi = np.array(id_rsi)
    
    print("-" * 60)
    print("SONUÇLAR (Tolerans: 0.01)")
    print("-" * 60)
    
    check_metric("ARS", py_ars, id_ars, 0.05) # ARS biraz daha hassas olabilir
    check_metric("Momentum", py_mom, id_mom)
    check_metric("HHV", py_hhv, id_hhv)
    check_metric("LLV", py_llv, id_llv)
    check_metric("RSI", py_rsi, id_rsi, 0.1) # RSI ufak farklar olabilir (wilder smoothing)
    
if __name__ == "__main__":
    run_test()
