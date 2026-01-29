# -*- coding: utf-8 -*-
"""
IdealQuant - Optimizasyon Motoru Testi
Ryzen 9 9950X Performans Doğrulama
"""

import pandas as pd
import numpy as np
import time
from src.optimization.optimizer import GridOptimizer
from src.strategies.strategy_1 import Strategy1

def run_test():
    # 1. Veri Hazırla (Sentetik veya Mevcut)
    # Gerçek veri kullanalım (XU030 1DK - proje içi)
    try:
        data_path = 'd:\\Projects\\IdealQuant\\data\\VIP_X030T_1dk_.csv'
        df = pd.read_csv(data_path, sep=';', decimal=',', encoding='cp1254')
        # Sütun isimlerini kontrol et ve uyumlu hale getir
        if 'Acilis' not in df.columns:
            df.columns = ['tarih', 'saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'ortalama', 'Hacim', 'lot']
        if 'DateTime' not in df.columns:
            df['DateTime'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'], format='%d.%m.%Y %H:%M:%S')
        print(f"Data yuklendi: {len(df)} bar.")
    except Exception as e:
        print(f"Veri yuklenemedi, sentetik veri olusturuluyor: {e}")
        n = 10000
        df = pd.DataFrame({
            'Acilis': np.random.randn(n).cumsum() + 100,
            'Yuksek': np.random.randn(n).cumsum() + 101,
            'Dusuk': np.random.randn(n).cumsum() + 99,
            'Kapanis': np.random.randn(n).cumsum() + 100,
            'Hacim': np.random.randint(100, 1000, n)
        })

    # 2. Parametre Grid Tanımla (Strateji 1 için)
    # AHLT'deki gibi geniş bir aralık seçelim
    param_grid = {
        'ars_k': [0.005, 0.010, 0.015],
        'yatay_esik': [5, 10, 15],
        'min_onay_skoru': [4, 5, 6],
        'exit_skoru': [3, 4]
    }
    # 3 * 3 * 3 * 2 = 54 kombinasyon
    
    # Performans testi için daha büyük bir grid
    full_test = False
    if full_test:
        param_grid = {
            'ars_k': np.linspace(0.005, 0.030, 5).tolist(),
            'yatay_esik': [5, 10, 15, 20],
            'min_onay_skoru': [4, 5, 6],
            'exit_skoru': [3, 4, 5],
            'qqef_period': [10, 14, 21]
        }
        # 5 * 4 * 3 * 3 * 3 = 540 kombinasyon
    
    # 3. Optimizer Başlat
    optimizer = GridOptimizer(Strategy1, df, param_grid)
    
    # DEBUG: Tek bir testi senkron çalıştırıp hatayı görelim
    print("\n--- DEBUG: Tekli Test Calistiriliyor ---")
    single_res = optimizer._run_single_backtest(Strategy1, df, optimizer.combinations[0])
    print(f"Tekli Test Sonucu: {single_res}")
    
    if 'error' in single_res:
        print(f"HATA BULUNDU: {single_res['error']}")
        return

    # 4. Çalıştır
    start_time = time.time()
    results = optimizer.run(workers=24)
    end_time = time.time()
    
    if 'error' in results.columns:
        print("\n--- HATALAR TESPİT EDİLDİ ---")
        print(results[results['error'].notna()][['params', 'error']].head())
    
    # 5. Sonuçları Göster
    print("\n--- EN İYİ 10 SONUÇ ---")
    print(results.head(10).to_string())
    
    print(f"\n[RESULT] Toplam sure: {end_time - start_time:.2f} saniye.")
    print(f"[PERF] Saniyede test edilen bar: {(len(df) * len(optimizer.combinations)) / (end_time - start_time) / 1e6:.2f} Milyon")

if __name__ == "__main__":
    run_test()
