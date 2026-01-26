# -*- coding: utf-8 -*-
"""
IdealQuant - Backtest Testi
ARS Trend stratejisi + YatayFiltre ile backtest
"""

import sys
import io

# Windows UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'd:/Projects/IdealQuant/src')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from engine.backtest import Backtester, print_backtest_report
from strategies.ars_trend import StrategyConfig


def load_data(csv_path: str) -> tuple:
    """OHLC verilerini yükle"""
    print(f"Veri yükleniyor: {csv_path}")
    
    df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254')
    df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
    
    # Timestamp oluştur
    timestamps = []
    for _, row in df.iterrows():
        try:
            dt = datetime.strptime(f"{row['Tarih']} {row['Saat']}", "%d.%m.%Y %H:%M:%S")
            timestamps.append(dt)
        except:
            timestamps.append(datetime.now())
    
    opens = df['Acilis'].values.astype(float)
    highs = df['Yuksek'].values.astype(float)
    lows = df['Dusuk'].values.astype(float)
    closes = df['Kapanis'].values.astype(float)
    typical = (highs + lows + closes) / 3
    
    print(f"  -> {len(closes)} bar yüklendi")
    print(f"  -> Tarih aralığı: {timestamps[0]} - {timestamps[-1]}")
    
    return opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(), typical.tolist(), timestamps


def main():
    print("\n" + "=" * 70)
    print("  IdealQuant - ARS Trend + YatayFiltre Backtest")
    print("=" * 70)
    
    # Veri yükle
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    opens, highs, lows, closes, typical, timestamps = load_data(csv_path)
    
    # 5DK konfigürasyonu
    config = StrategyConfig.for_timeframe(5)
    
    print("\n" + "-" * 70)
    print("Test 1: ARS Trend + YatayFiltre (YatayFiltre AÇIK)")
    print("-" * 70)
    
    # Backtest - YatayFiltre ile
    backtester_with_filter = Backtester(
        opens, highs, lows, closes, typical, timestamps,
        strategy_config=config,
        use_yatay_filtre=True
    )
    result_with = backtester_with_filter.run(start_bar=100)
    print_backtest_report(result_with, "ARS Trend (Filtreli)")
    
    print("\n" + "-" * 70)
    print("Test 2: ARS Trend (YatayFiltre KAPALI)")
    print("-" * 70)
    
    # Backtest - YatayFiltre olmadan
    backtester_no_filter = Backtester(
        opens, highs, lows, closes, typical, timestamps,
        strategy_config=config,
        use_yatay_filtre=False
    )
    result_without = backtester_no_filter.run(start_bar=100)
    print_backtest_report(result_without, "ARS Trend (Filtresiz)")

    print("\n" + "-" * 70)
    print("Test 3: Score Based Strateji (6 İndikatör)")
    print("-" * 70)
    
    # Score Based Strateji için Backtester'ı hafifçe modifiye etmek gerekebilir
    # Ancak şimdilik Backtester sınıfı ARSTrendStrategy'ye sıkı bağlı
    # Bu yüzden manuel olarak stratejiyi değiştirelim (dirty hack for test)
    
    from strategies.score_based import ScoreBasedStrategy
    
    backtester_score = Backtester(
        opens, highs, lows, closes, typical, timestamps,
        use_yatay_filtre=True
    )
    # Stratejiyi override et
    backtester_score.strategy = ScoreBasedStrategy(opens, highs, lows, closes, typical)
    
    result_score = backtester_score.run(start_bar=100)
    print_backtest_report(result_score, "Score Based (Filtreli)")
    
    # Karşılaştırma
    print("\n" + "=" * 90)
    print("  KARŞILAŞTIRMA")
    print("=" * 90)
    print(f"\n  {'Metrik':<25} | {'ARS (Filtreli)':<15} | {'ARS (Filtresiz)':<15} | {'Score Based':<15}")
    print("-" * 90)
    print(f"  {'Toplam İşlem':<25} | {result_with.total_trades:<15} | {result_without.total_trades:<15} | {result_score.total_trades:<15}")
    print(f"  {'Kazanma Oranı':<25} | %{result_with.win_rate:<14.1f} | %{result_without.win_rate:<14.1f} | %{result_score.win_rate:<14.1f}")
    print(f"  {'Toplam K/Z':<25} | {result_with.total_pnl:<15.2f} | {result_without.total_pnl:<15.2f} | {result_score.total_pnl:<15.2f}")
    print(f"  {'Profit Factor':<25} | {result_with.profit_factor:<15.2f} | {result_without.profit_factor:<15.2f} | {result_score.profit_factor:<15.2f}")
    print(f"  {'Max Drawdown':<25} | {result_with.max_drawdown:<15.2f} | {result_without.max_drawdown:<15.2f} | {result_score.max_drawdown:<15.2f}")
    print("=" * 90)
    
    # YatayFiltre etkisi
    if result_without.total_trades > 0:
        trade_reduction = (1 - result_with.total_trades / result_without.total_trades) * 100
        print(f"\n  YatayFiltre ile işlem sayısı: %{trade_reduction:.1f} azaldı")
    
    if result_with.profit_factor > result_without.profit_factor:
        print("  ✓ YatayFiltre profit factor'ü artırdı")
    else:
        print("  ✗ YatayFiltre profit factor'ü azalttı")


if __name__ == "__main__":
    main()
