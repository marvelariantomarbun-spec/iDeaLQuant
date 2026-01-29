# -*- coding: utf-8 -*-
"""
IdealQuant - Optimizasyon Motoru (Grid Search)
Ryzen 9 9950X (32 thread) için optimize edilmiş paralel motor.
"""

import itertools
import time
from typing import List, Dict, Any, Callable, Type
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from dataclasses import dataclass, fields, is_dataclass

@dataclass
class OptimizationResult:
    params: Dict[str, Any]
    net_profit: float
    trade_count: int
    profit_factor: float
    win_rate: float
    max_drawdown: float
    return_on_dd: float

class GridOptimizer:
    def __init__(self, strategy_class: Type, data_df: pd.DataFrame, param_grid: Dict[str, List[any]]):
        """
        Args:
            strategy_class: Backtest yapılacak strateji sınıfı (Strategy1 veya ARSTrendStrategyV2)
            data_df: OHLCV verisini içeren DataFrame
            param_grid: Optimize edilecek parametrelerin ve değerlerinin sözlüğü
                        Örn: {'ars_k': [0.01, 0.02], 'period': [5, 10]}
        """
        self.strategy_class = strategy_class
        self.data_df = data_df
        self.param_grid = param_grid
        self.combinations = self._generate_combinations()
        
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations

    @staticmethod
    def _run_single_backtest(strategy_class, data_df, params) -> Dict[str, Any]:
        """
        Bu metod statik olmalı çünkü ProcessPoolExecutor tarafından serialize edilecek.
        """
        # Verileri hazırla
        opens = data_df['Acilis'].values.tolist()
        highs = data_df['Yuksek'].values.tolist()
        lows = data_df['Dusuk'].values.tolist()
        closes = data_df['Kapanis'].values.tolist()
        typical = ((data_df['Yuksek'] + data_df['Dusuk'] + data_df['Kapanis']) / 3).values.tolist()
        
        # Zaman verisi eğer strateji bekliyorsa (V2 gibi)
        times = None
        if 'DateTime' in data_df.columns:
            times = data_df['DateTime'].tolist()

        # Config oluştur (Dataclass ise)
        # Stratejinin beklediği config sınıfını bul (Strategy1Config veya StrategyConfigV2)
        # Basitlik için şimdilik stratejilere params sözlüğünü doğrudan geçebilen bir wrapper ekleyeceğiz
        # Veya strateji sınıfının __init__ metodunu buna göre güncelleyeceğiz.
        
        try:
            # Stratejiyi başlat
            # Not: Her worker kendi içinde nesnesini oluşturur
            if times:
                strategy = strategy_class(opens, highs, lows, closes, typical, times=times, config_dict=params)
            else:
                strategy = strategy_class(opens, highs, lows, closes, typical, config_dict=params)
                
            # Backtest Döngüsü
            current_position = "FLAT"
            entry_price = 0.0
            extreme_price = 0.0
            trades = []
            
            n = len(closes)
            for i in range(1, n):
                # Sinyal al
                # V2 stratejisi entry_price ve extreme_price bekliyor
                if hasattr(strategy, 'get_signal'):
                    # Strateji tipine göre dinamik çağrı
                    import inspect
                    sig = inspect.signature(strategy.get_signal)
                    if 'entry_price' in sig.parameters:
                        signal = strategy.get_signal(i, current_position, entry_price, extreme_price)
                    else:
                        signal = strategy.get_signal(i, current_position)
                else:
                    continue

                # Trailing stop için extreme güncelle
                if current_position == "LONG":
                    extreme_price = max(extreme_price, highs[i])
                elif current_position == "SHORT":
                    extreme_price = min(extreme_price, lows[i])

                # Sinyal İşle (Signal enum veya string)
                # Signal enum ise .value kullan, string ise doğrudan kullan
                sig_val = signal.value if hasattr(signal, 'value') else signal
                
                if sig_val == "A" and current_position != "LONG":
                    if current_position == "SHORT":
                        trades.append(entry_price - closes[i]) # Short P&L
                    current_position = "LONG"
                    entry_price = closes[i]
                    extreme_price = closes[i]
                elif sig_val == "S" and current_position != "SHORT":
                    if current_position == "LONG":
                        trades.append(closes[i] - entry_price) # Long P&L
                    current_position = "SHORT"
                    entry_price = closes[i]
                    extreme_price = closes[i]
                elif sig_val == "F" and current_position != "FLAT":
                    pnl = (closes[i] - entry_price) if current_position == "LONG" else (entry_price - closes[i])
                    trades.append(pnl)
                    current_position = "FLAT"
                    entry_price = 0
                    extreme_price = 0

            # Metrikleri hesapla
            net_profit = sum(trades)
            trade_count = len(trades)
            winners = [t for t in trades if t > 0]
            losers = [t for t in trades if t < 0]
            
            win_rate = (len(winners) / trade_count * 100) if trade_count > 0 else 0
            gross_profit = sum(winners)
            gross_loss = abs(sum(losers))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
            
            # Sonucu dön
            res = params.copy()
            res.update({
                'NetProfit': net_profit,
                'TradeCount': trade_count,
                'ProfitFactor': profit_factor,
                'WinRate': win_rate
            })
            return res
            
        except Exception as e:
            return {'error': str(e), 'params': params}

    def run(self, workers: int = 24) -> pd.DataFrame:
        print(f"Optimizasyon baslatiliyor: {len(self.combinations)} kombinasyon...")
        print(f"Kullanilan worker sayisi: {workers}")
        
        start_time = time.time()
        
        # Paralel işleme
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # map kullanmak daha verimli olabilir
            futures = [executor.submit(self._run_single_backtest, self.strategy_class, self.data_df, combo) 
                       for combo in self.combinations]
            
            for i, future in enumerate(futures):
                results.append(future.result())
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / len(self.combinations) * 100
                    eta = (elapsed / (i + 1)) * (len(self.combinations) - (i + 1))
                    print(f"Progress: %{progress:.1f} | ETA: {eta:.0f}s | Speed: {(i+1)/elapsed:.1f} combo/sec")

        df_results = pd.DataFrame(results)
        
        total_time = time.time() - start_time
        print(f"Optimizasyon tamamlandi! Toplam sure: {total_time:.1f} saniye.")
        print(f"Ortalama hiz: {len(self.combinations)/total_time:.1f} kombinasyon/saniye.")
        
        return df_results.sort_values('NetProfit', ascending=False)
