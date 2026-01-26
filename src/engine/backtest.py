# -*- coding: utf-8 -*-
"""
IdealQuant - Backtest Motoru
Strateji ve filtre entegrasyonu ile backtest
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from strategies.ars_trend import ARSTrendStrategy, StrategyConfig, Signal
from filters.yatay_filtre import YatayFiltre


@dataclass
class Trade:
    """Tek bir işlem kaydı"""
    entry_bar: int
    entry_price: float
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_time: datetime = None
    exit_reason: str = ""
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    
    def close(self, bar: int, price: float, time: datetime, reason: str):
        """Pozisyonu kapat"""
        self.exit_bar = bar
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        self.bars_held = bar - self.entry_bar
        
        if self.direction == "LONG":
            self.pnl = price - self.entry_price
            self.pnl_pct = (price / self.entry_price - 1) * 100
        else:  # SHORT
            self.pnl = self.entry_price - price
            self.pnl_pct = (self.entry_price / price - 1) * 100


@dataclass
class BacktestResult:
    """Backtest sonuçları"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Performans metrikleri
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    def calculate_metrics(self):
        """Performans metriklerini hesapla"""
        if not self.trades:
            return
        
        self.total_trades = len(self.trades)
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        
        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_pnl_pct = sum(t.pnl_pct for t in self.trades)
        
        if wins:
            self.avg_win = np.mean([t.pnl for t in wins])
        if losses:
            self.avg_loss = np.mean([abs(t.pnl) for t in losses])
        
        total_win = sum(t.pnl for t in wins) if wins else 0
        total_loss = sum(abs(t.pnl) for t in losses) if losses else 0
        self.profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # Drawdown hesapla
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for val in self.equity_curve:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown = max_dd
            self.max_drawdown_pct = (max_dd / peak * 100) if peak > 0 else 0


class Backtester:
    """
    Backtest motoru
    
    YatayFiltre + ARSTrendStrategy entegrasyonu
    """
    
    def __init__(self,
                 opens: List[float],
                 highs: List[float],
                 lows: List[float],
                 closes: List[float],
                 typical: List[float],
                 timestamps: Optional[List[datetime]] = None,
                 strategy_config: Optional[StrategyConfig] = None,
                 use_yatay_filtre: bool = True):
        """
        Args:
            opens, highs, lows, closes, typical: OHLC verileri
            timestamps: Bar zamanları
            strategy_config: Strateji konfigürasyonu
            use_yatay_filtre: Yatay filtre kullanılsın mı?
        """
        self.n = len(closes)
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.typical = typical
        self.timestamps = timestamps or [datetime.now()] * self.n
        
        # Strateji
        self.strategy = ARSTrendStrategy(
            opens, highs, lows, closes, typical, 
            strategy_config
        )
        
        # Yatay filtre
        self.use_yatay_filtre = use_yatay_filtre
        if use_yatay_filtre:
            self.yatay_filtre = YatayFiltre(closes, highs, lows, typical)
        else:
            self.yatay_filtre = None
    
    def run(self, start_bar: int = 50) -> BacktestResult:
        """
        Backtest çalıştır
        
        Args:
            start_bar: Başlangıç bar'ı (warmup için)
        
        Returns:
            BacktestResult
        """
        result = BacktestResult()
        
        # Pozisyon durumu
        position = "FLAT"  # "LONG", "SHORT", "FLAT"
        entry_price = 0.0
        extreme_price = 0.0  # max (LONG) veya min (SHORT)
        current_trade: Optional[Trade] = None
        
        # Equity curve (başlangıç = 0)
        equity = 0.0
        result.equity_curve = [0.0] * self.n
        
        for i in range(start_bar, self.n):
            # Yatay filtre kontrolü
            if self.use_yatay_filtre:
                trade_allowed = self.yatay_filtre.islem_izni(i)
            else:
                trade_allowed = True
            
            # Sinyal al
            signal = self.strategy.get_signal(i, position, entry_price, extreme_price)
            
            # Sinyal işleme
            if signal == Signal.FLAT and position != "FLAT":
                # Pozisyon kapat
                if current_trade:
                    exit_reason = "signal_flat"
                    if position == "LONG":
                        _, exit_reason = self.strategy.check_long_exit(i, entry_price, extreme_price)
                    elif position == "SHORT":
                        _, exit_reason = self.strategy.check_short_exit(i, entry_price, extreme_price)
                    
                    current_trade.close(i, self.closes[i], self.timestamps[i], exit_reason)
                    result.trades.append(current_trade)
                    equity += current_trade.pnl
                    
                position = "FLAT"
                current_trade = None
                entry_price = 0.0
                extreme_price = 0.0
            
            elif signal == Signal.LONG and position == "FLAT" and trade_allowed:
                # LONG aç
                position = "LONG"
                entry_price = self.closes[i]
                extreme_price = self.closes[i]
                current_trade = Trade(
                    entry_bar=i,
                    entry_price=entry_price,
                    entry_time=self.timestamps[i],
                    direction="LONG"
                )
            
            elif signal == Signal.SHORT and position == "FLAT" and trade_allowed:
                # SHORT aç
                position = "SHORT"
                entry_price = self.closes[i]
                extreme_price = self.closes[i]
                current_trade = Trade(
                    entry_bar=i,
                    entry_price=entry_price,
                    entry_time=self.timestamps[i],
                    direction="SHORT"
                )
            
            # Yatay piyasaya geçiş - açık pozisyonu kapat
            if not trade_allowed and position != "FLAT" and current_trade:
                current_trade.close(i, self.closes[i], self.timestamps[i], "yatay_filtre")
                result.trades.append(current_trade)
                equity += current_trade.pnl
                
                position = "FLAT"
                current_trade = None
                entry_price = 0.0
                extreme_price = 0.0
            
            # Extreme price güncelle
            if position == "LONG":
                extreme_price = max(extreme_price, self.closes[i])
            elif position == "SHORT":
                extreme_price = min(extreme_price, self.closes[i])
            
            # Equity curve güncelle
            unrealized_pnl = 0.0
            if position == "LONG":
                unrealized_pnl = self.closes[i] - entry_price
            elif position == "SHORT":
                unrealized_pnl = entry_price - self.closes[i]
            
            result.equity_curve[i] = equity + unrealized_pnl
        
        # Son pozisyonu kapat
        if position != "FLAT" and current_trade:
            current_trade.close(self.n - 1, self.closes[-1], self.timestamps[-1], "end_of_data")
            result.trades.append(current_trade)
        
        # Metrikleri hesapla
        result.calculate_metrics()
        
        return result


def print_backtest_report(result: BacktestResult, title: str = "BACKTEST SONUÇLARI"):
    """Backtest sonuçlarını yazdır"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
    print(f"\n  Toplam İşlem      : {result.total_trades}")
    print(f"  Karlı İşlem       : {result.winning_trades}")
    print(f"  Zararlı İşlem     : {result.losing_trades}")
    print(f"  Kazanma Oranı     : %{result.win_rate:.1f}")
    print("-" * 60)
    print(f"  Toplam K/Z        : {result.total_pnl:.2f}")
    print(f"  Toplam K/Z %      : %{result.total_pnl_pct:.2f}")
    print(f"  Ortalama Kazanç   : {result.avg_win:.2f}")
    print(f"  Ortalama Kayıp    : {result.avg_loss:.2f}")
    print(f"  Profit Factor     : {result.profit_factor:.2f}")
    print("-" * 60)
    print(f"  Max Drawdown      : {result.max_drawdown:.2f}")
    print(f"  Max Drawdown %    : %{result.max_drawdown_pct:.2f}")
    print("=" * 60)
