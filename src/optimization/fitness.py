# -*- coding: utf-8 -*-
"""
Extended Fitness Module
=======================
Çok faktörlü fitness hesaplama fonksiyonları.
Tüm optimizer'lar tarafından kullanılır.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class FitnessConfig:
    """Fitness hesaplama konfigürasyonu"""
    # Sermaye (DD oranı hesaplamak için)
    initial_capital: float = 10000.0
    
    # İşlem sayısı limitleri
    min_trades: int = 20
    ideal_min_trades: int = 50
    max_trades: int = 500
    
    # Ortalama işlem karı limitleri (TL)
    min_avg_trade: float = 30.0
    ideal_avg_trade: float = 50.0
    
    # Risk limitleri
    max_dd_ratio: float = 0.20  # Sermayenin max %20'si
    ideal_calmar: float = 2.0   # Net Kar / Max DD
    
    # Ağırlıklar
    pf_bonus_weight: float = 0.1
    dd_penalty_weight: float = 0.5
    trade_count_weight: float = 0.3
    avg_trade_weight: float = 0.2


def calculate_extended_metrics(
    closes: np.ndarray,
    entry_prices: list,
    exit_prices: list,
    directions: list,  # 1 for long, -1 for short
    initial_capital: float = 10000.0
) -> Dict[str, float]:
    """
    Backtest sonuçlarından genişletilmiş metrikler hesapla.
    
    Args:
        closes: Kapanış fiyatları
        entry_prices: Giriş fiyatları listesi
        exit_prices: Çıkış fiyatları listesi
        directions: İşlem yönleri (1=long, -1=short)
        initial_capital: Başlangıç sermayesi
    
    Returns:
        Dict: Tüm metrikler
    """
    trades = len(entry_prices)
    
    if trades == 0:
        return {
            'net_profit': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'pf': 0,
            'max_dd': 0,
            'trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'risk_reward': 0,
            'calmar_ratio': 0,
            'fitness': 0
        }
    
    # PnL hesapla
    pnls = []
    for i in range(trades):
        if directions[i] == 1:
            pnl = exit_prices[i] - entry_prices[i]
        else:
            pnl = entry_prices[i] - exit_prices[i]
        pnls.append(pnl)
    
    pnls = np.array(pnls)
    
    # Temel metrikler
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = np.abs(np.sum(pnls[pnls < 0]))
    net_profit = gross_profit - gross_loss
    
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    
    # Win/Loss
    win_count = np.sum(pnls > 0)
    loss_count = np.sum(pnls < 0)
    win_rate = (win_count / trades) * 100 if trades > 0 else 0
    
    # Ortalamalar
    avg_trade = net_profit / trades if trades > 0 else 0
    avg_win = np.mean(pnls[pnls > 0]) if win_count > 0 else 0
    avg_loss = np.abs(np.mean(pnls[pnls < 0])) if loss_count > 0 else 0
    risk_reward = (avg_win / avg_loss) if avg_loss > 0 else 999.0
    
    # Equity curve ve Max DD
    equity = initial_capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = np.max(dd) if len(dd) > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = (net_profit / max_dd) if max_dd > 0 else 999.0
    
    return {
        'net_profit': net_profit,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'pf': pf,
        'max_dd': max_dd,
        'trades': trades,
        'win_count': int(win_count),
        'loss_count': int(loss_count),
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'risk_reward': risk_reward,
        'calmar_ratio': calmar_ratio
    }


def calculate_fitness(
    metrics: Dict[str, float],
    config: Optional[FitnessConfig] = None
) -> float:
    """
    Genişletilmiş çok faktörlü fitness hesapla.
    
    Args:
        metrics: calculate_extended_metrics çıktısı
        config: FitnessConfig (opsiyonel)
    
    Returns:
        float: Fitness skoru
    """
    if config is None:
        config = FitnessConfig()
    
    net_profit = metrics.get('net_profit', 0)
    pf = metrics.get('pf', 0)
    max_dd = metrics.get('max_dd', 0)
    trades = metrics.get('trades', 0)
    avg_trade = metrics.get('avg_trade', 0)
    calmar = metrics.get('calmar_ratio', 0)
    win_rate = metrics.get('win_rate', 0)
    
    # Negatif veya sıfır kar = düşük fitness
    if net_profit <= 0:
        return net_profit * 0.5  # Negatif fitness
    
    # Base: Net Profit
    fitness = net_profit
    
    # 1. Profit Factor Bonus (PF > 1.5)
    if pf > 1.5:
        pf_bonus = (pf - 1) * config.pf_bonus_weight
        fitness *= (1 + min(pf_bonus, 0.5))  # Max %50 bonus
    elif pf < 1.0:
        fitness *= 0.5  # PF < 1 = kayıp veren strateji
    
    # 2. Max DD Ceza
    if max_dd > 0:
        dd_ratio = max_dd / config.initial_capital
        if dd_ratio > config.max_dd_ratio:
            # Sermayenin %20'sinden fazla DD = ağır ceza
            fitness *= (1 - min(config.dd_penalty_weight, dd_ratio))
        else:
            # Kabul edilebilir DD
            fitness *= (1 - dd_ratio * 0.3)
    
    # 3. İşlem Sayısı Kontrolü
    if trades < config.min_trades:
        fitness *= 0.3  # Çok az işlem - güvenilmez
    elif trades < config.ideal_min_trades:
        ratio = trades / config.ideal_min_trades
        fitness *= (0.7 + 0.3 * ratio)  # Kademeli azalma
    elif trades > config.max_trades:
        # Çok fazla işlem - komisyon riski
        excess = (trades - config.max_trades) / config.max_trades
        fitness *= (1 - min(0.3, excess * 0.1))
    
    # 4. Ortalama İşlem Karı
    if avg_trade < config.min_avg_trade:
        fitness *= 0.5  # Komisyon yerse zarar
    elif avg_trade < config.ideal_avg_trade:
        ratio = avg_trade / config.ideal_avg_trade
        fitness *= (0.8 + 0.2 * ratio)
    
    # 5. Calmar Ratio (Risk-Adjusted Return)
    if calmar > config.ideal_calmar:
        fitness *= 1.15  # İyi risk/return bonusu
    elif calmar < 1.0:
        fitness *= 0.8  # Kötü risk/return
    
    # 6. Win Rate Bonus (opsiyonel - çok ağır değil)
    if win_rate > 50:
        fitness *= (1 + (win_rate - 50) * 0.002)  # Max %10 bonus
    
    return fitness


def quick_fitness(
    net_profit: float,
    pf: float,
    max_dd: float,
    trades: int,
    win_count: int = 0,
    initial_capital: float = 10000.0
) -> float:
    """
    Hızlı fitness hesaplama (backtest içinde kullanım için).
    Tam metrik hesaplaması gerekmediğinde kullanılır.
    """
    if net_profit <= 0:
        return net_profit * 0.5
    
    fitness = net_profit
    
    # PF bonus
    if pf > 1.5:
        fitness *= (1 + (pf - 1) * 0.1)
    elif pf < 1.0:
        fitness *= 0.5
    
    # DD ceza
    if max_dd > 0:
        dd_ratio = max_dd / initial_capital
        fitness *= (1 - min(0.5, dd_ratio))
    
    # Trade count
    if trades < 20:
        fitness *= 0.3
    elif trades < 50:
        fitness *= 0.7
    elif trades > 500:
        fitness *= 0.8
    
    # Avg trade
    avg_trade = net_profit / trades if trades > 0 else 0
    if avg_trade < 30:
        fitness *= 0.5
    elif avg_trade < 50:
        fitness *= 0.8
    
    return fitness


# Test
if __name__ == "__main__":
    # Örnek metrikler
    test_metrics = {
        'net_profit': 15000,
        'pf': 1.8,
        'max_dd': 2500,
        'trades': 120,
        'avg_trade': 125,
        'calmar_ratio': 6.0,
        'win_rate': 55
    }
    
    fitness = calculate_fitness(test_metrics)
    print(f"Test Fitness: {fitness:,.0f}")
    print(f"Quick Fitness: {quick_fitness(15000, 1.8, 2500, 120):,.0f}")
