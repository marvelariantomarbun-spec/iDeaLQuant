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
    initial_capital: float = 10000.0
    
    # İşlem maliyetleri (Puan bazlı, örn: 5.0 = 5 puan kayma+komisyon)
    commission: float = 0.0
    slippage: float = 0.0
    
    # İşlem sayısı limitleri (Overtrading engelleme)
    min_trades: int = 20
    ideal_min_trades: int = 50
    overtrading_limit: int = 1500  # 1500 işlemden sonrası ağır cezalı
    
    # Ortalama işlem karı (TL/Puan)
    min_avg_profit: float = 10.0   # En az 10 puan/TL kar kalmalı
    
    # Risk limitleri
    max_dd_ratio: float = 0.20
    
    # Ağırlıklar
    # Profit Factor (PF) limits
    min_pf: float = 1.5    # Kullanıcı isteği: en az 1.50
    max_pf: float = 3.0    # 3.0 üzeri genelde overfit/noise fitting

    # R^2 (Equity Curve Smoothness)
    min_r2: float = 0.85   # Regresyon katsayısı (0-1 arası)

def calculate_fitness(
    metrics: Dict[str, float],
    config: Optional[FitnessConfig] = None
) -> float:
    """
    Genişletilmiş çok faktörlü fitness hesapla.
    Kriterler:
    1. Net Kâr (Maliyet düşülmüş)
    2. Profit Factor (1.5 - 2.5 arası ideal)
    3. İşlem Sayısı (50 - 1000 arası ideal)
    4. Max Drawdown (%20 altı)
    5. R^2 (Equity Smoothness) - Varsa
    """
    if config is None:
        config = FitnessConfig()
    
    net_profit = metrics.get('net_profit', 0)
    pf = metrics.get('pf', 0)
    max_dd = metrics.get('max_dd', 0)
    trades = metrics.get('trades', 0)
def quick_fitness(
    net_profit: float,
    pf: float,
    max_dd: float,
    trades: int,
    sharpe: float = 0.0,
    win_count: int = 0,
    initial_capital: float = 10000.0,
    commission: float = 0.0,
    slippage: float = 0.0
) -> float:
    """Hızlı hesaplama (Optimizer döngüleri için)"""
    
    # Kâr ve Maliyet
    cost = trades * (commission + slippage)
    adj_profit = net_profit - cost
    
    # Zarar varsa direkt ele (negatif puan)
    if adj_profit <= 0: return -99999 + adj_profit
    
    score = adj_profit
    
    # --- PROFIT FACTOR FILTRESI (Kullanıcı: "1.5 altı olmaz, 2.0 üstü şüpheli") ---
    if pf < 1.3:
        score *= 0.1 # Ölü
    elif pf < 1.5:
        score *= 0.4 # Zayıf
    elif pf > 3.0: 
        score *= 0.8 # Aşırı yüksek PF (Overfit şüphesi) - Eskiden 2.5 idi, biraz gevşettik
    else:
        # 1.5 - 3.0 arası: En sevdiğimiz bölge
        score *= 1.2 
        
    # --- OVERTRADING FILTRESI ---
    if trades > 1500:
        score *= (1500 / trades) ** 2 # Çok sert düşüş
    elif trades < 50: # Alt limit
        score *= 0.5 # İstatistiksel olarak anlamsız
        
    # --- DRAWDOWN FILTRESI ---
    if max_dd > (initial_capital * 0.25):
        score *= 0.5
        
    # --- AVG PROFIT FILTRESI ---
    avg_pnl = adj_profit / trades if trades > 0 else 0
    if avg_pnl < 15.0: # En az 15 puan işlem başı kar
        score *= (avg_pnl / 15.0) 

    # --- SHARPE RATIO BONUSU ---
    if sharpe > 2.0:
        score *= 1.15 # %15 Bonus (Yüksek risk/getiri kalitesi)
    elif sharpe > 1.5:
        score *= 1.05 # %5 Bonus
    elif sharpe < 0.5:
        score *= 0.9 # Düşük Sharpe cezası

    return score

def calculate_sharpe(returns: np.array, risk_free=0.0, trades_per_year=252.0) -> float:
    """
    Yıllıklandırılmış Sharpe Ratio hesapla.

    Args:
        returns: İşlem bazlı getiri listesi (PnL veya % return)
        risk_free: Risksiz getiri oranı (varsayılan 0)
        trades_per_year: Yıllık ortalama işlem sayısı (varsayılan 252)
                         Günlük getiri için 252, trade bazlı için gerçek yıllık trade sayısı kullanın.

    Returns:
        float: Yıllıklandırılmış Sharpe Ratio
    """
    if len(returns) < 2: return 0.0
    
    excess_returns = returns - risk_free/trades_per_year
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0: return 0.0
    
    # Yıllıklandırma: Trade başına ortalama getiri / std * sqrt(yıllık trade sayısı)
    return (mean_excess / std_excess) * np.sqrt(trades_per_year)


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
