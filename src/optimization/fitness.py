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
    
    # 0. Maliyet Düşümü
    total_cost = trades * (config.commission + config.slippage)
    adjusted_profit = net_profit - total_cost
    
    if adjusted_profit <= 0:
        return adjusted_profit - 1000 # Zarar edenler dipte kalsın
    
    fitness = adjusted_profit
    
    # 1. Profit Factor "Sweet Spot" (Kullanıcı Talebi)
    # 1.5 altı -> Ciddi ceza
    # 1.5 - 2.5 -> Bonus
    # 2.5 üzeri -> Azalan getiri (Overfit şüphesi)
    if pf < 1.25:
        fitness *= 0.1 # Çöp
    elif pf < 1.5:
        fitness *= 0.5 # Kabul edilebilir sınırın altı
    elif 1.5 <= pf <= 2.5:
        fitness *= (1 + (pf - 1.5)) # Bonus: Örn PF 2.0 -> %50 bonus
    elif pf > 3.0:
        fitness *= 0.8 # Overfit cezası (Şüphe)
        
    # 2. Max DD Limit
    dd_ratio = max_dd / config.initial_capital
    if dd_ratio > config.max_dd_ratio:
        fitness *= (0.2 / dd_ratio) # Limit aşıldıkça puan erir
        
    # 3. İşlem Sayısı (İstatistiksel Güvenilirlik)
    if trades < 50:
        fitness *= (trades / 50) # Yetersiz veri cezası
    elif trades > 1000:
        fitness *= (1000 / trades) ** 2 # Overtrading - Sert Ceza
        
    # 4. İşlem Başı Ortalama Kar
    avg_profit = adjusted_profit / trades if trades > 0 else 0
    if avg_profit < config.min_avg_profit:
        fitness *= (avg_profit / config.min_avg_profit) ** 2

    # 5. Sharpe Ratio Bonusu (Yeni)
    sharpe = metrics.get('sharpe', 0)
    if sharpe > 2.0:
        fitness *= 1.1 # %10 Bonus
    elif sharpe < 0.5:
        fitness *= 0.8 # Riskli

    return fitness


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

def calculate_sharpe(returns: np.array, risk_free=0.0) -> float:
    """Sharpe Oranı hesapla (Yıllıklandırılmış)"""
    if len(returns) < 2: return 0.0
    
    excess_returns = returns - risk_free/252
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0: return 0.0
    
    # Günlük veriden yıllık Sharpe (karekök 252)
    # Eğer trade bazlı ise trade sayısı üzerinden normalize edilmeli
    # Biz burada basitçe mean/std * sqrt(N) kullanıyoruz
    return (mean_excess / std_excess) * np.sqrt(len(returns))


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
