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
    """
    Dengeli çok faktörlü fitness hesaplama.
    
    Eski formül: score = net_profit × çarpanlar → net kâr dominant
    Yeni formül: score = profit_score × quality_score × risk_score × trade_score
    Her faktör 0-100 arası normalize edilir, net kâr artık dominant değil.
    """
    import math
    
    # Maliyet düşülmüş kâr
    cost = trades * (commission + slippage)
    adj_profit = net_profit - cost
    
    # Zarar varsa erken çık
    if adj_profit <= 0:
        return -99999 + adj_profit
    if trades == 0:
        return -99999
    
    # =========================================================================
    # 1. PROFIT SCORE (0-100) — Logaritmik ölçek, büyük farkları daraltır
    # =========================================================================
    # log10(1000) = 3, log10(10000) = 4, log10(100000) = 5
    # Aradaki fark 100x ama skor farkı sadece 2x (3 vs 5)
    profit_score = min(100, math.log10(max(adj_profit, 1)) * 20)
    
    # =========================================================================
    # 2. QUALITY SCORE (0-100) — PF ve Sharpe birleşik
    # =========================================================================
    # PF Skoru (0-50)
    if pf < 1.0:
        pf_score = 0
    elif pf < 1.3:
        pf_score = 5                    # Ölü bölge
    elif pf < 1.5:
        pf_score = 15                   # Zayıf
    elif pf <= 2.5:
        pf_score = 30 + (pf - 1.5) * 20  # Sweet spot: 30-50
    elif pf <= 3.0:
        pf_score = 45                   # İyi ama dikkat
    else:
        pf_score = max(10, 45 - (pf - 3.0) * 15)  # Overfit cezası
    
    # Sharpe Skoru (0-50)
    if sharpe <= 0:
        sharpe_score = 0
    elif sharpe < 0.5:
        sharpe_score = sharpe * 20       # 0-10
    elif sharpe < 1.5:
        sharpe_score = 10 + (sharpe - 0.5) * 30  # 10-40
    elif sharpe <= 3.0:
        sharpe_score = 40 + (sharpe - 1.5) * 6.67  # 40-50
    else:
        sharpe_score = 50
    
    quality_score = pf_score + sharpe_score  # 0-100
    
    # =========================================================================
    # 3. RISK SCORE (0-100) — DD ve avg trade
    # =========================================================================
    # Drawdown oranı (düşük = iyi)
    dd_ratio = max_dd / initial_capital if initial_capital > 0 else 1.0
    if dd_ratio <= 0.10:
        dd_score = 50                    # Mükemmel: DD < %10
    elif dd_ratio <= 0.20:
        dd_score = 40                    # İyi: DD < %20
    elif dd_ratio <= 0.30:
        dd_score = 25                    # Riskli: DD < %30
    else:
        dd_score = max(0, 15 - (dd_ratio - 0.30) * 50)  # Tehlikeli
    
    # İşlem başı kâr
    avg_pnl = adj_profit / trades
    if avg_pnl >= 30:
        avg_score = 50
    elif avg_pnl >= 15:
        avg_score = 25 + (avg_pnl - 15) / 15 * 25  # 25-50 arası
    elif avg_pnl >= 5:
        avg_score = (avg_pnl - 5) / 10 * 25         # 0-25 arası
    else:
        avg_score = 0
    
    risk_score = dd_score + avg_score  # 0-100
    
    # =========================================================================
    # 4. TRADE SCORE (0-100) — İstatistiksel güvenilirlik
    # =========================================================================
    if trades < 20:
        trade_score = 10                 # Çok az → güvenilmez
    elif trades < 50:
        trade_score = 30                 # Minimum seviye
    elif trades <= 500:
        trade_score = 50 + (trades - 50) / 450 * 50  # 50-100 arası, ideal bölge
    elif trades <= 1500:
        trade_score = 90                 # Yeterli
    else:
        trade_score = max(30, 90 - (trades - 1500) / 500 * 30)  # Overtrading cezası
    
    # =========================================================================
    # BİRLEŞİK SKOR (4 faktör eşit ağırlıklı)
    # =========================================================================
    # Her biri 0-100 arası → toplam 0-400 → normalize 0-10000
    fitness = (profit_score * quality_score * risk_score * trade_score) / 10000
    
    # Minör ayar: çok küçük sonuçları filtrele
    if fitness < 0.01:
        return 0
    
    return fitness


def calculate_robust_fitness(results: list, param_keys: list = None) -> list:
    """
    Sonuç listesindeki her sonuç için komşu yoğunluğuna dayalı robust_fitness hesapla.
    
    Mantık:
    1. Her sonuç çifti arasında parametrik mesafe hesapla (normalize)
    2. Her sonuç için yakın komşu sayısını ve ortalama fitness'ını bul
    3. robust_fitness = raw_fitness × (0.5 + 0.5 × density_score)
    
    İzole uç nokta → density ≈ 0 → %50 ceza
    Kalabalık bölge merkezi → density ≈ 1 → tam puan
    
    Args:
        results: [{param1: val, param2: val, ..., fitness: score}, ...]
        param_keys: Hangi parametrelerin mesafe hesabına gireceği (None=otomatik)
    
    Returns:
        Aynı liste, her dict'e 'robust_fitness' ve 'density_score' eklenmiş
    """
    if not results or len(results) < 3:
        for r in results:
            r['robust_fitness'] = r.get('fitness', 0)
            r['density_score'] = 1.0
        return results
    
    # Metrik olmayan anahtarları listele
    _metric_keys = {
        'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness',
        'robust_fitness', 'density_score', 'stability', 'win_rate',
        'test_net', 'test_pf', 'test_sharpe', 'win_count'
    }
    
    # Parametre anahtarlarını otomatik tespit
    if param_keys is None:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        param_keys = [k for k in all_keys if k not in _metric_keys 
                      and isinstance(results[0].get(k), (int, float))]
    
    if not param_keys:
        for r in results:
            r['robust_fitness'] = r.get('fitness', 0)
            r['density_score'] = 1.0
        return results
    
    # 1. Parametreleri normalize et (min-max scaling)
    param_ranges = {}
    for key in param_keys:
        vals = [r.get(key, 0) for r in results if key in r]
        if vals:
            vmin, vmax = min(vals), max(vals)
            param_ranges[key] = (vmin, vmax - vmin) if vmax > vmin else (vmin, 1.0)
    
    # 2. Her sonuç çifti arasında mesafe hesapla
    n = len(results)
    
    def normalized_distance(r1, r2):
        """İki sonuç arasındaki normalize parametrik mesafe (0-1)"""
        dist_sq = 0
        count = 0
        for key in param_keys:
            if key in r1 and key in r2 and key in param_ranges:
                vmin, vrange = param_ranges[key]
                d1 = (r1[key] - vmin) / vrange
                d2 = (r2[key] - vmin) / vrange
                dist_sq += (d1 - d2) ** 2
                count += 1
        return (dist_sq / max(count, 1)) ** 0.5
    
    # 3. Her sonuç için density score hesapla
    # Mesafe eşiği: parametre uzayının %15'i içindeki komşular
    distance_threshold = 0.15
    max_fitness = max(r.get('fitness', 0) for r in results)
    if max_fitness <= 0:
        max_fitness = 1.0
    
    for i, result in enumerate(results):
        raw_fitness = result.get('fitness', 0)
        
        # Yakın komşuları bul
        neighbor_count = 0
        neighbor_fitness_sum = 0
        
        for j, other in enumerate(results):
            if i == j:
                continue
            dist = normalized_distance(result, other)
            if dist <= distance_threshold:
                neighbor_count += 1
                neighbor_fitness_sum += other.get('fitness', 0)
        
        # Density score hesapla
        if neighbor_count > 0:
            neighbor_avg = neighbor_fitness_sum / neighbor_count
            # Komşu yoğunluğu (kaç komşu var) × komşu kalitesi (ortalamaları ne kadar iyi)
            count_ratio = min(1.0, neighbor_count / max(5, n * 0.1))  # En az %10 komşu
            quality_ratio = neighbor_avg / max_fitness
            density_score = (count_ratio * 0.5 + quality_ratio * 0.5)
        else:
            density_score = 0.0
        
        # Robust fitness: raw × (0.5 + 0.5 × density)
        result['density_score'] = round(density_score, 3)
        result['robust_fitness'] = raw_fitness * (0.5 + 0.5 * density_score)
    
    return results

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
