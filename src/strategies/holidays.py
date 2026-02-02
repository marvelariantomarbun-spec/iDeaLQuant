# -*- coding: utf-8 -*-
"""
IdealQuant - Tatil ve Vade Yönetimi Modülü
Bayram tarihleri, resmi tatiller ve vade sonu iş günü hesaplama.
"""

from datetime import date, timedelta, time
from typing import Dict

# ===============================================================================================
# DİNAMİK BAYRAM TARİHLERİ (2024-2030) - IdealData ile birebir aynı
# ===============================================================================================
BAYRAM_TARIHLERI: Dict[int, Dict[str, date]] = {
    2024: {'ramazan': date(2024, 4, 10), 'kurban': date(2024, 6, 16)},
    2025: {'ramazan': date(2025, 3, 30), 'kurban': date(2025, 6, 6)},
    2026: {'ramazan': date(2026, 3, 20), 'kurban': date(2026, 5, 27)},
    2027: {'ramazan': date(2027, 3, 9), 'kurban': date(2027, 5, 16)},
    2028: {'ramazan': date(2028, 2, 26), 'kurban': date(2028, 5, 5)},
    2029: {'ramazan': date(2029, 2, 14), 'kurban': date(2029, 4, 24)},
    2030: {'ramazan': date(2030, 2, 3), 'kurban': date(2030, 4, 13)},
}

# Resmi Tatiller (MM-DD formatında tuple)
RESMI_TATILLER = [
    (1, 1),   # Yılbaşı
    (4, 23),  # 23 Nisan
    (5, 1),   # 1 Mayıs
    (5, 19),  # 19 Mayıs
    (7, 15),  # 15 Temmuz
    (8, 30),  # 30 Ağustos
    (10, 29), # 29 Ekim
]


def is_bayram_tatili(d: date) -> bool:
    """Bayram tatili mi kontrol et (Ramazan 3 gün, Kurban 4 gün)"""
    yil = d.year
    if yil not in BAYRAM_TARIHLERI:
        return False
    
    bayramlar = BAYRAM_TARIHLERI[yil]
    ramazan = bayramlar['ramazan']
    kurban = bayramlar['kurban']
    
    # Ramazan Bayramı (3 gün)
    if ramazan <= d <= ramazan + timedelta(days=3):
        return True
    
    # Kurban Bayramı (4 gün)
    if kurban <= d <= kurban + timedelta(days=4):
        return True
    
    return False


def is_arefe(d: date) -> bool:
    """Arefe günü mü kontrol et"""
    yil = d.year
    if yil not in BAYRAM_TARIHLERI:
        return False
    
    bayramlar = BAYRAM_TARIHLERI[yil]
    ramazan_arefe = bayramlar['ramazan'] - timedelta(days=1)
    kurban_arefe = bayramlar['kurban'] - timedelta(days=1)
    
    return d == ramazan_arefe or d == kurban_arefe


def is_holiday_eve(d: date) -> bool:
    """Arefe günü için alternatif isim (score_based.py uyumu)"""
    return is_arefe(d)


def is_resmi_tatil(d: date) -> bool:
    """Resmi tatil mi kontrol et"""
    return (d.month, d.day) in RESMI_TATILLER


def is_tatil_gunu(d: date) -> bool:
    """Herhangi bir tatil günü mü (hafta sonu dahil)"""
    # Hafta sonu
    if d.weekday() >= 5:
        return True
    # Resmi tatil
    if is_resmi_tatil(d):
        return True
    # Bayram tatili
    if is_bayram_tatili(d):
        return True
    return False


def vade_sonu_is_gunu(dt, vade_tipi: str = "ENDEKS") -> date:
    """
    Vade sonu iş gününü hesapla - IdealData ile birebir aynı mantık
    
    Args:
        dt: Tarih (date veya datetime)
        vade_tipi: "ENDEKS" (çift ay) veya "SPOT" (her ay)
    
    Returns:
        Vade sonu iş günü (date)
    """
    import calendar
    
    # datetime ise date'e çevir
    if hasattr(dt, 'date'):
        dt = dt.date()
    
    # Ayın son günü
    ay_sonu = date(dt.year, dt.month, calendar.monthrange(dt.year, dt.month)[1])
    d = ay_sonu
    
    # Max 15 gün geri git
    for _ in range(15):
        # Hafta sonu
        if d.weekday() >= 5:
            d -= timedelta(days=1)
            continue
        
        # Resmi tatil
        if is_resmi_tatil(d):
            d -= timedelta(days=1)
            continue
        
        # Bayram tatili
        if is_bayram_tatili(d):
            d -= timedelta(days=1)
            continue
        
        break
    
    return d


def is_seans_icinde(t: time) -> bool:
    """Seans saati içinde mi kontrol et (09:30-18:15, 19:00-23:00)"""
    gun_seansi = time(9, 30) <= t < time(18, 15)
    aksam_seansi = time(19, 0) <= t < time(23, 0)
    return gun_seansi or aksam_seansi
