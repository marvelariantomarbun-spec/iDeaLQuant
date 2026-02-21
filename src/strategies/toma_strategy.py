"""
Strategy 4: TOMA + Momentum
---------------------------
Karma Strateji: TOMA trend filtresi ve Uzun Vadeli Momentum/TRIX teyidi.

Kurallar:
1. MOM1(1900) > 101.5 ve TRIX(120) Uyuşmazlık -> A/S (Öncelik: Düşük)
2. MOM1(1900) < 98 ve TRIX(120) Uyuşmazlık -> A/S (Öncelik: Düşük)
3. HHV/LLV Breakout + TOMA Filtresi -> A/S (Öncelik: Yüksek, Sinyal 1 ve 2'yi ezer)

Parametreler (Default):
    mom_period: 1900
    mom_upper: 101.5
    mom_lower: 98.0
    trix_period: 120
    hh_ll_period: 20
    toma_period: 2
    toma_opt: 2.1
"""

from typing import Dict, Any, List
from ..indicators.trend import HHV, LLV, TOMA
from ..indicators.oscillators import TRIX
from ..indicators.core import Momentum
from .common import Signal


class TomaStrategy:
    def __init__(self, params: Dict[str, Any]):
        self.mom_period = int(params.get('mom_period', 1900))
        self.mom_upper = params.get('mom_upper', 101.5)
        self.mom_lower = params.get('mom_lower', 98.0)
        self.trix_period = int(params.get('trix_period', 120))
        
        # HHV/LLV Period (Kısa Vade)
        self.hh_ll_period = int(params.get('hh_ll_period', 20))
        
        # Uzun Vade HHV/LLV (Momentum Sinyali için)
        self.hh_ll_long_period1 = int(params.get('hh_ll_long_period1', 150))
        self.hh_ll_long_period2 = int(params.get('hh_ll_long_period2', 190))
        
        self.toma_period = int(params.get('toma_period', 2))
        self.toma_opt = params.get('toma_opt', 2.1)
        
        # Minimum veri gereksinimi (Momentum 1900 en büyüğü)
        self.min_bars = self.mom_period + 10
        self.cache = None

    @classmethod
    def from_config_dict(cls, data: Dict[str, List[float]], config: Dict[str, Any], dates: List[Any] = None) -> 'TomaStrategy':
        # UI parametrelerini strateji parametrelerine cevir
        params = {
            'mom_period': config.get('mom_period', 1900),
            'mom_upper': config.get('mom_limit_high', 101.5),
            'mom_lower': config.get('mom_limit_low', 98.0),
            'trix_period': config.get('trix_period', 120),
            'toma_period': config.get('toma_period', 97),
            'toma_opt': config.get('toma_opt', 1.5),
            
            # HHV/LLV Periods
            'hh_ll_period': config.get('hhv1_period', 20), # TOMA Filtre
            'hh_ll_long_period1': config.get('hhv2_period', 150), # L1
            'hh_ll_long_period2': config.get('llv2_period', 190), # L1
            
            # Layer 2 Params (mapped to class fields if needed, or handled in calc)
            'hhv3_period': config.get('hhv3_period', 150),
            'llv3_period': config.get('llv3_period', 190),
            
            'trix_lb1': config.get('trix_lb1', 145),
            'trix_lb2': config.get('trix_lb2', 160)
        }
        
        instance = cls(params)
        
        if hasattr(data, 'closes'): # IndicatorCache detected
            instance.cache = data
            instance.closes = data.closes
            instance.highs = data.highs
            instance.lows = data.lows
        else:
            instance.closes = data.get('closes', [])
            instance.highs = data.get('highs', [])
            instance.lows = data.get('lows', [])
        # Assign extra params that might not be in __init__
        instance.hhv3_period = params['hhv3_period']
        instance.llv3_period = params['llv3_period']
        instance.trix_lb1 = params['trix_lb1']
        instance.trix_lb2 = params['trix_lb2']
        
        return instance

    def generate_all_signals(self):
        from .common import Signal
        signals = self.calculate_signals(self.closes, self.highs, self.lows)
        n = len(signals)
        exits_long = [False] * n
        exits_short = [False] * n
        
        # Always-in system: opposite signal = exit
        pos = 0  # 0=flat, 1=long, -1=short
        for i in range(n):
            if signals[i] == Signal.LONG:
                if pos == -1:
                    exits_short[i] = True  # Close short
                pos = 1
            elif signals[i] == Signal.SHORT:
                if pos == 1:
                    exits_long[i] = True  # Close long
                pos = -1
        
        # Convert Signal enum to int for backtest compatibility
        int_signals = [0] * n
        for i in range(n):
            if signals[i] == Signal.LONG:
                int_signals[i] = 1
            elif signals[i] == Signal.SHORT:
                int_signals[i] = -1
        
        return int_signals, exits_long, exits_short

    def calculate_signals(self, closes: List[float], highs: List[float], lows: List[float]) -> List[Signal]:
        n = len(closes)
        signals = [Signal.NONE] * n
        
        if n < self.min_bars:
            return signals

        # --- INDIKATORLER ---
        if self.cache and hasattr(self.cache, 'get_toma'):
            # Use Cached Indicators
            toma_line, _ = self.cache.get_toma(self.toma_period, self.toma_opt)
            
            hh1 = self.cache.get_hhv(self.hh_ll_period)
            ll1 = self.cache.get_llv(self.hh_ll_period)
            
            hh2 = self.cache.get_hhv(self.hh_ll_long_period1)
            ll2 = self.cache.get_llv(self.hh_ll_long_period2)
            
            hh3_p = getattr(self, 'hhv3_period', self.hh_ll_long_period1)
            ll3_p = getattr(self, 'llv3_period', self.hh_ll_long_period2)
            hh3 = self.cache.get_hhv(hh3_p)
            ll3 = self.cache.get_llv(ll3_p)
            
            trix_lb1 = getattr(self, 'trix_lb1', 110)
            trix_lb2 = getattr(self, 'trix_lb2', 140)
            
            mom1 = self.cache.get_momentum(self.mom_period)
            trix1 = self.cache.get_trix(self.trix_period)
            
        else:
            # Calculate Fresh
            # 1. TOMA
            toma_line, _ = TOMA(closes, self.toma_period, self.toma_opt)
            
            # 2. HHV/LLV (Kısa Vade)
            hh1 = HHV(highs, self.hh_ll_period)
            ll1 = LLV(lows, self.hh_ll_period)
            
            # 3. HHV/LLV (Uzun Vade - Momentum Kuralı için)
            hh2 = HHV(highs, self.hh_ll_long_period1)
            ll2 = LLV(lows, self.hh_ll_long_period2)
            
            # Layer 2 Params
            hh3_p = getattr(self, 'hhv3_period', self.hh_ll_long_period1)
            ll3_p = getattr(self, 'llv3_period', self.hh_ll_long_period2)
            hh3 = HHV(highs, hh3_p)
            ll3 = LLV(lows, ll3_p)
            
            # TRIX Lookbacks
            trix_lb1 = getattr(self, 'trix_lb1', 110)
            trix_lb2 = getattr(self, 'trix_lb2', 140)

            # 4. Momentum (1900)
            mom1 = Momentum(closes, self.mom_period)
            
            # 5. TRIX (120)
            trix1 = TRIX(closes, self.trix_period)
        # trix2 = trix1 

        son_yon = Signal.NONE
        
        # Veri yeterliliği için döngü başlangıcı
        start_idx = max(self.mom_period, self.trix_period + max(trix_lb1, trix_lb2)) 
        
        for i in range(start_idx, n):
            sinyal = Signal.NONE
            
            # --- KURAL 1: MOM > 101.5 (Aşırı Alım Bölgesinde Uyuşmazlık/Teyit) ---
            if mom1[i] > self.mom_upper:
                # LONG: Yeni Zirve (HH2) VE TRIX Dip Dönüşü (110 bar öncesine göre düşük ama artıyor)
                if (hh2[i] > hh2[i-1]) and (trix1[i] < trix1[i-trix_lb1]) and (trix1[i] > trix1[i-1]):
                    sinyal = Signal.LONG
                
                # SHORT: Yeni Dip (LL2) VE TRIX Tepe Dönüşü (110 bar öncesine göre yüksek ama düşüyor)
                if (ll2[i] < ll2[i-1]) and (trix1[i] > trix1[i-trix_lb1]) and (trix1[i] < trix1[i-1]):
                    sinyal = Signal.SHORT
            
            # --- KURAL 2: MOM < 98 (Aşırı Satım Bölgesinde Uyuşmazlık/Teyit) ---
            if mom1[i] < self.mom_lower:
                # LONG
                if (hh3[i] > hh3[i-1]) and (trix1[i] < trix1[i-trix_lb2]) and (trix1[i] > trix1[i-1]):
                    sinyal = Signal.LONG
                
                # SHORT
                if (ll3[i] < ll3[i-1]) and (trix1[i] > trix1[i-trix_lb2]) and (trix1[i] < trix1[i-1]):
                    sinyal = Signal.SHORT
            
            # --- KURAL 3: HHV/LLV + TOMA (Ana Trend) ---
            # Bu kural, Kural 1 ve 2'den gelen sinyali EZER (Overwrite).
            if (hh1[i] > hh1[i-1]) and (closes[i] > toma_line[i]):
                sinyal = Signal.LONG
            
            if (ll1[i] < ll1[i-1]) and (closes[i] < toma_line[i]):
                sinyal = Signal.SHORT
            
            # --- POZİSYON GÜNCELLEME ---
            if sinyal != Signal.NONE and sinyal != son_yon:
                signals[i] = sinyal
                son_yon = sinyal
            else:
                signals[i] = Signal.NONE # Veya son_yon'u tekrar atayabiliriz (Backtest yapısına göre)
                # IdealData mantığı: Sinyal[i] sadece değişim anında doludur veya hep doludur.
                # Bizim backtest engine genelde 'Signal' arrayinde sadece değişimleri bekler.
                # Ancak burada 'son_yon' takibi yapıldığı için değişim anını kaydediyoruz.

        return signals
