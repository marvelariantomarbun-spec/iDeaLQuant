"""
IdealQuant Indicators Package
Complete indicator library with IdealData compatibility
"""

# =============================================================================
# CORE INDICATORS
# =============================================================================
from .core import (
    # Moving Averages
    MA, SMA, EMA, WMA, HullMA, RMA,
    
    # Oscillators
    RSI, Momentum, Stochastic, QQEF,
    
    # Volatility
    ATR, BollingerBands,
    
    # Trend
    ADX,
    
    # High/Low
    HHV, LLV,
    
    # Custom
    ARS, ARS_Dynamic, Qstick, RVI, NetLot,
    ChaikinMoneyFlow, MoneyFlowIndex, MFI, MACDV
)

# =============================================================================
# EXTENDED MOVING AVERAGES
# =============================================================================
from .moving_avg import (
    DEMA, TEMA, KAMA, FRAMA, WWMA, SMMA, ZLEMA, T3
)

# =============================================================================
# OSCILLATORS
# =============================================================================
from .oscillators import (
    CCI, MACD, StochRSI, WilliamsR, ROC,
    UltimateOscillator, TRIX, DPO, ChandeMomentum,
    RMI, AwesomeOscillator, ElliotWaveOscillator,
    StochasticFast, StochasticSlow
)

# =============================================================================
# TREND INDICATORS
# =============================================================================
from .trend import (
    DirectionalIndicatorPlus, DirectionalIndicatorMinus,
    DI_Plus, DI_Minus,
    AroonUp, AroonDown, AroonOsc, Aroon,
    ParabolicSAR, Ichimoku, IchimokuResult,
    PriceChannelUp, PriceChannelDown, PriceChannel,
    VHF, LinearReg, LinearRegSlope
)

# =============================================================================
# VOLUME INDICATORS
# =============================================================================
from .volume import (
    OBV, PVT, ADL, ChaikinOsc, NVI, PVI,
    KlingerOsc, MassIndex, EaseOfMovement, ForceIndex
)

# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================
from .volatility import (
    BollingerUp, BollingerDown, BollingerMid, BollingerWidth, BollingerPercentB,
    KeltnerUp, KeltnerDown, KeltnerChannel,
    EnvelopeUp, EnvelopeDown, EnvelopeMid,
    StandardDeviation, ChaikinVolatility, TrueRange, NATR
)


__all__ = [
    # Core - MA
    'MA', 'SMA', 'EMA', 'WMA', 'HullMA', 'RMA',
    # Extended MA
    'DEMA', 'TEMA', 'KAMA', 'FRAMA', 'WWMA', 'SMMA', 'ZLEMA', 'T3',
    
    # Core - Oscillators
    'RSI', 'Momentum', 'Stochastic', 'QQEF',
    # Extended Oscillators
    'CCI', 'MACD', 'StochRSI', 'WilliamsR', 'ROC',
    'UltimateOscillator', 'TRIX', 'DPO', 'ChandeMomentum',
    'RMI', 'AwesomeOscillator', 'ElliotWaveOscillator',
    'StochasticFast', 'StochasticSlow',
    
    # Volatility
    'ATR', 'BollingerBands',
    'BollingerUp', 'BollingerDown', 'BollingerMid', 'BollingerWidth', 'BollingerPercentB',
    'KeltnerUp', 'KeltnerDown', 'KeltnerChannel',
    'EnvelopeUp', 'EnvelopeDown', 'EnvelopeMid',
    'StandardDeviation', 'ChaikinVolatility', 'TrueRange', 'NATR',
    
    # Trend
    'ADX', 'DirectionalIndicatorPlus', 'DirectionalIndicatorMinus', 'DI_Plus', 'DI_Minus',
    'AroonUp', 'AroonDown', 'AroonOsc', 'Aroon',
    'ParabolicSAR', 'Ichimoku', 'IchimokuResult',
    'PriceChannelUp', 'PriceChannelDown', 'PriceChannel',
    'VHF', 'LinearReg', 'LinearRegSlope',
    
    # High/Low
    'HHV', 'LLV',
    
    # Volume
    'OBV', 'PVT', 'ADL', 'ChaikinOsc', 'NVI', 'PVI',
    'KlingerOsc', 'MassIndex', 'EaseOfMovement', 'ForceIndex',
    'ChaikinMoneyFlow', 'MoneyFlowIndex', 'MFI',
    
    # Custom
    'ARS', 'ARS_Dynamic', 'Qstick', 'RVI', 'NetLot', 'MACDV'
]
