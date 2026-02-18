# -*- coding: utf-8 -*-
"""
IdealQuant - Optimizer Panel
Parametre aralıkları ve grup bazlı optimizasyon
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QSpinBox,
    QDoubleSpinBox, QFormLayout, QMessageBox, QSplitter,
    QScrollArea, QCheckBox, QFrame, QGridLayout
)
from PySide6.QtCore import Signal, Qt, QThread, QTimer
from typing import Dict, List, Any
import pandas as pd
import time
import os

from src.core.database import db


# Strateji 1 Parametre Grupları (20 parametre)
STRATEGY1_PARAM_GROUPS = {
    'ARS': {
        'label': 'ARS Parametreleri',
        'params': {
            'ars_period': {'label': 'ARS Periyot', 'type': 'int', 'default': 3, 'min': 1, 'max': 15, 'step': 2},
            'ars_k': {'label': 'ARS K', 'type': 'float', 'default': 0.01, 'min': 0.005, 'max': 0.100, 'step': 0.010},
        }
    },
    'ADX': {
        'label': 'ADX Parametreleri',
        'params': {
            'adx_period': {'label': 'ADX Periyot', 'type': 'int', 'default': 17, 'min': 10, 'max': 30, 'step': 5},
            'adx_threshold': {'label': 'ADX Esik', 'type': 'float', 'default': 25.0, 'min': 15.0, 'max': 50.0, 'step': 5.0},
        }
    },
    'MACDV': {
        'label': 'MACD-V Parametreleri',
        'params': {
            'macdv_short': {'label': 'Kisa Periyot', 'type': 'int', 'default': 13, 'min': 8, 'max': 20, 'step': 2},
            'macdv_long': {'label': 'Uzun Periyot', 'type': 'int', 'default': 28, 'min': 20, 'max': 40, 'step': 5},
            'macdv_signal': {'label': 'Sinyal Periyot', 'type': 'int', 'default': 8, 'min': 5, 'max': 15, 'step': 2},
            'macdv_threshold': {'label': 'MACDV Esik', 'type': 'float', 'default': 0.0, 'min': 0.001, 'max': 50.0, 'step': 10.0},
        }
    },
    'NetLot': {
        'label': 'NetLot Parametreleri',
        'params': {
            'netlot_period': {'label': 'Periyot', 'type': 'int', 'default': 5, 'min': 3, 'max': 15, 'step': 2},
            'netlot_threshold': {'label': 'Esik', 'type': 'float', 'default': 20.0, 'min': 10.0, 'max': 50.0, 'step': 5.0},
        }
    },
    'Yatay_BB': {
        'label': 'Yatay Filtre - Bollinger',
        'params': {
            'bb_period': {'label': 'BB Periyot', 'type': 'int', 'default': 20, 'min': 10, 'max': 100, 'step': 5},
            'bb_std': {'label': 'BB StdDev', 'type': 'float', 'default': 2.0, 'min': 1.5, 'max': 3.0, 'step': 0.25},
            'bb_width_multiplier': {'label': 'BB Width Mult', 'type': 'float', 'default': 0.8, 'min': 0.5, 'max': 1.5, 'step': 0.1},
            'bb_avg_period': {'label': 'BB Avg Periyot', 'type': 'int', 'default': 50, 'min': 30, 'max': 100, 'step': 10},
        }
    },
    'Yatay_Onay': {
        'label': 'Yatay Filtre - Onay',
        'params': {
            'ars_mesafe_threshold': {'label': 'ARS Mesafe', 'type': 'float', 'default': 0.25, 'min': 0.10, 'max': 0.50, 'step': 0.05},
            'yatay_ars_bars': {'label': 'ARS Bar', 'type': 'int', 'default': 10, 'min': 5, 'max': 20, 'step': 5},
            'yatay_adx_threshold': {'label': 'Yatay ADX Esik', 'type': 'float', 'default': 20.0, 'min': 10.0, 'max': 40.0, 'step': 5.0},
            'filter_score_threshold': {'label': 'Filtre Skor', 'type': 'int', 'default': 2, 'min': 1, 'max': 4, 'step': 1},
        }
    },

    'Skor': {
        'label': 'Skor Ayarlari (Kademeli)',
        'params': {
            'min_score': {'label': 'Min Skor', 'type': 'int', 'default': 3, 'min': 2, 'max': 4, 'step': 1},
            'exit_score': {'label': 'Cikis Skor', 'type': 'int', 'default': 3, 'min': 2, 'max': 4, 'step': 1},
            'contrary_score_max': {'label': 'Karşıt Skor Max', 'type': 'int', 'default': 2, 'min': 1, 'max': 3, 'step': 1},
        },
        'is_cascaded': True
    }
}


# Strateji 2 Parametre Grupları (21 parametre)
STRATEGY2_PARAM_GROUPS = {
    'ARS': {
        'label': 'ARS Parametreleri',
        'params': {
            'ars_ema_period': {'label': 'EMA Periyot', 'type': 'int', 'default': 3, 'min': 2, 'max': 12, 'step': 1},
            'ars_atr_period': {'label': 'ATR Periyot', 'type': 'int', 'default': 10, 'min': 7, 'max': 20, 'step': 2},
            'ars_atr_mult': {'label': 'ATR Carpan', 'type': 'float', 'default': 0.5, 'min': 0.3, 'max': 1.5, 'step': 0.1},
            'ars_min_band': {'label': 'Min Band', 'type': 'float', 'default': 0.002, 'min': 0.001, 'max': 0.005, 'step': 0.001},
            'ars_max_band': {'label': 'Max Band', 'type': 'float', 'default': 0.015, 'min': 0.010, 'max': 0.025, 'step': 0.005},
        }
    },
    'Giris_Filtreleri': {
        'label': 'Giris Filtreleri',
        'params': {
            'momentum_period': {'label': 'Momentum Periyot', 'type': 'int', 'default': 5, 'min': 3, 'max': 10, 'step': 1},
            'momentum_threshold': {'label': 'Momentum Esik', 'type': 'float', 'default': 100.0, 'min': 50.0, 'max': 200.0, 'step': 25.0},
            # momentum_base sabit 200.0 - Momentum indikatoru 100 merkezli oldugu icin bu deger degismemeli
            'breakout_period': {'label': 'Breakout Periyot', 'type': 'int', 'default': 10, 'min': 5, 'max': 30, 'step': 5},
            'mfi_period': {'label': 'MFI Periyot', 'type': 'int', 'default': 14, 'min': 10, 'max': 21, 'step': 2},
            'mfi_hhv_period': {'label': 'MFI HHV', 'type': 'int', 'default': 14, 'min': 10, 'max': 21, 'step': 2},
            'mfi_llv_period': {'label': 'MFI LLV', 'type': 'int', 'default': 14, 'min': 10, 'max': 21, 'step': 2},
            'volume_hhv_period': {'label': 'Hacim HHV', 'type': 'int', 'default': 14, 'min': 10, 'max': 21, 'step': 2},
        }
    },
    'Cikis_Risk': {
        'label': 'Cikis / Risk Yonetimi',
        'params': {
            'atr_exit_period': {'label': 'ATR Exit Periyot', 'type': 'int', 'default': 14, 'min': 10, 'max': 21, 'step': 2},
            'atr_sl_mult': {'label': 'SL Carpan', 'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.5},
            'atr_tp_mult': {'label': 'TP Carpan', 'type': 'float', 'default': 5.0, 'min': 3.0, 'max': 8.0, 'step': 1.0},
            'atr_trail_mult': {'label': 'Trail Carpan', 'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.5},
            'exit_confirm_bars': {'label': 'Onay Bar', 'type': 'int', 'default': 2, 'min': 1, 'max': 5, 'step': 1},
            'exit_confirm_mult': {'label': 'Onay Carpan', 'type': 'float', 'default': 1.0, 'min': 0.5, 'max': 2.0, 'step': 0.25},
        },
        'is_cascaded': True
    },
    'Ince_Ayar': {
        'label': 'Ince Ayar (Kademeli)',
        'params': {
            'volume_mult': {'label': 'Hacim Carpan', 'type': 'float', 'default': 0.8, 'min': 0.5, 'max': 1.5, 'step': 0.1},
            'volume_llv_period': {'label': 'Hacim LLV', 'type': 'int', 'default': 14, 'min': 10, 'max': 21, 'step': 2},
        },
        'is_cascaded': True
    }
}


# Strateji 3 Parametre Gruplari (Paradise - 12 parametre)
STRATEGY3_PARAM_GROUPS = {
    'Trend': {
        'label': 'Trend Parametreleri',
        'params': {
            'ema_period': {'label': 'EMA Periyot', 'type': 'int', 'default': 21, 'min': 5, 'max': 80, 'step': 1},
            'dsma_period': {'label': 'DSMA Periyot', 'type': 'int', 'default': 50, 'min': 15, 'max': 150, 'step': 5},
            'ma_period': {'label': 'MA Periyot', 'type': 'int', 'default': 20, 'min': 5, 'max': 80, 'step': 1},
        }
    },
    'Breakout': {
        'label': 'Breakout Parametreleri',
        'params': {
            'hh_period': {'label': 'HH/LL Periyot', 'type': 'int', 'default': 25, 'min': 5, 'max': 80, 'step': 1},
            'vol_hhv_period': {'label': 'Hacim HHV Periyot', 'type': 'int', 'default': 14, 'min': 5, 'max': 50, 'step': 1},
        }
    },
    'Momentum': {
        'label': 'Momentum Parametreleri',
        'params': {
            'mom_period': {'label': 'Momentum Periyot', 'type': 'int', 'default': 60, 'min': 10, 'max': 150, 'step': 5},
            'mom_alt': {'label': 'MOM Alt Bant', 'type': 'float', 'default': 98.0, 'min': 90.0, 'max': 99.5, 'step': 0.5},
            'mom_ust': {'label': 'MOM Ust Bant', 'type': 'float', 'default': 102.0, 'min': 100.5, 'max': 110.0, 'step': 0.5},
        }
    },
    'Cikis_Risk': {
        'label': 'Cikis / Risk Yonetimi',
        'params': {
            'atr_period': {'label': 'ATR Periyot', 'type': 'int', 'default': 14, 'min': 5, 'max': 30, 'step': 1},
            'atr_sl': {'label': 'ATR Stop Loss', 'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 5.0, 'step': 0.25},
            'atr_tp': {'label': 'ATR Take Profit', 'type': 'float', 'default': 4.0, 'min': 1.0, 'max': 10.0, 'step': 0.5},
            'atr_trail': {'label': 'ATR Trailing', 'type': 'float', 'default': 2.5, 'min': 0.5, 'max': 6.0, 'step': 0.25},
        },
        'is_cascaded': True
    }
}
# Strateji 4 Parametre Grupları (TOMA ve Katmanlari)
# Strateji 4 Parametre Grupları (TOMA ve Katmanlari)
STRATEGY4_PARAM_GROUPS = {
    'Layer3_TOMA': {
        'label': 'Layer 3: TOMA (Trend)',
        'params': {
            'toma_period': {'label': 'TOMA Periyot', 'type': 'int', 'default': 97, 'min': 10, 'max': 200, 'step': 5},
            'toma_opt': {'label': 'TOMA Opt %', 'type': 'float', 'default': 1.5, 'min': 0.1, 'max': 5.0, 'step': 0.1},
            'hhv1_period': {'label': 'TOMA Filtre HHV', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'step': 5},
            'llv1_period': {'label': 'TOMA Filtre LLV', 'type': 'int', 'default': 20, 'min': 10, 'max': 50, 'step': 5},
        }
    },
    'Global_Settings': {
        'label': 'Global Indikator Ayarlari',
        'params': {
             'mom_period': {'label': 'Momentum Periyot', 'type': 'int', 'default': 1900, 'min': 100, 'max': 3000, 'step': 100},
             'trix_period': {'label': 'TRIX Periyot', 'type': 'int', 'default': 120, 'min': 10, 'max': 200, 'step': 10},
        },
        'is_cascaded': False 
    },
    'Layer1_MomHigh': {
        'label': 'Layer 1: Mom High (>101.5)',
        'params': {
            'mom_limit_high': {'label': 'Mom High >', 'type': 'float', 'default': 101.5, 'min': 95.0, 'max': 110.0, 'step': 0.5},
            'trix_lb1': {'label': 'TRIX LB (High)', 'type': 'int', 'default': 145, 'min': 50, 'max': 200, 'step': 5},
            'hhv2_period': {'label': 'L1 HHV Periyot', 'type': 'int', 'default': 150, 'min': 50, 'max': 300, 'step': 10},
            'llv2_period': {'label': 'L1 LLV Periyot', 'type': 'int', 'default': 190, 'min': 50, 'max': 300, 'step': 10},
        }
    },
    'Layer2_MomLow': {
        'label': 'Layer 2: Mom Low (<99.0)',
        'params': {
            'mom_limit_low': {'label': 'Mom Low <', 'type': 'float', 'default': 99.0, 'min': 90.0, 'max': 105.0, 'step': 0.5},
            'trix_lb2': {'label': 'TRIX LB (Low)', 'type': 'int', 'default': 160, 'min': 50, 'max': 200, 'step': 5},
            'hhv3_period': {'label': 'L2 HHV Periyot', 'type': 'int', 'default': 150, 'min': 50, 'max': 300, 'step': 10},
            'llv3_period': {'label': 'L2 LLV Periyot', 'type': 'int', 'default': 190, 'min': 50, 'max': 300, 'step': 10},
        }
    },
    'Risk': {
        'label': 'Risk Yonetimi',
        'params': {
            'kar_al': {'label': 'Kar Al %', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 10.0, 'step': 0.5},
            'iz_stop': {'label': 'Izleyen Stop %', 'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 5.0, 'step': 0.25},
        },
        'is_cascaded': True
    }
}

# ==============================================================================
# TIMEFRAME-ADAPTIVE PARAMETER SCALING
# ==============================================================================
# Periyot (lookback) parametreleri: timeframe'e gore olceklenmeli
# Esik/carpan/skor parametreleri: dimensionless, olceklenmez

SCALABLE_PARAMS = {
    # Strateji 1
    'ars_period', 'adx_period', 'macdv_short', 'macdv_long', 'macdv_signal',
    'netlot_period', 'bb_period', 'bb_avg_period', 'yatay_ars_bars',
    # Strateji 2
    'ars_ema_period', 'ars_atr_period', 'momentum_period', 'breakout_period',
    'mfi_period', 'mfi_hhv_period', 'mfi_llv_period', 'volume_hhv_period',
    'atr_exit_period', 'volume_llv_period',
    # Strateji 3 (Paradise)
    'ema_period', 'dsma_period', 'ma_period', 'hh_period', 'vol_hhv_period',
    'mom_period', 'atr_period',
    # Strateji 4 (TOMA)
    'toma_period', 'trix_lb1', 'trix_lb2' # Mom limits are dimensionless thresholds
}

REFERENCE_PERIOD = 5  # dk - Stratejilerin orijinal tasarim periyodu

def scale_param_groups(param_groups: dict, current_period_dk: int) -> dict:
    """Parametre gruplarini timeframe'e gore olcekle.
    
    Formul: scale_factor = REFERENCE_PERIOD / current_period_dk
    Sadece SCALABLE_PARAMS icindeki parametrelere uygulanir.
    """
    if current_period_dk == REFERENCE_PERIOD:
        return param_groups  # 5dk = referans, degisiklik yok
    
    import copy
    scaled = copy.deepcopy(param_groups)
    scale = REFERENCE_PERIOD / current_period_dk
    
    for group_name, group_config in scaled.items():
        for param_name, param_config in group_config['params'].items():
            if param_name not in SCALABLE_PARAMS:
                continue
            
            is_int = param_config['type'] == 'int'
            
            raw_min = param_config['min'] * scale
            raw_max = param_config['max'] * scale
            raw_default = param_config['default'] * scale
            raw_step = param_config['step'] * scale
            
            if is_int:
                param_config['min'] = max(2, round(raw_min))
                param_config['max'] = max(3, round(raw_max))
                param_config['default'] = max(param_config['min'], round(raw_default))
                param_config['step'] = max(1, round(raw_step))
            else:
                param_config['min'] = max(0.001, round(raw_min, 4))
                param_config['max'] = max(param_config['min'] + 0.001, round(raw_max, 4))
                param_config['default'] = max(param_config['min'], round(raw_default, 4))
                param_config['step'] = max(0.001, round(raw_step, 4))
            
            # default, min-max araliginda olmali
            param_config['default'] = min(param_config['default'], param_config['max'])
    
    return scaled


class ParameterGroupWidget(QGroupBox):
    """Tek bir parametre grubu widget'ı"""
    
    def __init__(self, group_name: str, group_config: dict):
        super().__init__(group_config['label'])
        self.group_name = group_name
        self.group_config = group_config
        self.param_widgets = {}
        # Boyut politikası - içeriğe göre genişle
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 15, 5, 5)  # Başlık için üstte boşluk
        
        # Kademeli grup uyarısı
        if self.group_config.get('is_cascaded', False):
            cascade_label = QLabel("(Bu grup, diger gruplar optimize edildikten sonra calisir)")
            cascade_label.setStyleSheet("color: #e94560; font-style: italic;")
            layout.addWidget(cascade_label)
        
        # Parametre tablosu
        params = self.group_config['params']
        table = QTableWidget(len(params), 5)
        table.setHorizontalHeaderLabels(['Parametre', 'Min', 'Max', 'Adim', 'Aktif'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        
        for row, (param_name, param_config) in enumerate(params.items()):
            # Parametre adı
            name_item = QTableWidgetItem(param_config['label'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)
            
            # Min
            if param_config['type'] == 'int':
                min_spin = QSpinBox()
                min_spin.setRange(1, 10000) # Increased Range
                min_spin.setValue(param_config['min'])
            else:
                min_spin = QDoubleSpinBox()
                min_spin.setRange(0.0, 10000.0)
                min_spin.setDecimals(3)
                min_spin.setValue(param_config['min'])
            table.setCellWidget(row, 1, min_spin)
            
            # Max
            if param_config['type'] == 'int':
                max_spin = QSpinBox()
                max_spin.setRange(1, 10000) # Increased Range
                max_spin.setValue(param_config['max'])
            else:
                max_spin = QDoubleSpinBox()
                max_spin.setRange(0.0, 10000.0)
                max_spin.setDecimals(3)
                max_spin.setValue(param_config['max'])
            table.setCellWidget(row, 2, max_spin)
            
            # Adım
            if param_config['type'] == 'int':
                step_spin = QSpinBox()
                step_spin.setRange(1, 100)
                step_spin.setValue(param_config['step'])
            else:
                step_spin = QDoubleSpinBox()
                step_spin.setRange(0.001, 100.0)
                step_spin.setDecimals(3)
                step_spin.setValue(param_config['step'])
            table.setCellWidget(row, 3, step_spin)
            
            # Aktif checkbox
            active_widget = QWidget()
            active_layout = QHBoxLayout(active_widget)
            active_layout.setContentsMargins(0, 0, 0, 0)
            active_layout.setAlignment(Qt.AlignCenter)
            active_check = QCheckBox()
            active_check.setChecked(True)
            active_layout.addWidget(active_check)
            table.setCellWidget(row, 4, active_widget)
            
            # Widget'ları sakla
            self.param_widgets[param_name] = {
                'min': min_spin,
                'max': max_spin,
                'step': step_spin,
                'active': active_check,
                'type': param_config['type']
            }
        
        # Satır yüksekliğini ayarla
        row_height = 36  # Spinbox okları için yeterli alan
        header_height = 22  # Header daha kompakt
        
        # Header yüksekliğini ayarla
        table.horizontalHeader().setFixedHeight(header_height)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        
        # Satır yüksekliklerini ayarla
        for row in range(len(params)):
            table.setRowHeight(row, row_height)
        
        # Tablo yüksekliğini tüm satırları gösterecek şekilde ayarla
        total_height = header_height + (len(params) * row_height) + 2
        table.setFixedHeight(total_height)
        
        # Spinbox styling - varsayılan okları kullan
        table.setStyleSheet("""
            QHeaderView::section {
                font-size: 9pt;
                padding: 2px;
            }
            QSpinBox, QDoubleSpinBox {
                padding-right: 18px;
                min-height: 28px;
            }
        """)
        layout.addWidget(table)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(2)
    
    def get_ranges(self) -> Dict[str, List[Any]]:
        """Parametre aralıklarını döndür"""
        ranges = {}
        for param_name, widgets in self.param_widgets.items():
            if not widgets['active'].isChecked():
                continue
            
            min_val = widgets['min'].value()
            max_val = widgets['max'].value()
            step_val = widgets['step'].value()
            
            if widgets['type'] == 'int':
                values = list(range(int(min_val), int(max_val) + 1, int(step_val)))
            else:
                values = []
                v = min_val
                while v <= max_val + 0.0001:
                    values.append(round(v, 4))
                    v += step_val
            
            ranges[param_name] = values
        
        return ranges


class OptimizationWorker(QThread):
    """Arka plan optimizasyon thread'i"""
    
    # Progress: percent, message, elapsed_str, eta_str
    progress = Signal(int, str, str, str)
    result_ready = Signal(list)
    partial_results = Signal(list)   # Canlı streaming icin
    error = Signal(str)
    
    def __init__(self, config: dict, data, param_ranges: dict, 
                 method: str = "Hibrit Grup", strategy_index: int = 0,
                 process_id: str = None, n_parallel: int = 4,
                 narrowed_ranges: dict = None, do_oos: bool = True, train_pct: int = 70):
        super().__init__()
        self.config = config
        self.data = data
        self.param_ranges = param_ranges
        self.method = method
        self.strategy_index = strategy_index
        self.process_id = process_id
        self.n_parallel = n_parallel
        self.narrowed_ranges = narrowed_ranges  # Cascade icin dar araliklar
        self.do_oos = do_oos
        self.train_pct = train_pct
        self.commission = 0.0
        self.train_pct = train_pct
        self.commission = 0.0
        self.slippage = 0.0
        self.slippage = 0.0
        self.test_data = None  # Init
        self._is_running = True
        self.start_time = None
        self.pool = None  # For multiprocessing pool (S4)
        self.current_optimizer = None  # For Object-based optimizers
    
    def _emit_progress(self, percent: int, message: str):
        """İlerleme ve zaman bilgisini gönder"""
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        eta_str = "--:--"
        if percent > 0:
            total_time = (elapsed / percent) * 100
            remaining = total_time - elapsed
            eta_str = self._format_time(remaining)
            
        self.progress.emit(percent, message, elapsed_str, eta_str)

    def _format_time(self, seconds: float) -> str:
        """Saniyeyi MM:SS formatına çevir"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def run(self):
        self.start_time = time.time()
        
        # Auto-Split (Train/Test)
        original_data = self.data
        test_data = None
        
        if self.do_oos and self.data is not None:
            split_idx = int(len(self.data) * (self.train_pct / 100.0))
            train_data = self.data.iloc[:split_idx].copy()
            test_data = self.data.iloc[split_idx:].copy()
            
            # Optimizer sadece train verisini görsün
            self.data = train_data
            self.progress.emit(0, f"Veri bölündü: %{self.train_pct} Egitim ({len(train_data)} bar), %{100-self.train_pct} Test ({len(test_data)} bar)", "", "--:--")
        
        self.test_data = test_data
        
        try:
            if self.strategy_index == 3 and self.method == "Hibrit Grup": # Strategy 4 (TOMA) - Hybrid Mode
                self._run_sequential_layer()
            elif self.method == "Hibrit Grup":
                self._run_hybrid()
            elif self.method == "Genetik":
                self._run_genetic()
            elif self.method == "Bayesian":
                self._run_bayesian()
            else:
                self.error.emit(f"Bilinmeyen yöntem: {self.method}")
                
            # restore data after run
            self.data = original_data
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
            self.data = original_data

    def _run_sequential_layer(self):
        """Strateji 4 (TOMA) Sequential Layer Optimizasyonu (Full Parametreler)"""
        from src.optimization.strategy4_optimizer import fast_backtest_strategy4, IndicatorCache
        import numpy as np
        
        # 1. Veri Hazirligi
        self._emit_progress(1, "Veri analiz ediliyor...")
        dates = self.data['DateTime'].values
        closes = self.data['Kapanis'].values
        mask_arr = np.ones(len(closes), dtype=bool) 
        
        # MASK (Vade/Tatil)
        # MASK (Vade/Tatil)
        try:
            from src.engine.data import OHLCV
            vade_tipi = self.config.get('vade_tipi', "ENDEKS")
            
            # OHLCV expects standard English lowercase columns
            # Create a clean DataFrame (to avoid duplicate columns if both En/Tr exist)
            df_temp = pd.DataFrame()
            
            def _get_col(df, candidates):
                for c in candidates:
                    if c in df.columns: return df[c]
                return None
            
            df_temp['datetime'] = _get_col(self.data, ['DateTime', 'Date', 'Tarih', 'datetime'])
            df_temp['open'] = _get_col(self.data, ['Acilis', 'Open', 'open'])
            df_temp['high'] = _get_col(self.data, ['Yuksek', 'High', 'high'])
            df_temp['low'] = _get_col(self.data, ['Dusuk', 'Low', 'low'])
            df_temp['close'] = _get_col(self.data, ['Kapanis', 'Close', 'close'])
            df_temp['volume'] = _get_col(self.data, ['Hacim', 'Volume', 'volume', 'Lot'])
            
            # Ensure valid data
            df_temp.dropna(subset=['datetime', 'close'], inplace=True)
            
            ohlcv = OHLCV(df_temp)
            mask_arr = ohlcv.get_trading_mask(vade_tipi)
            self._emit_progress(1, f"Maske uygulandı: {vade_tipi} (Aktif bar: {np.sum(mask_arr)}/{len(mask_arr)})")
        except Exception as e:
            print(f"Mask Error: {e}")
            mask_arr = np.ones(len(closes), dtype=bool) 
        
        # 2. Cache
        self._emit_progress(2, "Cache baslatiliyor...")
        cache = IndicatorCache(self.data)
        grid = self.param_ranges
        
        # 3. Parametreleri Hazirla
        # Phase 1 Params
        toma_ranges = [(p, o) for p in grid.get('toma_period', [97]) for o in grid.get('toma_opt', [1.5])]
        hhv1_ranges = grid.get('hhv1_period', [20])
        llv1_ranges = grid.get('llv1_period', [20])
        
        # Phase 2 Params (Globals + L1)
        mom_periods = grid.get('mom_period', [1900])
        trix_periods = grid.get('trix_period', [120])
        mom_high_ranges = [(mh, lb) for mh in grid.get('mom_limit_high', [101.5]) for lb in grid.get('trix_lb1', [145])]
        hhv2_ranges = grid.get('hhv2_period', [150])
        llv2_ranges = grid.get('llv2_period', [190])
        
        # Phase 3 Params (L2 + Risk)
        mom_low_ranges = [(ml, lb) for ml in grid.get('mom_limit_low', [99.0]) for lb in grid.get('trix_lb2', [160])]
        hhv3_ranges = grid.get('hhv3_period', [150])
        llv3_ranges = grid.get('llv3_period', [190])
        risk_ranges = [(ka, iz) for ka in grid.get('kar_al', [0.0]) for iz in grid.get('iz_stop', [0.0])]
        
        # Dummy Params for disabled layers
        p_mom_low_dummy = -9999.0
        p_mom_high_dummy = 9999.0
        
        # ==============================================================================
        # PHASE 1: TOMA Scan (TOMA Params + Layer 3 Filters)
        # ==============================================================================
        self._emit_progress(5, f"FAZ 1: TOMA ve Filtre Taramasi ({len(toma_ranges)*len(hhv1_ranges)} kombinasyon)...")
        
        # Pre-load required cache for Phase 1 (hhv1/llv1)
        # mom/trix not used in Phase 1 if layers are disabled
        # But fast_backtest expects arrays. Send dummies or safe defaults
        dummy_mom = np.zeros(len(closes))
        dummy_trix = np.zeros(len(closes))
        dummy_hhv2 = np.zeros(len(closes))
        dummy_llv2 = np.zeros(len(closes))
        dummy_hhv3 = np.zeros(len(closes))
        dummy_llv3 = np.zeros(len(closes))
        
        best_phase1 = None
        best_p1_score = -float('inf')
        
        counter = 0
        total_p1 = len(toma_ranges) * len(hhv1_ranges) * len(llv1_ranges)
        
        for tp, to in toma_ranges:
            toma_val, toma_trend = cache.get_toma(tp, to)
            
            for h1p in hhv1_ranges:
                hhv1 = cache.get_hhv(h1p)
                for l1p in llv1_ranges:
                    llv1 = cache.get_llv(l1p)
                    
                    counter +=1
                    if counter % 50 == 0:
                        self._emit_progress(5 + int(30 * counter/total_p1), f"Faz 1 [{counter}/{total_p1}]: TOMA={tp} Opt={to} H1={h1p} L1={l1p}")
                        if not self._is_running: return

                    res = fast_backtest_strategy4(
                        closes, toma_trend, toma_val,
                        hhv1, llv1, dummy_hhv2, dummy_llv2, dummy_hhv3, dummy_llv3,
                        dummy_mom, dummy_trix, mask_arr,
                        p_mom_low_dummy, p_mom_high_dummy,
                        100, 100, # Dummy Trix LBs
                        0.0, 0.0 # No Risk
                    )
                    
                    score = res[0] * res[2] if res[2] > 0 else 0 # NP * PF
                    if score > best_p1_score:
                        best_p1_score = score
                        best_phase1 = {'toma_period': tp, 'toma_opt': to, 'hhv1': h1p, 'llv1': l1p}
                        # Canlı streaming: yeni en iyi bulunduğunda gönder
                        self.partial_results.emit([{
                            'net_profit': res[0], 'trades': res[1], 'pf': res[2],
                            'max_dd': res[3], 'sharpe': res[4], 'fitness': score,
                            'toma_period': tp, 'toma_opt': to, 'hhv1_period': h1p, 'llv1_period': l1p,
                            '_phase': 'Faz 1 (TOMA)'
                        }])
        
        if not best_phase1:
            self.error.emit("Faz 1 sonuc bulunamadi.")
            return

        self._emit_progress(35, f"Faz 1 En Iyi: TOMA {best_phase1['toma_period']}/{best_phase1['toma_opt']}, H1:{best_phase1['hhv1']}")
        
        # ==============================================================================
        # PHASE 2: Layer 1 + Global Indicators (PARALLEL — Lightweight)
        # ==============================================================================
        self._emit_progress(35, "FAZ 2: Global Indikatorler ve Layer 1 (PARALEL)...")
        
        # Fix Phase 1
        fix_tp, fix_to = best_phase1['toma_period'], best_phase1['toma_opt']
        toma_val, toma_trend = cache.get_toma(fix_tp, fix_to)
        hhv1 = cache.get_hhv(best_phase1['hhv1'])
        llv1 = cache.get_llv(best_phase1['llv1'])
        
        # Pre-compute all indicator arrays into shared_data dict
        # This dict is serialized ONCE to each worker via initializer (not per-task!)
        shared_data_p2 = {
            'closes': closes,
            'toma_trend': toma_trend,
            'toma_val': toma_val,
            'hhv1': hhv1,
            'llv1': llv1,
            'mask': mask_arr,
            'mom': {mp: cache.get_mom(mp) for mp in mom_periods},
            'trix': {tp: cache.get_trix(tp) for tp in trix_periods},
            'hhv': {hp: cache.get_hhv(hp) for hp in set(list(hhv2_ranges) + list(hhv3_ranges))},
            'llv': {lp: cache.get_llv(lp) for lp in set(list(llv2_ranges) + list(llv3_ranges))},
        }
        
        # Build flat task list — ONLY SCALARS (6 values per task, ~50 bytes)
        p2_tasks = []
        for mom_p in mom_periods:
            for trix_p in trix_periods:
                for h2p in hhv2_ranges:
                    for l2p in llv2_ranges:
                        for mh, lb1 in mom_high_ranges:
                            # TRIX Lookback Constraint: lb1 <= trix_p * 3
                            if lb1 > trix_p * 3:
                                continue
                            p2_tasks.append((mom_p, trix_p, h2p, l2p, mh, lb1))
        
        self._emit_progress(36, f"Faz 2: {len(p2_tasks)} kombinasyon ({len(p2_tasks)*50//1024} KB bellek)...")
        
        best_phase2 = None
        best_p2_score = -float('inf')
        
        if len(p2_tasks) > 0:
            from multiprocessing import Pool, cpu_count
            from src.optimization.strategy4_optimizer import s4_parallel_init, s4_p2_eval
            
            n_workers = min(self.n_parallel or 16, cpu_count())
            
            try:
                self.pool = Pool(
                    processes=n_workers,
                    initializer=s4_parallel_init,
                    initargs=(shared_data_p2,),
                    maxtasksperchild=500
                )
                done = 0
                for result in self.pool.imap_unordered(s4_p2_eval, p2_tasks, chunksize=max(1, len(p2_tasks) // (n_workers * 4))):
                    done += 1
                    
                    # Sonucu DÖNGÜ İÇİNDE topla (eski kod bunu yapmıyordu!)
                    if result is not None:
                        score, params = result
                        if score > best_p2_score:
                            best_p2_score = score
                            best_phase2 = params
                    
                    if done % 200 == 0:
                        prog = 36 + int(28 * done / len(p2_tasks))
                        # Son sonucun parametrelerini goster
                        if result is not None:
                            _, rp = result if result else (0, {})
                            p2_txt = f"Mom={rp.get('mom_period','')} Trix={rp.get('trix_period','')} MH={rp.get('mom_limit_high','')} LB1={rp.get('trix_lb1','')} H2={rp.get('hhv2','')} L2={rp.get('llv2','')}"
                        else:
                            p2_txt = ""
                        self._emit_progress(prog, f"Faz 2 [{done}/{len(p2_tasks)}]: {p2_txt}")
                        # Canlı streaming: en iyi sonucu gönder
                        if best_phase2:
                            self.partial_results.emit([{
                                'net_profit': best_p2_score, 'fitness': best_p2_score,
                                **best_phase2, '_phase': 'Faz 2 (Layer 1)'
                            }])
                        if not self._is_running:
                            self.pool.terminate()
                            self.pool = None
                            return
            finally:
                if self.pool:
                    self.pool.close()
                    self.pool.join()
                    self.pool = None

        if not best_phase2:
             # Fallback defaults if search failed or empty
             best_phase2 = {'mom_period': 1900, 'trix_period': 120, 'mom_limit_high': 101.5, 'trix_lb1': 145, 'hhv2':150, 'llv2':190}
             
        self._emit_progress(65, f"Faz 2 En Iyi: MomP {best_phase2['mom_period']}, Limit {best_phase2['mom_limit_high']}")

        # ==============================================================================
        # PHASE 3: Layer 2 + Risk (PARALLEL — Lightweight)
        # ==============================================================================
        self._emit_progress(65, "FAZ 3: Layer 2 (Mom Low) ve Risk (PARALEL)...")
        
        # Fix Phase 2
        fix_mom_p = best_phase2['mom_period']
        fix_trix_p = best_phase2['trix_period']
        fix_mh = best_phase2['mom_limit_high']
        fix_lb1 = best_phase2['trix_lb1']
        
        mom_arr = cache.get_mom(fix_mom_p)
        trix_arr = cache.get_trix(fix_trix_p)
        hhv2 = cache.get_hhv(best_phase2['hhv2'])
        llv2 = cache.get_llv(best_phase2['llv2'])
        
        # Update shared data for Phase 3 (reuse arrays from Phase 2, add fixed Phase 2 arrays)
        shared_data_p3 = {
            'closes': closes,
            'toma_trend': toma_trend,
            'toma_val': toma_val,
            'hhv1': hhv1,
            'llv1': llv1,
            'mask': mask_arr,
            'hhv2_fixed': hhv2,
            'llv2_fixed': llv2,
            'mom_fixed': mom_arr,
            'trix_fixed': trix_arr,
            # HHV/LLV dict for Phase 3 variable arrays
            'hhv': {hp: cache.get_hhv(hp) for hp in hhv3_ranges},
            'llv': {lp: cache.get_llv(lp) for lp in llv3_ranges},
        }
        
        # Metadata dict (sent once per task, tiny ~200 bytes)
        p3_meta = {
            'fix_mh': fix_mh, 'fix_lb1': fix_lb1,
            'fix_tp': fix_tp, 'fix_to': fix_to,
            'p1_hhv1': best_phase1['hhv1'], 'p1_llv1': best_phase1['llv1'],
            'fix_mom_p': fix_mom_p, 'fix_trix_p': fix_trix_p,
            'p2_hhv2': best_phase2['hhv2'], 'p2_llv2': best_phase2['llv2'],
        }
        
        # Build flat task list — ONLY SCALARS + tiny meta dict
        p3_tasks = []
        for h3p in hhv3_ranges:
            for l3p in llv3_ranges:
                for ml, lb2 in mom_low_ranges:
                    # Relaxed TRIX Constraint for Phase 3: 
                    # Phase 2 might have picked a small TRIX Period, don't kill Phase 3 exploration.
                    # We accept all lb2 ranges defined by user.
                    for ka, iz in risk_ranges:
                        p3_tasks.append((h3p, l3p, ml, lb2, ka, iz, p3_meta))
        
        self._emit_progress(66, f"Faz 3: {len(p3_tasks)} kombinasyon ({len(p3_tasks)*200//1024//1024} MB bellek)...")
        
        final_results = []
        
        if len(p3_tasks) > 0:
            from multiprocessing import Pool, cpu_count
            from src.optimization.strategy4_optimizer import s4_parallel_init, s4_p3_eval
            
            n_workers = min(self.n_parallel or 16, cpu_count())
            
            try:
                self.pool = Pool(
                    processes=n_workers,
                    initializer=s4_parallel_init,
                    initargs=(shared_data_p3,),
                    maxtasksperchild=500
                )
                done = 0
                for result in self.pool.imap_unordered(s4_p3_eval, p3_tasks, chunksize=max(1, len(p3_tasks) // (n_workers * 4))):
                    done += 1
                    if done % 500 == 0:
                        prog = 66 + int(33 * done / len(p3_tasks))
                        # Son sonucun parametrelerini goster
                        if result is not None:
                            p3_txt = f"ML={result.get('mom_limit_low','')} LB2={result.get('trix_lb2','')} H3={result.get('hhv3','')} L3={result.get('llv3','')} KA={result.get('kar_al','')} IZ={result.get('iz_stop','')}"
                        else:
                            p3_txt = ""
                        self._emit_progress(prog, f"Faz 3 [{done}/{len(p3_tasks)}]: {p3_txt}")
                        if not self._is_running:
                            self.pool.terminate()
                            self.pool = None
                            return
                    
                    if result is not None:
                        final_results.append(result)
                    
                    # Canlı streaming: her 2000 sonuçta top 50 gönder
                    if done % 2000 == 0 and final_results:
                        from src.optimization.fitness import quick_fitness as _qf
                        temp = sorted(final_results, key=lambda x: x.get('net_profit', 0), reverse=True)[:50]
                        for _r in temp:
                            if 'fitness' not in _r:
                                _r['fitness'] = _qf(_r['net_profit'], _r.get('pf',0), _r.get('max_dd',0), _r.get('trades',0), sharpe=_r.get('sharpe',0))
                        self.partial_results.emit(temp)

            finally:
                if self.pool:
                    self.pool.close()
                    self.pool.join()
                    self.pool = None
                    


        # Final Sort - use fitness if sharpe is available, else net_profit
        from src.optimization.fitness import quick_fitness
        for r in final_results:
            sh = r.get('sharpe', 0.0)
            r['fitness'] = quick_fitness(
                r['net_profit'], r['pf'], r['max_dd'], r['trades'],
                sharpe=sh, commission=0.0, slippage=0.0
            )
        
        final_results.sort(key=lambda x: x.get('fitness', x['net_profit']), reverse=True)
        top_results = final_results[:500]  # Show Top 500
        
        # OOS Validation for S4
        if self.do_oos and self.test_data is not None and top_results:
            self._emit_progress(98, "S4 Test verisinde validasyon yapiliyor...")
            for r in top_results[:50]:  # Top 50 icin validasyon
                oos_res = self._validate_s4_result(r)
                r.update(oos_res)
        
        self._emit_progress(100, "Optimizasyon Tamamlandi!")
        self.result_ready.emit(top_results)


    
    def _run_hybrid(self):
        """Hibrit Grup optimizasyonu"""
        from src.optimization.hybrid_group_optimizer import (
            HybridGroupOptimizer, IndicatorCache, STRATEGY1_GROUPS, STRATEGY2_GROUPS, STRATEGY3_GROUPS, ParameterGroup
        )
        
        self._emit_progress(5, "Cache olusturuluyor...")
        cache = IndicatorCache(self.data)
        
        # Strateji seçiminden orijinal grupları al
        original_groups = {0: STRATEGY1_GROUPS, 1: STRATEGY2_GROUPS, 2: STRATEGY3_GROUPS}.get(self.strategy_index, STRATEGY1_GROUPS)
        strategy_name = {0: "Strateji 1", 1: "Strateji 2", 2: "Strateji 3 (Paradise)"}.get(self.strategy_index, "Strateji 1")
        
        # UI'dan gelen range'leri gruplara uyarla (Dynamic Sync)
        synced_groups = []
        for og in original_groups:
            # Bu grubun parametrelerini UI'dan gelenlerle güncelle
            synced_params = {}
            for p_name in og.params.keys():
                if p_name in self.param_ranges:
                    # UI'dan gelen tam listeyi (min-max-step sonucu olan) al
                    synced_params[p_name] = self.param_ranges[p_name]
                else:
                    # UI'da olmayan parametre (olmamalı ama güvenlik için)
                    synced_params[p_name] = og.params[p_name]
            
            synced_groups.append(ParameterGroup(
                name=og.name,
                params=synced_params,
                is_independent=og.is_independent,
                default_values=og.default_values
            ))

        self._emit_progress(10, f"Hibrit Optimizer baslatiliyor ({strategy_name})...")
        
        # Canlı streaming wrapper
        _last_stream_pct = [0]
        def _hybrid_progress(pct, msg):
            self._emit_progress(pct, msg)
            # Her %10 ilerleme için partial sonuçları gönder
            if pct - _last_stream_pct[0] >= 10:
                _last_stream_pct[0] = pct
                try:
                    top = optimizer.get_best_results(top_n=50)
                    if top:
                        self.partial_results.emit(top)
                except Exception:
                    pass
        
        optimizer = HybridGroupOptimizer(
            synced_groups, 
            process_id=self.process_id, 
            strategy_index=self.strategy_index,
            is_cancelled_callback=lambda: not self._is_running,
            on_progress_callback=_hybrid_progress,
            n_parallel=self.n_parallel,
            commission=self.commission,
            slippage=self.slippage
        )
        self.current_optimizer = optimizer
        
        self._emit_progress(10, "Iterative Coordinate Descent baslatiliyor...")
        results = optimizer.run(turbo=True, iterative=True, max_rounds=3)

        
        if not self._is_running:
            return
        
        self._emit_progress(95, "Sonuçlar derleniyor...")
        results = optimizer.get_best_results(top_n=20)
        
        # OOS Validasyon
        if self.do_oos and self.test_data is not None:
             self._emit_progress(98, "Test verisinde validasyon yapiliyor...")
             excluded = {'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 'group', 'win_count', 'win_rate'}
             
             for res in results:
                 # Parametreleri ayikla
                 params = {k: v for k, v in res.items() if k not in excluded}
                 oos_res = self._validate_result(params)
                 res.update(oos_res)
        
        self._emit_progress(100, "Tamamlandı!")
        self.result_ready.emit(results)
    
    def _run_genetic(self):
        """Genetik Algoritma optimizasyonu - Her iki strateji için"""
        from src.optimization.genetic_optimizer import GeneticOptimizer, GeneticConfig
        
        strategy_name = {0: "Strateji 1", 1: "Strateji 2", 2: "Paradise"}.get(self.strategy_index, "Strateji")
        
        # Cascade modu kontrolu
        if self.narrowed_ranges:
            self._emit_progress(5, f"Genetik (CASCADE) baslatiliyor ({strategy_name})...")
            print(f"[CASCADE] Genetik dar aralikta calisacak: {len(self.narrowed_ranges)} parametre")
        else:
            self._emit_progress(5, f"Genetik Algoritma baslatiliyor ({strategy_name})...")
        
        config = GeneticConfig(
            population_size=50,
            generations=30,
            elite_ratio=0.1,
            crossover_rate=0.8,
            mutation_rate=0.15
        )
        
        optimizer = GeneticOptimizer(
            self.data, 
            config, 
            strategy_index=self.strategy_index,
            n_parallel=self.n_parallel,
            commission=self.commission,
            slippage=self.slippage,
            is_cancelled_callback=lambda: not self._is_running,
            narrowed_ranges=self.narrowed_ranges  # Cascade icin dar aralik
        )
        self.current_optimizer = optimizer
        
        # Nesil bazlı ilerleme ve canlı streaming için callback
        def on_gen_complete(gen, max_gen, best_fit):
            progress = 10 + int((gen / max_gen) * 85)
            # En iyi bireyin parametrelerini goster
            param_txt = ""
            if hasattr(optimizer, 'population_results') and optimizer.population_results:
                try:
                    best_ind = max(optimizer.population_results, key=lambda x: x.get('fitness', x.get('net_profit', 0)))
                    # Fitness ve performans haric parametreleri al
                    skip_keys = {'net_profit','trades','pf','max_dd','sharpe','fitness','win_rate','avg_win','avg_loss'}
                    params_only = {k: v for k, v in best_ind.items() if k not in skip_keys}
                    # Kisa format: key=val
                    param_txt = " | " + " ".join(f"{k}={v}" for k, v in list(params_only.items())[:8])
                except Exception:
                    pass
            self._emit_progress(progress, f"Nesil {gen}/{max_gen} - Fit: {best_fit:,.0f}{param_txt}")
            # Her 3 nesilde top sonuçları gönder
            if gen % 3 == 0 and hasattr(optimizer, 'population_results'):
                try:
                    top = sorted(optimizer.population_results, key=lambda x: x.get('fitness', x.get('net_profit', 0)), reverse=True)[:50]
                    if top:
                        self.partial_results.emit(top)
                except Exception:
                    pass
        
        optimizer.on_generation_complete = on_gen_complete
        
        self._emit_progress(10, "Optimizasyon çalışıyor...")
        result = optimizer.run(verbose=False)
        
        self._emit_progress(100, "Genetik optimizasyon tamamlandı!")
        
        # Genetik sonucu dict olarak döner, best_result'ı çıkar
        if result and result.get('best_result'):
            best = result['best_result']
            best_params = result.get('best_params', {})
            # Sonucu GUI formatına çevir
            formatted_result = {
                'net_profit': best.get('net_profit', 0),
                'trades': best.get('trades', 0),
                'pf': best.get('pf', 0),
                'max_dd': best.get('max_dd', 0),
                'sharpe': best.get('sharpe', 0),
                'fitness': best.get('fitness', 0),
                **best_params
            }
            # OOS Validasyon
            if self.do_oos and self.test_data is not None:
                self._emit_progress(98, "Test verisinde validasyon yapiliyor...")
                oos_res = self._validate_result(best_params)
                formatted_result.update(oos_res)
                
            self.result_ready.emit([formatted_result])
        else:
            self.result_ready.emit([])
    
    def _run_bayesian(self):
        """Bayesian (Optuna) optimizasyonu - Her iki strateji için"""
        from src.optimization.bayesian_optimizer import BayesianOptimizer
        
        strategy_name = {0: "Strateji 1", 1: "Strateji 2", 2: "Paradise"}.get(self.strategy_index, "Strateji")
        
        # Cascade modu kontrolu
        if self.narrowed_ranges:
            self._emit_progress(5, f"Bayesian (CASCADE) baslatiliyor ({strategy_name})...")
            print(f"[CASCADE] Bayesian dar aralikta calisacak: {len(self.narrowed_ranges)} parametre")
        else:
            self._emit_progress(5, f"Bayesian Optimizer baslatiliyor ({strategy_name})...")
        
        n_trials = 500  # Deneme sayısı (Artırıldı)
        
        self._emit_progress(10, f"Optuna calismasi olusturuluyor...")
        optimizer = BayesianOptimizer(
            self.data, 
            n_trials=n_trials, 
            strategy_index=self.strategy_index,
            n_parallel=self.n_parallel,
            commission=self.commission,
            slippage=self.slippage,
            is_cancelled_callback=lambda: not self._is_running,
            narrowed_ranges=self.narrowed_ranges  # Cascade icin dar aralik
        )
        self.current_optimizer = optimizer
        
        # Trial bazlı ilerleme ve canlı streaming için callback
        def on_trial_complete(trial_no, max_trials, best_fit):
            progress = 10 + int((trial_no / max_trials) * 85)
            # En iyi denemenin parametrelerini goster
            param_txt = ""
            if hasattr(optimizer, 'all_results') and optimizer.all_results:
                try:
                    best_trial = max(optimizer.all_results, key=lambda x: x.get('fitness', x.get('net_profit', 0)))
                    skip_keys = {'net_profit','trades','pf','max_dd','sharpe','fitness','win_rate','avg_win','avg_loss'}
                    params_only = {k: v for k, v in best_trial.items() if k not in skip_keys}
                    param_txt = " | " + " ".join(f"{k}={v}" for k, v in list(params_only.items())[:8])
                except Exception:
                    pass
            self._emit_progress(progress, f"Deneme {trial_no}/{max_trials} - Fit: {best_fit:,.0f}{param_txt}")
            # Her 10 denemede top sonuçları gönder
            if trial_no % 10 == 0 and hasattr(optimizer, 'all_results'):
                try:
                    top = sorted(optimizer.all_results, key=lambda x: x.get('fitness', x.get('net_profit', 0)), reverse=True)[:50]
                    if top:
                        self.partial_results.emit(top)
                except Exception:
                    pass
            
        optimizer.on_trial_complete = on_trial_complete
        
        self._emit_progress(15, "Akıllı arama başlıyor...")
        result = optimizer.run(verbose=False)
        
        self._emit_progress(100, "Bayesian optimizasyon tamamlandı!")
        
        # Bayesian sonucu dict olarak döner, best_result'ı çıkar
        if result and result.get('best_result'):
            best = result['best_result']
            best_params = result.get('best_params', {})
            # Sonucu GUI formatına çevir
            formatted_result = {
                'net_profit': best.get('net_profit', 0),
                'trades': best.get('trades', 0),
                'pf': best.get('pf', 0),
                'max_dd': best.get('max_dd', 0),
                'sharpe': best.get('sharpe', 0),
                'fitness': best.get('fitness', 0),
                **best_params
            }
            # OOS Validasyon
            if self.do_oos and self.test_data is not None:
                self._emit_progress(98, "Test verisinde validasyon yapiliyor...")
                oos_res = self._validate_result(best_params)
                formatted_result.update(oos_res)
            
            self.result_ready.emit([formatted_result])
        else:
            self.result_ready.emit([])
    
    def stop(self):
        """Worker'ı ve proses'i durdur"""
        self._is_running = False
        if self.pool:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception as e:
                print(f"Pool terminate error: {e}")
        
        # Obje tabanlı optimizer durdurma (Genetic/Bayesian/Hybrid)
        if self.current_optimizer and hasattr(self.current_optimizer, 'stop'):
             try:
                 self.current_optimizer.stop()
             except:
                 pass

    def _validate_result(self, params):
        """Test verisi uzerinde validasyon yap"""
        if self.test_data is None: return {}
        
        try:
            from src.optimization.hybrid_group_optimizer import IndicatorCache
            test_cache = IndicatorCache(self.test_data)
            
            if self.strategy_index == 0:
                from src.strategies.score_based import ScoreBasedStrategy
                strategy = ScoreBasedStrategy.from_config_dict(test_cache, params)
            elif self.strategy_index == 2:
                from src.strategies.paradise_strategy import ParadiseStrategy
                strategy = ParadiseStrategy.from_config_dict(test_cache, params)
            elif self.strategy_index == 3:
                # S4: Use fast_backtest_strategy4 directly
                return self._validate_s4_result(params)
            else:
                from src.strategies.ars_trend_v2 import ARSTrendStrategyV2
                strategy = ARSTrendStrategyV2.from_config_dict(test_cache, params)
                
            signals, ex_long, ex_short = strategy.generate_all_signals()
            
            # Backtest
            trading_days = 252.0
            if test_cache.dates and len(test_cache.dates) > 1:
                try:
                    trading_days = (test_cache.dates[-1] - test_cache.dates[0]).days
                except: pass

            net, trades, pf, dd, sharpe = self._simple_backtest(
                test_cache.closes, signals, ex_long, ex_short, trading_days=trading_days
            )
            
            return {
                'test_net': net,
                'test_trades': trades,
                'test_pf': pf,
                'test_dd': dd,
                'test_sharpe': sharpe
            }
        except Exception as e:
            print(f"Validasyon hatasi: {e}")
            return {}

    def _validate_s4_result(self, params):
        """S4 icin OOS validasyon: test verisinde fast_backtest_strategy4 calistir"""
        if self.test_data is None: return {}
        
        try:
            import numpy as np
            from src.optimization.strategy4_optimizer import fast_backtest_strategy4, IndicatorCache as S4IndicatorCache
            
            test_cache = S4IndicatorCache(self.test_data)
            closes = test_cache.closes
            
            # TOMA
            tp = int(params.get('toma_period', 97))
            to = float(params.get('toma_opt', 1.5))
            toma_val, toma_trend = test_cache.get_toma(tp, to)
            
            # HHV/LLV
            hhv1 = test_cache.get_hhv(int(params.get('hhv1_period', 20)))
            llv1 = test_cache.get_llv(int(params.get('llv1_period', 20)))
            hhv2 = test_cache.get_hhv(int(params.get('hhv2_period', 150)))
            llv2 = test_cache.get_llv(int(params.get('llv2_period', 190)))
            hhv3 = test_cache.get_hhv(int(params.get('hhv3_period', 150)))
            llv3 = test_cache.get_llv(int(params.get('llv3_period', 190)))
            
            # Indicators
            mom_arr = test_cache.get_mom(int(params.get('mom_period', 1900)))
            trix_arr = test_cache.get_trix(int(params.get('trix_period', 120)))
            
            # Mask (all true for test)
            mask_arr = np.ones(len(closes), dtype=bool)
            
            ml = float(params.get('mom_limit_low', 99.0))
            mh = float(params.get('mom_limit_high', 101.5))
            lb1 = int(params.get('trix_lb1', 145))
            lb2 = int(params.get('trix_lb2', 160))
            ka = float(params.get('kar_al', 0.0))
            iz = float(params.get('iz_stop', 0.0))
            
            np_val, tr, pf, dd, sh = fast_backtest_strategy4(
                closes, toma_trend, toma_val,
                hhv1, llv1, hhv2, llv2, hhv3, llv3,
                mom_arr, trix_arr, mask_arr,
                ml, mh, lb1, lb2, ka, iz
            )
            
            return {
                'test_net': np_val,
                'test_trades': tr,
                'test_pf': pf,
                'test_dd': dd,
                'test_sharpe': sh
            }
        except Exception as e:
            print(f"S4 Validasyon hatasi: {e}")
            return {}

    def _simple_backtest(self, closes, signals, ex_long, ex_short, trading_days: float = 252.0):
        """Basit backtest ve Sharpe hesabi"""
        pos, entry_price = 0, 0.0
        gross_profit, gross_loss, trades = 0.0, 0.0, 0
        cost = self.commission + self.slippage
        
        trade_returns = []
        peak_equity = 0.0
        current_equity = 0.0
        max_dd = 0.0
        
        for i in range(len(closes)):
            pnl = 0.0
            
            # Exit
            if pos == 1 and ex_long[i]:
                pnl = (closes[i] - entry_price) - cost
                pos = 0; trades += 1
            elif pos == -1 and ex_short[i]:
                pnl = (entry_price - closes[i]) - cost
                pos = 0; trades += 1
                
            if pnl != 0:
                trade_returns.append(pnl)
                if pnl > 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                
                current_equity += pnl
                if current_equity > peak_equity: peak_equity = current_equity
                dd = peak_equity - current_equity
                if dd > max_dd: max_dd = dd
            
            # Entry (Ayni bar'da giris yok varsayimi - basitlik icin)
            if pos == 0:
                if signals[i] == 1: pos = 1; entry_price = closes[i]
                elif signals[i] == -1: pos = -1; entry_price = closes[i]
                
        net = gross_profit - gross_loss
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 999
        
        import numpy as np
        from src.optimization.fitness import calculate_sharpe
        sharpe = 0.0
        if len(trade_returns) > 1:
            if trading_days < 1: trading_days = 252.0
            trades_per_year_metric = len(trade_returns) * (252.0 / trading_days)
            sharpe = calculate_sharpe(np.array(trade_returns), trades_per_year=trades_per_year_metric)
            
        return net, trades, pf, max_dd, sharpe


class OptimizerPanel(QWidget):
    """Optimizasyon paneli - Parametre aralıkları ile"""
    
    optimization_complete = Signal(list)
    
    CHECKPOINT_FILE = 'optimizer_checkpoint.json'
    
    def __init__(self):
        super().__init__()
        self.config = {}
        self.data = None
        self.worker = None
        self.group_widgets = {}
        self.current_process_id = None
        self.optimization_queue = []
        self.hybrid_results = []  # Hibrit sonuclarini sakla (Cascade icin)
        self._stop_requested = False  # Stop button flag
        self._queue_total = 0  # Total items for global progress
        self.live_monitor_win = None  # Canlı izleme penceresi
        
        # Preset Manager
        from src.optimization.preset_manager import PresetManager
        self.preset_manager = PresetManager()
        
        # Timer için
        from PySide6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_timer)
        
        self._setup_ui()
        
        # Baslangicta checkpoint kontrolu (UI hazir olduktan sonra)
        from PySide6.QtCore import QTimer as _QT
        _QT.singleShot(500, self._check_checkpoint)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Üst panel - Yöntem ve genel ayarlar
        top_row = QHBoxLayout()
        
        # Süreç seçimi
        top_row.addWidget(QLabel("Süreç:"))
        self.process_combo = QComboBox()
        self.process_combo.setMinimumWidth(200)
        self.process_combo.currentTextChanged.connect(self._on_process_changed)
        top_row.addWidget(self.process_combo)
        
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_processes)
        top_row.addWidget(refresh_btn)
        
        top_row.addSpacing(20)
        
        # Periyot secimi (timeframe scaling icin)
        top_row.addWidget(QLabel("Periyot:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1 dk", "5 dk", "15 dk", "60 dk"])
        self.period_combo.setCurrentText("5 dk")
        self.period_combo.currentIndexChanged.connect(self._on_period_changed)
        top_row.addWidget(self.period_combo)
        
        top_row.addSpacing(10)
        
        # Strateji seçimi
        top_row.addWidget(QLabel("Strateji:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Strateji 1 - Gatekeeper", "Strateji 2 - ARS Trend v2", "Strateji 3 - Paradise", "Strateji 4 - TOMA + Momentum"])
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        top_row.addWidget(self.strategy_combo)
        
        top_row.addSpacing(20)
        
        # Yöntem
        top_row.addWidget(QLabel("Yontem:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Hibrit Grup", "Genetik", "Bayesian"])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        top_row.addWidget(self.method_combo)
        
        top_row.addSpacing(20)
        
        # Validasyon Ayarları
        top_row.addWidget(QLabel("Validasyon:"))
        self.validation_check = QCheckBox("Auto-Split")
        self.validation_check.setChecked(True)
        top_row.addWidget(self.validation_check)
        
        self.split_spin = QSpinBox()
        self.split_spin.setRange(50, 90)
        self.split_spin.setValue(70)
        self.split_spin.setSuffix("% Train")
        top_row.addWidget(self.split_spin)

        top_row.addSpacing(20)
        
        # Paralel işlem
        top_row.addWidget(QLabel("Paralel:"))
        self.workers_spin = QComboBox()
        self.workers_spin.addItems(["8", "16", "24", "32"])
        self.workers_spin.setCurrentText("32")
        top_row.addWidget(self.workers_spin)
        
        top_row.addStretch()
        layout.addLayout(top_row)
        
        # Splitter (Parametre grupları | Sonuçlar)
        splitter = QSplitter(Qt.Horizontal)
        
        # Sol - Parametre grupları (scrollable)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMinimumHeight(300)  # Minimum yükseklik
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setSpacing(10)
        self.groups_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(self.groups_container)
        
        left_layout.addWidget(scroll, 1)  # Stretch factor
        
        # Özet ve kontrol
        summary_group = self._create_summary_group()
        left_layout.addWidget(summary_group)
        
        splitter.addWidget(left_widget)
        
        # Sağ - Sonuçlar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        results_group = self._create_results_group()
        right_layout.addWidget(results_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([450, 550])
        
        layout.addWidget(splitter)
        
        # İlk stratejiyi yükle
        # self._on_strategy_changed(0) -> Artık timer ile yapılacak
        
        # Debounce timer for strategy params setup
        self.setup_timer = QTimer()
        self.setup_timer.setSingleShot(True)
        self.setup_timer.setInterval(200) # 200ms debounce
        self.setup_timer.timeout.connect(self._setup_strategy_params)
        
        # Baslangic tetiklemesi
        self.setup_timer.start()
    
    def _create_summary_group(self) -> QGroupBox:
        """Özet ve kontrol grubu"""
        group = QGroupBox("Ozet ve Kontrol")
        layout = QVBoxLayout(group)
        
        # Kombinasyon sayısı
        self.combo_label = QLabel("Toplam kombinasyon: ~ hesaplanmadi")
        layout.addWidget(self.combo_label)
        
        # Maliyet ayarları
        cost_row = QHBoxLayout()
        cost_row.addWidget(QLabel("Komisyon:"))
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 500)
        self.commission_spin.setValue(0)
        self.commission_spin.setSuffix(" pt")
        cost_row.addWidget(self.commission_spin)
        
        cost_row.addWidget(QLabel("Kayma:"))
        self.slippage_spin = QDoubleSpinBox()
        self.slippage_spin.setRange(0, 500)
        self.slippage_spin.setValue(0)
        self.slippage_spin.setSuffix(" pt")
        cost_row.addWidget(self.slippage_spin)
        layout.addLayout(cost_row)
        
        # Progress bar (Step)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Global Progress bar (Queue)
        self.global_progress_bar = QProgressBar()
        self.global_progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #673AB7; border-radius: 3px; text-align: center; background: #f5f5f5; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #9C27B0, stop:1 #673AB7); }"
        )
        self.global_progress_bar.setFormat("Genel: %p%")
        self.global_progress_bar.setVisible(False)  # Only show during Run All
        layout.addWidget(self.global_progress_bar)
        
        # Live Best Result Monitor (2-satır: tarama + en iyi)
        self.live_monitor_frame = QFrame()
        self.live_monitor_frame.setStyleSheet(
            "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1a237e, stop:1 #283593); "
            "border-radius: 6px; padding: 8px; margin: 4px 0; }"
        )
        monitor_vlayout = QVBoxLayout(self.live_monitor_frame)
        monitor_vlayout.setContentsMargins(10, 6, 10, 6)
        monitor_vlayout.setSpacing(4)
        
        # Satır 1: Mevcut tarama bilgisi + timer
        scan_row = QHBoxLayout()
        self.live_scan_icon = QLabel("\u2699")  # Gear icon
        self.live_scan_icon.setStyleSheet("font-size: 14px; color: #90CAF9; background: transparent;")
        scan_row.addWidget(self.live_scan_icon)
        self.live_scan_label = QLabel("Tarama baslatiliyor...")
        self.live_scan_label.setStyleSheet("color: #B0BEC5; font-size: 11px; background: transparent;")
        scan_row.addWidget(self.live_scan_label, 1)
        self.live_timer_label = QLabel("")
        self.live_timer_label.setStyleSheet("color: #FFD54F; font-size: 11px; font-weight: bold; background: transparent;")
        self.live_timer_label.setAlignment(Qt.AlignRight)
        scan_row.addWidget(self.live_timer_label)
        monitor_vlayout.addLayout(scan_row)
        
        # Satır 2: En iyi sonuç
        best_row = QHBoxLayout()
        self.live_monitor_icon = QLabel("\u2b50")  # Star emoji
        self.live_monitor_icon.setStyleSheet("font-size: 16px; color: #FFD600; background: transparent;")
        best_row.addWidget(self.live_monitor_icon)
        self.live_monitor_label = QLabel("En Iyi Sonuc bekleniyor...")
        self.live_monitor_label.setStyleSheet("color: #E8EAF6; font-weight: bold; font-size: 12px; background: transparent;")
        best_row.addWidget(self.live_monitor_label, 1)
        monitor_vlayout.addLayout(best_row)
        
        self.live_monitor_frame.setVisible(False)  # Show during optimization
        layout.addWidget(self.live_monitor_frame)
        
        # Status
        self.status_label = QLabel("Hazir")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #1976D2; font-size: 13px; margin: 5px;")
        layout.addWidget(self.status_label)
        
        # Butonlar
        btn_row = QHBoxLayout()
        
        self.calc_btn = QPushButton("Hesapla")
        self.calc_btn.clicked.connect(self._calculate_combinations)
        btn_row.addWidget(self.calc_btn)
        
        self.start_btn = QPushButton("Baslat")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self._start_optimization)
        btn_row.addWidget(self.start_btn)
        
        self.run_all_btn = QPushButton("Tümünü Çalıştır")
        self.run_all_btn.setStyleSheet("background-color: #673AB7; color: white;")
        self.run_all_btn.clicked.connect(self._run_all_optimizers)
        btn_row.addWidget(self.run_all_btn)
        
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_optimization)
        btn_row.addWidget(self.stop_btn)
        
        self.resume_btn = QPushButton("\u25B6 Devam Et")
        self.resume_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.resume_btn.setVisible(False)  # Only visible when checkpoint exists
        self.resume_btn.clicked.connect(self._resume_from_checkpoint)
        btn_row.addWidget(self.resume_btn)
        
        btn_row.addSpacing(20)
        
        # Preset butonları
        self.preset_save_btn = QPushButton("💾 Preset Kaydet")
        self.preset_save_btn.setStyleSheet("background-color: #00796B; color: white; font-size: 11px;")
        self.preset_save_btn.setToolTip("Mevcut parametre aralıklarını kaydet (sembol+periyot+strateji)")
        self.preset_save_btn.clicked.connect(self._save_preset)
        btn_row.addWidget(self.preset_save_btn)
        
        self.preset_load_btn = QPushButton("📂 Preset Yükle")
        self.preset_load_btn.setStyleSheet("background-color: #455A64; color: white; font-size: 11px;")
        self.preset_load_btn.setToolTip("Kaydedilmiş parametre aralıklarını yükle")
        self.preset_load_btn.clicked.connect(self._load_preset)
        btn_row.addWidget(self.preset_load_btn)
        
        layout.addLayout(btn_row)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Sonuçlar grubu - 3 Tab ile (Hibrit, Genetik, Bayesian)"""
        group = QGroupBox("Sonuclar")
        layout = QVBoxLayout(group)
        
        # İstatistikler header
        stats_row = QHBoxLayout()
        self.results_stats_label = QLabel("Henuz sonuc yok")
        stats_row.addWidget(self.results_stats_label)
        stats_row.addStretch()
        
        export_btn = QPushButton("Disa Aktar")
        export_btn.clicked.connect(self._export_results)
        stats_row.addWidget(export_btn)
        layout.addLayout(stats_row)
        
        # Tab Widget (3 metod için)
        from PySide6.QtWidgets import QTabWidget, QTextEdit, QSplitter
        
        self.results_tab_widget = QTabWidget()
        self.method_tables = {}  # Her metod için tablo
        self.method_params_text = {}  # Her metod için parametre gösterimi
        self.method_results = {}  # Her metod için sonuç verisi
        self.method_grid_panels = {}  # Her metod için grid paneli referansı
        
        method_names = ["Hibrit Grup", "Genetik", "Bayesian"]
        
        for method in method_names:
            # Her tab icin basitleştirilmiş layout
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(2, 2, 2, 2)
            tab_layout.setSpacing(2)
            
            # Splitter: Tablo üstte (kompakt), Params altta (geniş)
            splitter = QSplitter(Qt.Vertical)
            
            # Üst kısım: Sonuç tablosu (kompakt - sadece satır kadar yer kaplar)
            table = QTableWidget()
            table.setAlternatingRowColors(True)
            table.setSortingEnabled(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)
            table.setMaximumHeight(120)  # Kompakt tablo
            table.itemSelectionChanged.connect(lambda m=method: self._on_result_selected(m))
            splitter.addWidget(table)
            self.method_tables[method] = table
            
            # Alt kısım: Parametreler - Scrollable ve Grid Panel
            from PySide6.QtWidgets import QScrollArea
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QFrame.NoFrame)
            
            scroll_content = QWidget()
            self.method_params_text[method] = scroll_content
            scroll_layout = QVBoxLayout(scroll_content)
            scroll_layout.setContentsMargins(5, 2, 5, 2)
            scroll_layout.setSpacing(6)
            
            params_group = QGroupBox("Seçili Sonucun Parametre Ayrıntıları")
            params_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 6px; padding: 8px; }")
            params_group_layout = QVBoxLayout(params_group)
            
            # Parametrelerin gösterileceği grid alanı
            grid_panel = QWidget()
            grid_layout = QGridLayout(grid_panel)
            grid_layout.setSpacing(8)
            params_group_layout.addWidget(grid_panel)
            self.method_grid_panels[method] = grid_panel
            
            scroll_layout.addWidget(params_group)
            scroll_layout.addStretch()
            
            scroll_area.setWidget(scroll_content)
            splitter.addWidget(scroll_area)
            
            # Splitter oranlari: Tablo %25, Params %75
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 3)
            
            tab_layout.addWidget(splitter)
            self.results_tab_widget.addTab(tab_widget, method)
            self.method_results[method] = []
        
        layout.addWidget(self.results_tab_widget)
        
        return group
    
    def _on_result_selected(self, method: str):
        """Sonuç tablosunda satır seçildiğinde parametreleri grid paneline yerleştir"""
        table = self.method_tables.get(method)
        params_container = self.method_params_text.get(method) # Scroll content container
        results = self.method_results.get(method, [])
        
        if not table or not params_container or not results:
            return
            
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        if row_idx >= len(results):
            return
            
        result = results[row_idx]
        params = result.get('params', result)
        
        # Grid panelini direkt referanstan al
        grid_panel = self.method_grid_panels.get(method)
        if not grid_panel: return
        grid_layout = grid_panel.layout()
        if not grid_layout: return
        
        # Mevcut widgetları temizle
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # Strateji bazlı grup tanımını al
        # Strateji bazlı grup tanımını al
        strategy_idx = self.strategy_combo.currentIndex()
        if strategy_idx == 0: group_defs = STRATEGY1_PARAM_GROUPS
        elif strategy_idx == 2: group_defs = STRATEGY3_PARAM_GROUPS
        elif strategy_idx == 3: group_defs = STRATEGY4_PARAM_GROUPS
        else: group_defs = STRATEGY2_PARAM_GROUPS
        
        row = 0
        col = 0
        max_cols = 4 # Yan yana 4 parametre
        
        processed_params = set()
        
        # Gruplar halinde ekle
        for group_id, group_info in group_defs.items():
            group_label = group_info.get('label', group_id)
            group_params = group_info.get('params', {})
            
            # Bu gruptan en az bir parametre var mı?
            valid_group = any(p in params for p in group_params)
            if not valid_group: continue
            
            # Grup Başlığı (Yeni satır)
            if col != 0: 
                row += 1
                col = 0
            
            header = QLabel(f"--- {group_label} ---")
            header.setStyleSheet("color: #aaa; font-style: italic; margin-top: 5px;")
            grid_layout.addWidget(header, row, 0, 1, max_cols)
            row += 1
            
            for p_name, p_info in group_params.items():
                if p_name in params:
                    p_label = p_info.get('label', p_name)
                    val = params[p_name]
                    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                    
                    # Parametre etiketi ve değeri
                    param_widget = QWidget()
                    p_layout = QHBoxLayout(param_widget)
                    p_layout.setContentsMargins(0, 0, 0, 0)
                    
                    lbl = QLabel(f"{p_label}:")
                    lbl.setStyleSheet("color: #ddd;")
                    val_lbl = QLabel(val_str)
                    val_lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
                    
                    p_layout.addWidget(lbl)
                    p_layout.addWidget(val_lbl)
                    p_layout.addStretch()
                    
                    grid_layout.addWidget(param_widget, row, col)
                    processed_params.add(p_name)
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
            
            if col != 0:
                col = 0
                row += 1

        # Kalan parametreleri (eğer varsa) 'Diğer' grubuna ekle
        all_params = set(params.keys())
        remaining = all_params - processed_params - {'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 'group', 'win_count', 'win_rate', 'params', 'test_net', 'test_pf', 'test_trades', 'test_dd', 'test_sharpe'}
        
        if remaining:
            if col != 0: row += 1; col = 0
            header = QLabel("--- Diğer Parametreler ---")
            header.setStyleSheet("color: #aaa; font-style: italic; margin-top: 5px;")
            grid_layout.addWidget(header, row, 0, 1, max_cols)
            row += 1
            col = 0
            
            for p_name in sorted(list(remaining)):
                val = params[p_name]
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                
                param_widget = QWidget()
                p_layout = QHBoxLayout(param_widget)
                p_layout.setContentsMargins(0, 0, 0, 0)
                
                lbl = QLabel(f"{p_name}:")
                val_lbl = QLabel(val_str)
                val_lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
                
                p_layout.addWidget(lbl)
                p_layout.addWidget(val_lbl)
                p_layout.addStretch()
                
                grid_layout.addWidget(param_widget, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
    
    def _get_current_period_dk(self) -> int:
        """Secili periyodu dakika olarak dondur."""
        text = self.period_combo.currentText()
        return int(text.split()[0])  # "5 dk" -> 5
    
    def _on_period_changed(self, index: int):
        """Periyot degistiginde parametre araliklerini yeniden olcekle."""
        self._on_strategy_changed(self.strategy_combo.currentIndex())
    
    def _on_strategy_changed(self, index: int):
        """Strateji değiştiğinde timer'ı yeniden başlat (Debounce)"""
        # Timer varsa (henuz setup tamamlanmadiysa)
        if hasattr(self, 'setup_timer'):
            self.setup_timer.start()

    def _setup_strategy_params(self):
        """Timer dolunca parametre gruplarını güvenli şekilde oluştur"""
        index = self.strategy_combo.currentIndex()
        
        # Mevcut grupları temizle
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.group_widgets.clear()
        
        # Yeni gruplari ekle (timeframe'e gore olceklenmis)
        if index == 0:
            base_groups = STRATEGY1_PARAM_GROUPS
        elif index == 2:
            base_groups = STRATEGY3_PARAM_GROUPS
        elif index == 3:
            base_groups = STRATEGY4_PARAM_GROUPS
            print(f"[DEBUG S4] Loaded S4 Groups: {list(base_groups.keys())}")
            for k, v in base_groups.items():
                print(f"  - {k}: {len(v.get('params', {}))} params")
        else:
            base_groups = STRATEGY2_PARAM_GROUPS
        period_dk = self._get_current_period_dk()
        param_groups = scale_param_groups(base_groups, period_dk)
        
        for group_name, group_config in param_groups.items():
            group_widget = ParameterGroupWidget(group_name, group_config)
            self.groups_layout.addWidget(group_widget)
            self.group_widgets[group_name] = group_widget
        
        self.groups_layout.addStretch()
        
        # Preset otomatik yükleme dene
        from PySide6.QtCore import QTimer as _QT2
        _QT2.singleShot(300, self._try_auto_load_preset)
    
    def _refresh_processes(self):
        """Süreç listesini yenile"""
        self.process_combo.clear()
        processes = db.get_all_processes()
        
        if not processes:
            self.process_combo.addItem("(Süreç yok - Veri yükleyin)")
            return
        
        for proc in processes:
            display = f"{proc['process_id']} ({proc['data_rows']} bar)"
            self.process_combo.addItem(display, proc['process_id'])
        
        # İlkini seç
        if processes:
            self.current_process_id = processes[0]['process_id']
    
    def _on_process_changed(self, text: str):
        """Süreç seçimi değiştiğinde veritabanından eski sonuçları yükle"""
        idx = self.process_combo.currentIndex()
        if idx < 0: return
        
        self.current_process_id = self.process_combo.itemData(idx)
        if not self.current_process_id: return
        
        # UI temizle
        for method in self.method_tables.keys():
            self.method_tables[method].setRowCount(0)
            self.method_results[method] = []
        
        # Veritabanından yükle
        results = db.get_optimization_results(self.current_process_id)
        if not results: return
        
        # Metoda göre grupla ve göster
        method_map = {
            'hibrit': 'Hibrit Grup',
            'genetik': 'Genetik',
            'bayesian': 'Bayesian'
        }
        
        grouped_results = {}
        for res in results:
            method_key = res.get('method', '').lower()
            ui_method = method_map.get(method_key, method_key.capitalize())
            
            if ui_method not in grouped_results:
                grouped_results[ui_method] = []
            
            # DB formatından UI formatına çevir
            formatted_res = {
                'net_profit': res.get('net_profit', 0),
                'trades': res.get('total_trades', 0),
                'pf': res.get('profit_factor', 0),
                'max_dd': res.get('max_drawdown', 0),
                'win_rate': res.get('win_rate', 0),
                'sharpe': res.get('sharpe', 0),
                'fitness': res.get('fitness', 0),
                'test_net': res.get('test_net', 0),
                'test_pf': res.get('test_pf', 0),
                'test_sharpe': res.get('test_sharpe', 0),
                'params': res.get('params', {})
            }
            grouped_results[ui_method].append(formatted_res)
            
        for ui_method, res_list in grouped_results.items():
            self._display_results(res_list, ui_method)
            
        # Preset otomatik yükleme dene
        self._try_auto_load_preset()

    
    def set_process(self, process_id: str):
        """DataPanel'den gelen süreç ID'sini ayarla"""
        print(f"[DEBUG] OptimizerPanel.set_process called with {process_id}")
        self.current_process_id = process_id
        self._refresh_processes()
        
        # Combo'da ilgili süreci seç
        for i in range(self.process_combo.count()):
            if self.process_combo.itemData(i) == process_id:
                self.process_combo.setCurrentIndex(i)
                break
    
    def _on_method_changed(self, index: int):
        """Yöntem değiştiğinde UI'ı güncelle"""
        # Grid Search: Adım göster
        # Bayesian/Genetik: Adım gizle (sadece min/max)
        pass  # TODO: Yönteme göre UI güncelle
    
    def _calculate_combinations(self):
        """Toplam kombinasyon sayısını hesapla"""
        total = 1
        for group_name, group_widget in self.group_widgets.items():
            ranges = group_widget.get_ranges()
            group_combo = 1
            for param_name, values in ranges.items():
                group_combo *= len(values)
            total *= group_combo if group_combo > 0 else 1
        
        # Süre tahmini (250 test/sn varsayımıyla)
        estimated_seconds = total / 250
        if estimated_seconds < 60:
            time_str = f"{estimated_seconds:.0f} saniye"
        elif estimated_seconds < 3600:
            time_str = f"{estimated_seconds/60:.1f} dakika"
        else:
            time_str = f"{estimated_seconds/3600:.1f} saat"
        
        self.combo_label.setText(f"Toplam kombinasyon: {total:,} (~{time_str})")
    
    def _get_all_ranges(self) -> dict:
        """Tüm parametre aralıklarını al"""
        all_ranges = {}
        for group_name, group_widget in self.group_widgets.items():
            ranges = group_widget.get_ranges()
            all_ranges.update(ranges)
        return all_ranges
    
    def _derive_narrowed_ranges(self, results: list) -> dict:
        """
        Hibrit sonuclarindan dar parametre araliklari turet.
        Her parametre icin: min-max degerlerini bul, %20 buffer ekle.
        
        Returns:
            dict: {param_name: (min_val, max_val)} formati
        """
        if not results:
            return {}
        
        # Metrik anahtarlari (bunlar parametre degil)
        excluded_keys = {
            'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 
            'group', 'win_count', 'win_rate', 'params'
        }
        
        # Ilk sonuctan parametre isimlerini cikar
        first_result = results[0]
        param_keys = [k for k in first_result.keys() if k not in excluded_keys]
        
        if not param_keys:
            print("[CASCADE] Uyari: Parametre anahtari bulunamadi!")
            return {}
        
        narrowed = {}
        for key in param_keys:
            values = []
            for r in results:
                val = r.get(key)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            
            if values:
                min_val = min(values)
                max_val = max(values)
                
                # %20 buffer ekle (aralik sifirsa kucuk bir buffer)
                range_size = max_val - min_val
                if range_size > 0:
                    buffer = range_size * 0.2
                else:
                    # Tek deger, kucuk buffer
                    buffer = abs(min_val) * 0.1 if min_val != 0 else 0.01
                
                narrowed[key] = (min_val - buffer, max_val + buffer)
        
        print(f"[CASCADE] Dar araliklar turetildi: {len(narrowed)} parametre")
        return narrowed

    
    def _start_optimization(self):
        """Optimizasyonu başlat"""
        if self.data is None:
            QMessageBox.warning(self, "Uyari", "Lutfen once veri yukleyin.")
            return
        
        param_ranges = self._get_all_ranges()
        if not param_ranges:
            QMessageBox.warning(self, "Uyari", "En az bir parametre secin.")
            return
        
        # UI güncelle
        self.start_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Baslatiliyor...")
        
        # Timer başlat (1 saniye aralıkla)
        self.start_time = time.time()
        self.timer.start(1000)
        
        # Worker başlat
        method = self.method_combo.currentText()
        strategy_index = self.strategy_combo.currentIndex()
        n_parallel = int(self.workers_spin.currentText())
        
        # Maliyetleri DB'ye kaydet (Validation için)
        if self.current_process_id:
            db.update_process_costs(
                self.current_process_id, 
                self.commission_spin.value(), 
                self.slippage_spin.value()
            )
        
        # Cascade: Hibrit değilse ve hibrit sonuçları varsa dar aralık türet
        narrowed_ranges = None
        if method != "Hibrit Grup" and self.hybrid_results:
            narrowed_ranges = self._derive_narrowed_ranges(self.hybrid_results)
            if narrowed_ranges:
                print(f"[CASCADE] {method} dar aralikta calisacak")
        
        # Auto-Validation ayarlari
        do_oos = self.validation_check.isChecked()
        train_pct = self.split_spin.value()
        
        self.worker = OptimizationWorker(
            self.config, self.data, param_ranges, method, strategy_index,
            process_id=self.current_process_id,
            n_parallel=n_parallel,
            narrowed_ranges=narrowed_ranges,
            do_oos=do_oos,
            train_pct=train_pct
        )
        self.worker.commission = self.commission_spin.value()
        self.worker.slippage = self.slippage_spin.value()
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.partial_results.connect(self._on_partial_results)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        
        # Canlı İzleme Ekranını aç
        self._open_live_monitor(strategy_index, method)
    
    # ==============================================================================
    # CANLI İZLEME EKRANI
    # ==============================================================================
    def _open_live_monitor(self, strategy_index: int, method: str):
        """Canlı İzleme Ekranını aç veya güncelle"""
        try:
            from src.ui.widgets.live_monitor_window import LiveMonitorWindow
            if self.live_monitor_win is None or not self.live_monitor_win.isVisible():
                self.live_monitor_win = LiveMonitorWindow(
                    strategy_index=strategy_index,
                    method=method
                )
                self.live_monitor_win.showMaximized()
            else:
                # Pencere zaten açık — stratejiyi güncelle
                self.live_monitor_win.set_strategy(strategy_index, method)
        except Exception as e:
            print(f"[LIVE MONITOR] Açılamadı: {e}")
    
    def _on_partial_results(self, results: list):
        """Worker'dan gelen kısmi sonuçları canlı pencereye yönlendir"""
        if self.live_monitor_win and self.live_monitor_win.isVisible():
            self.live_monitor_win.update_results(results, is_final=False)
        
        # Live monitor label'ı da güncelle
        if results:
            best = max(results, key=lambda r: r.get('fitness', r.get('net_profit', 0)))
            net = best.get('net_profit', 0)
            pf = best.get('pf', 0)
            sh = best.get('sharpe', 0)
            fit = best.get('fitness', 0)
            self.live_monitor_label.setText(
                f"  ⭐ En İyi: Net {net:,.0f} | PF {pf:.2f} | Sharpe {sh:.2f} | Fitness {fit:,.0f}  "
            )
    
    # ==============================================================================
    # PRESET KAYIT / YUKLE
    # ==============================================================================
    def _save_preset(self):
        """Mevcut parametre aralıklarını preset olarak kaydet"""
        if not self.current_process_id:
            QMessageBox.warning(self, "Uyarı", "Önce bir süreç seçin.")
            return
        
        strategy_idx = self.strategy_combo.currentIndex()
        strategy_name = self.strategy_combo.currentText()
        period = self.period_combo.currentText()
        symbol = self.process_combo.currentText().split('(')[0].strip() if self.process_combo.currentText() else "unknown"
        
        # Parametre aralıklarını topla
        param_ranges = {}
        for group_id, group_widget in self.group_widgets.items():
            # group_widget is ParameterGroupWidget, access its param_widgets dict
            for param_name, w in group_widget.param_widgets.items():
                if isinstance(w, dict) and 'min' in w:
                    param_ranges[param_name] = {
                        'min': w['min'].value(),
                        'max': w['max'].value(),
                        'step': w['step'].value(),
                        'active': w['active'].isChecked() if 'active' in w else True
                    }
        
        if not param_ranges:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek parametre bulunamadı.")
            return
        
        path = self.preset_manager.save_preset(symbol, period, strategy_idx, strategy_name, param_ranges)
        self.status_label.setText(f"💾 Preset kaydedildi: {os.path.basename(path)}")
    
    def _load_preset(self):
        """Kaydedilmiş preset'i yükle (Liste üzerinden seçim)"""
        strategy_idx = self.strategy_combo.currentIndex()
        period = self.period_combo.currentText()
        symbol = self.process_combo.currentText().split('(')[0].strip() if self.process_combo.currentText() else "unknown"
        
        # Tüm presetleri listele
        all_presets = self.preset_manager.list_presets()
        
        # Uygun olanları filtrele (Strateji ve Periyot eşleşmeli)
        compatible = []
        for p in all_presets:
            # Json verisinden gelen period ve strategy_name kontrolü
            # Ancak strategy_index json'da var, listede var mi bakalim
            # list_presets() params count donduruyor, strategy index'i dosyadan okuyor
            # Bizim icin onemli olan strategy index eslesmesi
            
            # PresetManager.list_presets implementation reads JSON content
            # Let's verify matches. 
            # Note: list_presets helper dict keys: 'file', 'symbol', 'period', 'strategy' (name), 'created'
            # We prefer matching by Index if available inside file, assuming list_presets could be improved or we trust name
            
            # Match strict on Period
            if p.get('period') != period:
                continue
            
            # Match on Strategy Name or attempt to verify Index if possible
            # Current Strategy Name from UI
            current_strat_name = self.strategy_combo.currentText()
            saved_strat_name = p.get('strategy', '')
            
            # Simple check: Strategy name match OR filename suffix match (S1, S2...)
            # Dosya adından S_index çıkarma: ..._S{idx}.json
            try:
                fname = p['file']
                if fname.endswith(f"_S{strategy_idx+1}.json"):
                    compatible.append(p)
                    continue
            except:
                pass
                
            if saved_strat_name == current_strat_name:
                compatible.append(p)

        if not compatible:
            QMessageBox.information(self, "Bilgi", f"Bu strateji ({period}) için kayıtlı hiçbir preset bulunamadı.")
            return

        from PySide6.QtWidgets import QInputDialog
        
        # Listeyi hazırla: "Symbol | Tarih | Dosya"
        items = []
        current_selection = 0
        
        # Sort by Date desc
        compatible.sort(key=lambda x: x.get('created', ''), reverse=True)
        
        for i, p in enumerate(compatible):
            display = f"{p['symbol']}  —  {p['created']}  ({p['file']})"
            items.append(display)
            # Eğer mevcut sembol ile eşleşiyorsa varsayılan seç
            if p['symbol'] == symbol:
                current_selection = i
        
        item, ok = QInputDialog.getItem(self, "Preset Yükle", 
                                      f"Mevcut Strateji ({period}) için kayıtlar:", 
                                      items, current_selection, False)
        
        if ok and item:
            # Seçilen dosyayı bul
            selected_idx = items.index(item)
            selected_preset_info = compatible[selected_idx]
            
            # Yükle (full path)
            import json
            full_path = os.path.join(self.preset_manager.preset_dir, selected_preset_info['file'])
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                    self._apply_preset(preset_data)
                    self.status_label.setText(f"📂 Preset yüklendi: {selected_preset_info['file']}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Preset yüklenirken hata: {e}")

    def _apply_preset(self, preset: dict):
        """Preset değerlerini spin'lere uygula"""
        params = preset.get('params', {})
        for group_id, group_widget in self.group_widgets.items():
             # group_widget is ParameterGroupWidget, access its param_widgets dict
            for param_name, w in group_widget.param_widgets.items():
                if isinstance(w, dict) and param_name in params:
                    p = params[param_name]
                    try:
                        # Check if C++ object is valid
                        if 'min' in w and 'min' in p:
                            w['min'].setValue(p['min'])
                        if 'max' in w and 'max' in p:
                            w['max'].setValue(p['max'])
                        if 'step' in w and 'step' in p:
                            w['step'].setValue(p['step'])
                        if 'active' in w and 'active' in p:
                            w['active'].setChecked(p['active'])
                    except RuntimeError:
                        # "Internal C++ object ... already deleted"
                        print(f"[PRESET] Widget deleted for {param_name}, skipping.")
                        continue
    
    def _try_auto_load_preset(self):
        """Strateji veya süreç değiştiğinde otomatik preset yükleme dene"""
        print("[DEBUG] OptimizerPanel._try_auto_load_preset called")
        try:
            if not self.current_process_id:
                return
            
            # [FIX] Eger widget'lar henuz olusturulmadiysa veya temizlendiyse iptal et
            if not self.group_widgets:
                print("[PRESET] Group widgets not ready, skipping auto-load.")
                return

            strategy_idx = self.strategy_combo.currentIndex()
            period = self.period_combo.currentText()
            symbol = self.process_combo.currentText().split('(')[0].strip() if self.process_combo.currentText() else ""
            if not symbol:
                return
            preset = self.preset_manager.load_preset(symbol, period, strategy_idx)
            if preset:
                self._apply_preset(preset)
                self.status_label.setText(f"📂 Preset otomatik yüklendi ({preset.get('created', '')})")
        except Exception as e:
            print(f"[PRESET] Auto-load failed: {e}")
            import traceback
            traceback.print_exc()

    # ==============================================================================
    # CHECKPOINT (Resume) LOGIC
    # ==============================================================================
    def _get_checkpoint_path(self):
        """Checkpoint dosya yolunu dondur"""
        import os
        # Proje koku: src/ui/widgets/../../.. = proje root
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(base, self.CHECKPOINT_FILE)
    
    def _save_checkpoint(self):
        """Mevcut kuyruk durumunu JSON dosyasina kaydet"""
        import json
        try:
            checkpoint = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'strategy_index': self.strategy_combo.currentIndex(),
                'method_combo_text': self.method_combo.currentText(),
                'remaining_queue': list(self.optimization_queue),
                'queue_total': self._queue_total,
                'process_id': self.current_process_id,
                'validation_enabled': self.validation_check.isChecked(),
                'split_pct': self.split_spin.value(),
                'commission': self.commission_spin.value(),
                'slippage': self.slippage_spin.value(),
            }
            with open(self._get_checkpoint_path(), 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CHECKPOINT] Kayit hatasi: {e}")
    
    def _load_checkpoint(self):
        """Checkpoint dosyasini oku"""
        import json, os
        path = self._get_checkpoint_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CHECKPOINT] Okuma hatasi: {e}")
            return None
    
    def _clear_checkpoint(self):
        """Checkpoint dosyasini sil"""
        import os
        path = self._get_checkpoint_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except: pass
        self.resume_btn.setVisible(False)
    
    def _check_checkpoint(self):
        """Baslangicta checkpoint kontrolu yap, varsa 'Devam Et' butonunu goster"""
        cp = self._load_checkpoint()
        if cp and cp.get('remaining_queue'):
            remaining = cp['remaining_queue']
            ts = cp.get('timestamp', '?')
            self.resume_btn.setVisible(True)
            self.resume_btn.setToolTip(
                f"Son kesinti: {ts}\nKalan: {', '.join(remaining)}"
            )
            self.status_label.setText(
                f"⚡ Checkpoint bulundu ({ts}) — {', '.join(remaining)} kaldı. 'Devam Et' ile sürdürün."
            )
        else:
            self.resume_btn.setVisible(False)
    
    def _resume_from_checkpoint(self):
        """Checkpoint'ten devam et"""
        cp = self._load_checkpoint()
        if not cp or not cp.get('remaining_queue'):
            QMessageBox.information(self, "Bilgi", "Devam edilecek checkpoint bulunamadi.")
            self.resume_btn.setVisible(False)
            return
        
        if self.data is None:
            QMessageBox.warning(self, "Uyari", "Lutfen once veri yukleyin, sonra 'Devam Et' basiniz.")
            return
        
        # Restore state
        remaining = cp['remaining_queue']
        strategy_idx = cp.get('strategy_index', 0)
        
        # Strateji combo'sunu ayarla
        if strategy_idx < self.strategy_combo.count():
            self.strategy_combo.setCurrentIndex(strategy_idx)
        
        # Validation ayarlari
        self.validation_check.setChecked(cp.get('validation_enabled', False))
        self.split_spin.setValue(cp.get('split_pct', 80))
        self.commission_spin.setValue(cp.get('commission', 0))
        self.slippage_spin.setValue(cp.get('slippage', 0))
        
        # Process ID
        if cp.get('process_id'):
            self.current_process_id = cp['process_id']
        
        # Kuyruğu yükle
        self.optimization_queue = list(remaining)
        self._stop_requested = False
        self._queue_total = cp.get('queue_total', 3)
        
        # UI
        self.global_progress_bar.setValue(0)
        self.global_progress_bar.setVisible(True)
        self.live_monitor_frame.setVisible(True)
        self.live_monitor_label.setText("Checkpoint'ten devam ediliyor...")
        self.resume_btn.setVisible(False)
        
        self.total_start_time = time.time()
        
        # Sıradakini başlat
        self._start_next_in_queue()

    def _run_all_optimizers(self):
        """Tüm optimizasyon yöntemlerini sırayla çalıştır"""
        if self.data is None:
            QMessageBox.warning(self, "Uyari", "Lutfen once veri yukleyin.")
            return
            
        # Kuyruğu doldur
        self.optimization_queue = ["Hibrit Grup", "Genetik", "Bayesian"]
        self._stop_requested = False
        self._queue_total = len(self.optimization_queue)
        
        # Checkpoint kaydet
        self._save_checkpoint()
        
        # Global progress bar gorunur yap
        self.global_progress_bar.setValue(0)
        self.global_progress_bar.setVisible(True)
        self.live_monitor_frame.setVisible(True)
        self.live_monitor_label.setText("En Iyi Sonuc bekleniyor...")
        self.resume_btn.setVisible(False)
        
        # Genel baslangic zamani
        self.total_start_time = time.time()
        
        # Ilkini baslat
        self._start_next_in_queue()
        
    def _start_next_in_queue(self):
        """Kuyruktaki bir sonraki yontemi baslat"""
        if not self.optimization_queue:
            return
        
        # Update global progress
        completed = self._queue_total - len(self.optimization_queue)
        global_pct = int((completed / self._queue_total) * 100) if self._queue_total > 0 else 0
        self.global_progress_bar.setValue(global_pct)
            
        method = self.optimization_queue.pop(0)
        
        # Combo'yu guncelle
        index = self.method_combo.findText(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
            
        # Baslat
        self._start_optimization()
    
    def _stop_optimization(self):
        """Durdur: worker'i durdur + kuyruğu tamamen temizle"""
        self._stop_requested = True
        self.optimization_queue.clear()  # Kuyrugu temizle!
        self._clear_checkpoint()  # Checkpoint temizle
        if self.worker:
            self.worker.stop()
            self.timer.stop()
            self.status_label.setText(f"Durduruldu (Sure: {self._get_elapsed_str()})")
        # UI Reset
        self.global_progress_bar.setVisible(False)
        self.live_monitor_frame.setVisible(False)
        self.start_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.resume_btn.setVisible(False)
    
    def _on_progress(self, percent: int, message: str, elapsed: str, eta: str):
        self.progress_bar.setValue(percent)
        # Timer independently updates the label, but we can update the message part here
        self.current_message = message
        self.current_eta = eta
        self._update_status_label()
        
        # Canli monitor scan label guncelle
        if hasattr(self, 'live_scan_label') and self.live_monitor_frame.isVisible():
            self.live_scan_label.setText(message)
            self.live_timer_label.setText(f"Gecen: {elapsed} | Kalan: {eta}")
    
    def _update_status_label(self):
        elapsed = self._get_elapsed_str()
        msg = getattr(self, 'current_message', 'Çalışıyor...')
        eta = getattr(self, 'current_eta', '--:--')
        
        # Toplam süre (Queue modunda ise)
        total_info = ""
        if hasattr(self, 'total_start_time') and getattr(self, 'optimization_queue', None) is not None:
            total_elapsed = time.time() - self.total_start_time
            total_info = f"\n[GENEL TOPLAM] Süre: {self._format_time(total_elapsed)}"
            
        self.status_label.setText(f"{msg} (Geçen: {elapsed} - Kalan: {eta}){total_info}")

    def _get_elapsed_str(self) -> str:
        if not hasattr(self, 'start_time'):
             return "00:00:00"
        elapsed = time.time() - self.start_time
        return self._format_time(elapsed)

    def _update_timer(self):
        self._update_status_label()

    def _format_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _on_result(self, results: list):
        # Aktif metodu belirle
        method = self.method_combo.currentText()
        
        # Hibrit sonuclarini sakla (Cascade icin)
        if method == "Hibrit Grup" and results:
            self.hybrid_results = results
            print(f"[CASCADE] Hibrit sonuclari kaydedildi: {len(results)} adet")
        
        # Live Monitor Update
        if results and self.live_monitor_frame.isVisible():
            best = results[0] if results else {}
            np_val = best.get('net_profit', 0)
            pf_val = best.get('pf', 0)
            sh_val = best.get('sharpe', 0)
            fit_val = best.get('fitness', 0)
            t_val = best.get('trades', 0)
            self.live_monitor_label.setText(
                f"🏆 {method}: Net={np_val:,.0f}  PF={pf_val:.2f}  Sharpe={sh_val:.2f}  Fit={fit_val:,.0f}  ({t_val} islem)"
            )
            # Flash animation
            self.live_monitor_frame.setStyleSheet(
                "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1b5e20, stop:1 #2e7d32); "
                "border-radius: 6px; padding: 8px; margin: 4px 0; }"
            )
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: self.live_monitor_frame.setStyleSheet(
                "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1a237e, stop:1 #283593); "
                "border-radius: 6px; padding: 8px; margin: 4px 0; }"
            ))
        
        # Canlı İzleme Ekranına final sonuçları gönder
        if self.live_monitor_win and self.live_monitor_win.isVisible():
            self.live_monitor_win.update_results(results, method=method, is_final=True)
        
        self._display_results(results, method)
        
        # Veritabanına kaydet
        if self.current_process_id and results:
            method_key = method.lower().replace(" ", "_").replace("hibrit_grup", "hibrit")
            strategy_index = self.strategy_combo.currentIndex()
            
            # En iyi sonucu kaydet
            best = results[0] if results else {}
            params = best.get('params', best)
            
            db.save_optimization_result(
                process_id=self.current_process_id,
                strategy_index=strategy_index,
                method=method_key,
                params=params,
                result={
                    'net_profit': best.get('net_profit', 0),
                    'max_drawdown': best.get('max_dd', best.get('max_drawdown', 0)),
                    'profit_factor': best.get('pf', best.get('profit_factor', 0)),
                    'total_trades': best.get('trades', best.get('total_trades', 0)),
                    'win_rate': best.get('win_rate', 0),
                    'sharpe': best.get('sharpe', 0),
                    'fitness': best.get('fitness', 0),
                    'test_net': best.get('test_net', 0),
                    'test_pf': best.get('test_pf', 0),
                    'test_sharpe': best.get('test_sharpe', 0)
                }
            )
        
        self.optimization_complete.emit(results)
    
    def _on_error(self, error_msg: str):
        self.timer.stop()
        QMessageBox.critical(self, "Hata", f"Optimizasyon hatasi: {error_msg}")
        self.status_label.setText("Hata!")
    
    def _on_finished(self):
        self.worker = None
        self.timer.stop()
        self.progress_bar.setValue(100)
        
        final_time = self._get_elapsed_str()
        
        # Stop istendi mi?
        if self._stop_requested:
            # Kuyruk zaten temizlendi, UI zaten resetlendi
            self._stop_requested = False
            return
        
        # Kuyrukta islem varsa devam et
        if self.optimization_queue:
            # Update global progress
            completed = self._queue_total - len(self.optimization_queue)
            global_pct = int((completed / self._queue_total) * 100) if self._queue_total > 0 else 0
            self.global_progress_bar.setValue(global_pct)
            
            # Checkpoint guncelle (kalan kuyruk)
            self._save_checkpoint()
            
            self.status_label.setText(f"Tamamlandi ({final_time}). Siradaki baslatiliyor...")
            # Kisa bir gecikme ile baslat
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, self._start_next_in_queue)
        else:
            total_time_str = ""
            if hasattr(self, 'total_start_time') and self.total_start_time:
                total_elapsed = time.time() - self.total_start_time
                total_time_str = f" | Toplam Sure: {self._format_time(total_elapsed)}"
                # Temizle
                self.total_start_time = None
            
            # Basarili tamamlanma: checkpoint temizle
            self._clear_checkpoint()
                
            self.start_btn.setEnabled(True)
            self.run_all_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.resume_btn.setVisible(False)
            self.global_progress_bar.setValue(100)
            self.global_progress_bar.setVisible(False)
            self.live_monitor_frame.setVisible(False)
            self.status_label.setText(f"Optimizasyon Tamamlandi (Adim Suresi: {final_time}{total_time_str})")
    
    def _display_results(self, results: list, method: str = None):
        """Sonuçları ilgili tab'a yaz"""
        # Hangi metod için?
        if method is None:
            method = self.method_combo.currentText()
        
        table = self.method_tables.get(method)
        if not table:
            # Fallback: Mevcut tab'ı kullan
            method = list(self.method_tables.keys())[self.results_tab_widget.currentIndex()]
            table = self.method_tables.get(method)
        
        if not results:
            self.results_stats_label.setText(f"[{method}] Sonuc bulunamadi")
            return
        
        # Sonuçları sakla (parametre gösterimi için)
        self.method_results[method] = results
        
        # Tablo doldur
        cols = ['Sira', 'Kar (Egt)', 'Test Kar', 'Test PF', 'Islem', 'PF (Egt)', 'Max DD', 'Sharpe', 'Fitness']
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(results))
        
        for row_idx, result in enumerate(results):
            # Sira
            table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            
            # Egitim Kari
            net_profit = result.get('net_profit', 0)
            net_item = QTableWidgetItem(f"{net_profit:,.0f}")
            if net_profit > 0: net_item.setForeground(Qt.darkGreen)
            table.setItem(row_idx, 1, net_item)
            
            # Test Kari (OOS)
            test_net = result.get('test_net', 0)
            test_item = QTableWidgetItem(f"{test_net:,.0f}")
            if test_net > 0: test_item.setForeground(Qt.darkGreen)
            elif test_net < 0: test_item.setForeground(Qt.red)
            # Eger test sonucu yoksa (eski kayitlar vs)
            if 'test_net' not in result:
                test_item.setText("-")
                test_item.setForeground(Qt.gray)
            table.setItem(row_idx, 2, test_item)
            
            # Test PF
            test_pf = result.get('test_pf', 0)
            test_pf_item = QTableWidgetItem(f"{test_pf:.2f}")
            if 'test_pf' not in result: test_pf_item.setText("-")
            table.setItem(row_idx, 3, test_pf_item)
            
            # Islem sayisi
            table.setItem(row_idx, 4, QTableWidgetItem(str(result.get('trades', 0))))
            
            # Egitim PF
            table.setItem(row_idx, 5, QTableWidgetItem(f"{result.get('pf', 0):.2f}"))
            
            # Max DD
            table.setItem(row_idx, 6, QTableWidgetItem(f"{result.get('max_dd', 0):,.0f}"))
            
            # Sharpe
            table.setItem(row_idx, 7, QTableWidgetItem(f"{result.get('sharpe', 0):.2f}"))
            
            # Fitness skoru (Renkli)
            fitness = result.get('fitness', 0)
            fit_item = QTableWidgetItem(f"{fitness:,.0f}")
            if fitness > 0:
                fit_item.setForeground(Qt.darkGreen)
            else:
                fit_item.setForeground(Qt.red)
            table.setItem(row_idx, 8, fit_item)
        
        # Tab'ı aktif yap
        method_names = ["Hibrit Grup", "Genetik", "Bayesian"]
        if method in method_names:
            self.results_tab_widget.setCurrentIndex(method_names.index(method))
        
        self.results_stats_label.setText(f"[{method}] {len(results)} sonuc bulundu")
    
    def _export_results(self):
        QMessageBox.information(self, "Bilgi", "Sonuc disa aktarma yakinda eklenecek.")
    
    def set_strategy_config(self, config: dict):
        self.config = config
        # Strateji seçimini güncelle
        strategy_idx = config.get('strategy', 1) - 1
        self.strategy_combo.setCurrentIndex(strategy_idx)
    
    def set_data(self, df):
        with open("d:/Projects/IdealQuant/debug_log.txt", "a") as f:
            f.write(f"\n[DEBUG] OptimizerPanel set_data\n")
            f.write(f"  Columns: {df.columns.tolist()}\n")
            f.write(f"  Shape: {df.shape}\n")
        self.data = df
