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
    QScrollArea, QCheckBox, QFrame
)
from PySide6.QtCore import Signal, Qt, QThread
from typing import Dict, List, Any
import pandas as pd
import time

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
                min_spin.setRange(1, 1000)
                min_spin.setValue(param_config['min'])
            else:
                min_spin = QDoubleSpinBox()
                min_spin.setRange(0.001, 100.0)
                min_spin.setDecimals(3)
                min_spin.setValue(param_config['min'])
            table.setCellWidget(row, 1, min_spin)
            
            # Max
            if param_config['type'] == 'int':
                max_spin = QSpinBox()
                max_spin.setRange(1, 1000)
                max_spin.setValue(param_config['max'])
            else:
                max_spin = QDoubleSpinBox()
                max_spin.setRange(0.001, 100.0)
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
                step_spin.setRange(0.001, 10.0)
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
        self.test_data = None  # Init
        self._is_running = True
        self.start_time = None
    
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
            if self.method == "Hibrit Grup":
                self._run_hybrid()
            elif self.method == "Genetik":
                self._run_genetic()
            elif self.method == "Bayesian":
                self._run_bayesian()
            else:
                self.error.emit(f"Bilinmeyen yöntem: {self.method}")
                
            # OOS Validasyon (Eger validation aktifse ve sonuc varsa)
            # Not: _run_... metodlari self.result_ready.emit yapiyor.
            # Bunu yakalamak yerine _run metodlarindan sonucu donmesini beklemeliyiz veya sinyal mekanizmasini degistirmeliyiz.
            # Şu anki yapida _run metodlari direkt emit ediyor. OOS testini _run metodlarinin icine gommek veya emit edilen veriyi manupile etmek lazim.
            # En temizi: _run metodlari artik "sonuc donmeli", emit islemi "run" metodunun sonunda yapilmali.
            # Ancak kodu cok degistirmemek icin, Optimizer'larin result_ready signalini gecici olarak override edebiliriz? Hayir, karmasik olur.
            
            # En Pratik Çözüm: _run metodlari icinde validation yapalim.
            # self.do_oos ve self.test_data (yukarida olusturdugumuz) erisilebilir olacak.
            self.test_data = test_data # Instance variable yap
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
            
        # Restore data
        self.data = original_data
    
    def _run_hybrid(self):
        """Hibrit Grup optimizasyonu"""
        from src.optimization.hybrid_group_optimizer import (
            HybridGroupOptimizer, IndicatorCache, STRATEGY1_GROUPS, STRATEGY2_GROUPS, ParameterGroup
        )
        
        self._emit_progress(5, "Cache olusturuluyor...")
        cache = IndicatorCache(self.data)
        
        # Strateji seçiminden orijinal grupları al
        original_groups = STRATEGY1_GROUPS if self.strategy_index == 0 else STRATEGY2_GROUPS
        strategy_name = "Strateji 1" if self.strategy_index == 0 else "Strateji 2"
        
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
        optimizer = HybridGroupOptimizer(
            synced_groups, 
            process_id=self.process_id, 
            strategy_index=self.strategy_index,
            is_cancelled_callback=lambda: not self._is_running,
            on_progress_callback=self._emit_progress,
            n_parallel=self.n_parallel,
            commission=self.commission,
            slippage=self.slippage
        )
        
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
        
        strategy_name = "Strateji 1" if self.strategy_index == 0 else "Strateji 2"
        
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
        
        # Nesil bazlı ilerleme için callback
        def on_gen_complete(gen, max_gen, best_fit):
            progress = 10 + int((gen / max_gen) * 85)
            self._emit_progress(progress, f"Nesil {gen}/{max_gen} - En İyi: {best_fit:,.0f}")
        
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
        
        strategy_name = "Strateji 1" if self.strategy_index == 0 else "Strateji 2"
        
        # Cascade modu kontrolu
        if self.narrowed_ranges:
            self._emit_progress(5, f"Bayesian (CASCADE) baslatiliyor ({strategy_name})...")
            print(f"[CASCADE] Bayesian dar aralikta calisacak: {len(self.narrowed_ranges)} parametre")
        else:
            self._emit_progress(5, f"Bayesian Optimizer baslatiliyor ({strategy_name})...")
        
        n_trials = 100  # Deneme sayısı
        
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
        
        # Trial bazlı ilerleme için callback
        def on_trial_complete(trial_no, max_trials, best_fit):
            progress = 10 + int((trial_no / max_trials) * 85)
            self._emit_progress(progress, f"Deneme {trial_no}/{max_trials} - En İyi: {best_fit:,.0f}")
            
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
        self._is_running = False

    def _validate_result(self, params):
        """Test verisi uzerinde validasyon yap"""
        if self.test_data is None: return {}
        
        try:
            from src.optimization.hybrid_group_optimizer import IndicatorCache
            test_cache = IndicatorCache(self.test_data)
            
            if self.strategy_index == 0:
                from src.strategies.score_based import ScoreBasedStrategy
                # ScoreBasedStrategy config dict kabul eder
                strategy = ScoreBasedStrategy.from_config_dict(test_cache, params)
            else:
                from src.strategies.ars_trend_v2 import ARSTrendStrategyV2
                strategy = ARSTrendStrategyV2.from_config_dict(test_cache, params)
                
            signals, ex_long, ex_short = strategy.generate_all_signals()
            
            # Backtest
            net, trades, pf, dd, sharpe = self._simple_backtest(
                test_cache.closes, signals, ex_long, ex_short
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

    def _simple_backtest(self, closes, signals, ex_long, ex_short):
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
            sharpe = calculate_sharpe(np.array(trade_returns))
            
        return net, trades, pf, max_dd, sharpe


class OptimizerPanel(QWidget):
    """Optimizasyon paneli - Parametre aralıkları ile"""
    
    optimization_complete = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.config = {}
        self.data = None
        self.worker = None
        self.group_widgets = {}
        self.current_process_id = None
        self.optimization_queue = []
        self.hybrid_results = []  # Hibrit sonuclarini sakla (Cascade icin)
        
        # Timer için
        from PySide6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_timer)
        
        self._setup_ui()
    
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
        
        # Strateji seçimi
        top_row.addWidget(QLabel("Strateji:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Strateji 1 - Gatekeeper", "Strateji 2 - ARS Trend v2"])
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
        self._on_strategy_changed(0)
    
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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
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
        
        method_names = ["Hibrit Grup", "Genetik", "Bayesian"]
        
        for method in method_names:
            # Her tab için bir Splitter (üst: tablo, alt: parametreler)
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(2, 2, 2, 2)
            
            splitter = QSplitter(Qt.Vertical)
            
            # Üst kısım: Sonuç tablosu
            table = QTableWidget()
            table.setAlternatingRowColors(True)
            table.setSortingEnabled(True)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)
            table.itemSelectionChanged.connect(lambda m=method: self._on_result_selected(m))
            splitter.addWidget(table)
            self.method_tables[method] = table
            
            # Alt kısım: Parametreler
            params_group = QGroupBox("Secili Sonucun Parametreleri")
            params_layout = QVBoxLayout(params_group)
            params_layout.setContentsMargins(5, 5, 5, 5)
            
            params_text = QTextEdit()
            params_text.setReadOnly(True)
            params_text.setMaximumHeight(120)
            params_text.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
            params_text.setPlaceholderText("Sonuc tablosundan bir satir secin...")
            params_layout.addWidget(params_text)
            splitter.addWidget(params_group)
            self.method_params_text[method] = params_text
            
            # Splitter oranları
            splitter.setSizes([250, 120])
            
            tab_layout.addWidget(splitter)
            self.results_tab_widget.addTab(tab_widget, method)
            self.method_results[method] = []
        
        layout.addWidget(self.results_tab_widget)
        
        return group
    
    def _on_result_selected(self, method: str):
        """Sonuç tablosunda satır seçildiğinde parametreleri göster"""
        table = self.method_tables.get(method)
        params_text = self.method_params_text.get(method)
        results = self.method_results.get(method, [])
        
        if not table or not params_text or not results:
            return
        
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row_idx = selected_rows[0].row()
        if row_idx >= len(results):
            return
        
        result = results[row_idx]
        
        # Performans metrikleri dışındaki tüm parametreleri göster
        excluded_keys = {'net_profit', 'trades', 'pf', 'max_dd', 'sharpe', 'fitness', 'group', 'win_count', 'win_rate'}
        params = {k: v for k, v in result.items() if k not in excluded_keys}
        
        # Formatla
        lines = []
        for key, value in sorted(params.items()):
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        
        params_text.setText("\n".join(lines) if lines else "Parametre bulunamadi")
    
    def _on_strategy_changed(self, index: int):
        """Strateji değiştiğinde parametre gruplarını güncelle"""
        # Mevcut grupları temizle
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.group_widgets.clear()
        
        # Yeni grupları ekle
        param_groups = STRATEGY1_PARAM_GROUPS if index == 0 else STRATEGY2_PARAM_GROUPS
        
        for group_name, group_config in param_groups.items():
            group_widget = ParameterGroupWidget(group_name, group_config)
            self.groups_layout.addWidget(group_widget)
            self.group_widgets[group_name] = group_widget
        
        self.groups_layout.addStretch()
    
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
        """Süreç seçimi değiştiğinde"""
        idx = self.process_combo.currentIndex()
        if idx >= 0:
            self.current_process_id = self.process_combo.itemData(idx)
    
    def set_process(self, process_id: str):
        """DataPanel'den gelen süreç ID'sini ayarla"""
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
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        
    def _run_all_optimizers(self):
        """Tüm optimizasyon yöntemlerini sırayla çalıştır"""
        if self.data is None:
            QMessageBox.warning(self, "Uyari", "Lutfen once veri yukleyin.")
            return
            
        # Kuyruğu doldur
        self.optimization_queue = ["Hibrit Grup", "Genetik", "Bayesian"]
        
        # İlkini başlat
        self._start_next_in_queue()
        
    def _start_next_in_queue(self):
        """Kuyruktaki bir sonraki yöntemi başlat"""
        if not self.optimization_queue:
            return
            
        method = self.optimization_queue.pop(0)
        
        # Combo'yu güncelle
        index = self.method_combo.findText(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
            
        # Başlat
        self._start_optimization()
    
    def _stop_optimization(self):
        if self.worker:
            self.worker.stop()
            self.timer.stop()
            self.status_label.setText(f"Durduruldu (Süre: {self._get_elapsed_str()})")
    
    def _on_progress(self, percent: int, message: str, elapsed: str, eta: str):
        self.progress_bar.setValue(percent)
        # Timer independently updates the label, but we can update the message part here
        self.current_message = message
        self.current_eta = eta
        self._update_status_label()
    
    def _update_status_label(self):
        elapsed = self._get_elapsed_str()
        msg = getattr(self, 'current_message', 'Çalışıyor...')
        eta = getattr(self, 'current_eta', '--:--')
        self.status_label.setText(f"{msg} (Geçen: {elapsed} - Kalan: {eta})")

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
                    'fitness': best.get('fitness', 0)
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
        
        final_time = self._get_elapsed_str()
        
        # Kuyrukta işlem varsa devam et
        if self.optimization_queue:
            self.status_label.setText(f"Tamamlandı ({final_time}). Sıradaki başlatılıyor...")
            # Kısa bir gecikme ile başlat
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, self._start_next_in_queue)
        else:
            self.start_btn.setEnabled(True)
            self.run_all_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText(f"Optimizasyon Tamamlandı (Toplam Süre: {final_time})")
    
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
