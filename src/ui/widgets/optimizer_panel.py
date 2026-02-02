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

from src.core.database import db


# Strateji 1 Parametre Grupları (20 parametre)
STRATEGY1_PARAM_GROUPS = {
    'ARS': {
        'label': 'ARS Parametreleri',
        'params': {
            'ars_period': {'label': 'ARS Periyot', 'type': 'int', 'default': 3, 'min': 2, 'max': 15, 'step': 1},
            'ars_k': {'label': 'ARS K', 'type': 'float', 'default': 0.01, 'min': 0.005, 'max': 0.03, 'step': 0.005},
        }
    },
    'ADX': {
        'label': 'ADX Parametreleri',
        'params': {
            'adx_period': {'label': 'ADX Periyot', 'type': 'int', 'default': 17, 'min': 10, 'max': 30, 'step': 2},
            'adx_threshold': {'label': 'ADX Esik', 'type': 'float', 'default': 25.0, 'min': 15.0, 'max': 35.0, 'step': 5.0},
        }
    },
    'MACDV': {
        'label': 'MACD-V Parametreleri',
        'params': {
            'macdv_short': {'label': 'Kisa Periyot', 'type': 'int', 'default': 13, 'min': 8, 'max': 18, 'step': 1},
            'macdv_long': {'label': 'Uzun Periyot', 'type': 'int', 'default': 28, 'min': 20, 'max': 40, 'step': 2},
            'macdv_signal': {'label': 'Sinyal Periyot', 'type': 'int', 'default': 8, 'min': 5, 'max': 15, 'step': 1},
            'macdv_threshold': {'label': 'MACDV Esik', 'type': 'float', 'default': 0.0, 'min': -50.0, 'max': 50.0, 'step': 10.0},
        }
    },
    'NetLot': {
        'label': 'NetLot Parametreleri',
        'params': {
            'netlot_period': {'label': 'Periyot', 'type': 'int', 'default': 5, 'min': 3, 'max': 10, 'step': 1},
            'netlot_threshold': {'label': 'Esik', 'type': 'float', 'default': 20.0, 'min': 10.0, 'max': 50.0, 'step': 5.0},
        }
    },
    'Yatay_Filtre': {
        'label': 'Yatay Filtre',
        'params': {
            'ars_mesafe_threshold': {'label': 'ARS Mesafe', 'type': 'float', 'default': 0.25, 'min': 0.1, 'max': 0.5, 'step': 0.05},
            'bb_period': {'label': 'BB Periyot', 'type': 'int', 'default': 20, 'min': 15, 'max': 30, 'step': 5},
            'bb_std': {'label': 'BB StdDev', 'type': 'float', 'default': 2.0, 'min': 1.5, 'max': 3.0, 'step': 0.5},
            'bb_width_multiplier': {'label': 'BB Width Mult', 'type': 'float', 'default': 0.8, 'min': 0.5, 'max': 1.5, 'step': 0.1},
            'bb_avg_period': {'label': 'BB Avg Periyot', 'type': 'int', 'default': 50, 'min': 30, 'max': 100, 'step': 10},
            'yatay_ars_bars': {'label': 'ARS Bar', 'type': 'int', 'default': 10, 'min': 5, 'max': 20, 'step': 5},
            'yatay_adx_threshold': {'label': 'Yatay ADX Esik', 'type': 'float', 'default': 20.0, 'min': 15.0, 'max': 30.0, 'step': 5.0},
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
    
    progress = Signal(int, str)
    result_ready = Signal(list)
    error = Signal(str)
    
    def __init__(self, config: dict, data, param_ranges: dict, 
                 method: str = "Hibrit Grup", strategy_index: int = 0,
                 process_id: str = None):
        super().__init__()
        self.config = config
        self.data = data
        self.param_ranges = param_ranges
        self.method = method
        self.strategy_index = strategy_index  # 0 = Strateji 1, 1 = Strateji 2
        self.process_id = process_id
        self._is_running = True
    
    def run(self):
        try:
            if self.method == "Hibrit Grup":
                self._run_hybrid()
            elif self.method == "Genetik":
                self._run_genetic()
            elif self.method == "Bayesian":
                self._run_bayesian()
            else:
                self.error.emit(f"Bilinmeyen yöntem: {self.method}")
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
    
    def _run_hybrid(self):
        """Hibrit Grup optimizasyonu"""
        from src.optimization.hybrid_group_optimizer import (
            HybridGroupOptimizer, IndicatorCache, STRATEGY1_GROUPS, STRATEGY2_GROUPS
        )
        
        self.progress.emit(10, "Cache oluşturuluyor...")
        cache = IndicatorCache(self.data)
        
        # Strateji seçimine göre grupları belirle
        groups = STRATEGY1_GROUPS if self.strategy_index == 0 else STRATEGY2_GROUPS
        strategy_name = "Strateji 1" if self.strategy_index == 0 else "Strateji 2"
        
        self.progress.emit(20, f"Hibrit Optimizer başlatılıyor ({strategy_name})...")
        optimizer = HybridGroupOptimizer(
            groups, 
            process_id=self.process_id, 
            strategy_index=self.strategy_index
        )
        
        self.progress.emit(30, "Bağımsız gruplar optimize ediliyor...")
        optimizer.run_independent_phase()
        
        if not self._is_running:
            return
        
        self.progress.emit(70, "Kombinasyon aşaması...")
        optimizer.run_combination_phase()
        
        self.progress.emit(90, "Sonuçlar derleniyor...")
        results = optimizer.get_best_results(top_n=20)
        
        self.progress.emit(100, "Tamamlandı!")
        self.result_ready.emit(results)
    
    def _run_genetic(self):
        """Genetik Algoritma optimizasyonu - Her iki strateji için"""
        from src.optimization.genetic_optimizer import GeneticOptimizer, GeneticConfig
        
        strategy_name = "Strateji 1" if self.strategy_index == 0 else "Strateji 2"
        self.progress.emit(10, f"Genetik Algoritma başlatılıyor ({strategy_name})...")
        
        config = GeneticConfig(
            population_size=50,
            generations=30,
            elite_ratio=0.1,
            crossover_rate=0.8,
            mutation_rate=0.15
        )
        
        self.progress.emit(20, "Popülasyon oluşturuluyor...")
        optimizer = GeneticOptimizer(self.data, config, strategy_index=self.strategy_index)
        
        self.progress.emit(30, "Evrim başlıyor...")
        results = optimizer.run(verbose=False)
        
        self.progress.emit(100, "Genetik optimizasyon tamamlandı!")
        self.result_ready.emit(results if isinstance(results, list) else [results])
    
    def _run_bayesian(self):
        """Bayesian (Optuna) optimizasyonu - Her iki strateji için"""
        from src.optimization.bayesian_optimizer import BayesianOptimizer
        
        strategy_name = "Strateji 1" if self.strategy_index == 0 else "Strateji 2"
        self.progress.emit(10, f"Bayesian Optimizer başlatılıyor ({strategy_name})...")
        
        n_trials = 100  # Deneme sayısı
        
        self.progress.emit(20, f"Optuna çalışması oluşturuluyor ({n_trials} deneme)...")
        optimizer = BayesianOptimizer(self.data, n_trials=n_trials, strategy_index=self.strategy_index)
        
        self.progress.emit(30, "Akıllı arama başlıyor...")
        results = optimizer.run(verbose=False)
        
        self.progress.emit(100, "Bayesian optimizasyon tamamlandı!")
        self.result_ready.emit(results if isinstance(results, list) else [results])
    
    def stop(self):
        self._is_running = False


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
        
        # Paralel işlem
        top_row.addWidget(QLabel("Paralel:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(8)
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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Hazir")
        self.status_label.setAlignment(Qt.AlignCenter)
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
        
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_optimization)
        btn_row.addWidget(self.stop_btn)
        
        layout.addLayout(btn_row)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Sonuçlar grubu"""
        group = QGroupBox("Sonuclar")
        layout = QVBoxLayout(group)
        
        # İstatistikler
        stats_row = QHBoxLayout()
        self.results_stats_label = QLabel("Henuz sonuc yok")
        stats_row.addWidget(self.results_stats_label)
        stats_row.addStretch()
        
        export_btn = QPushButton("Disa Aktar")
        export_btn.clicked.connect(self._export_results)
        stats_row.addWidget(export_btn)
        layout.addLayout(stats_row)
        
        # Sonuç tablosu
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.results_table)
        
        return group
    
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
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Baslatiliyor...")
        
        # Worker başlat
        method = self.method_combo.currentText()
        strategy_index = self.strategy_combo.currentIndex()
        self.worker = OptimizationWorker(
            self.config, self.data, param_ranges, method, strategy_index,
            process_id=self.current_process_id
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
    
    def _stop_optimization(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Durduruluyor...")
    
    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
    
    def _on_result(self, results: list):
        self._display_results(results)
        
        # Veritabanına kaydet
        if self.current_process_id and results:
            method = self.method_combo.currentText().lower().replace(" ", "_").replace("hibrit_grup", "hibrit")
            strategy_index = self.strategy_combo.currentIndex()
            
            # En iyi sonucu kaydet
            best = results[0] if results else {}
            params = best.get('params', best)
            
            db.save_optimization_result(
                process_id=self.current_process_id,
                strategy_index=strategy_index,
                method=method,
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
        QMessageBox.critical(self, "Hata", f"Optimizasyon hatasi: {error_msg}")
        self.status_label.setText("Hata!")
    
    def _on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
    
    def _display_results(self, results: list):
        if not results:
            self.results_stats_label.setText("Sonuc bulunamadi")
            return
        
        cols = ['Sira', 'Net Kar', 'Islem', 'PF', 'Max DD']
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(len(results))
        
        for row_idx, result in enumerate(results):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{result.get('net_profit', 0):.0f}"))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(str(result.get('trades', 0))))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(f"{result.get('pf', 0):.2f}"))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(f"{result.get('max_dd', 0):.0f}"))
        
        self.results_stats_label.setText(f"{len(results)} sonuc bulundu")
    
    def _export_results(self):
        QMessageBox.information(self, "Bilgi", "Sonuc disa aktarma yakinda eklenecek.")
    
    def set_strategy_config(self, config: dict):
        self.config = config
        # Strateji seçimini güncelle
        strategy_idx = config.get('strategy', 1) - 1
        self.strategy_combo.setCurrentIndex(strategy_idx)
    
    def set_data(self, df):
        self.data = df
