# -*- coding: utf-8 -*-
"""
IdealQuant - Optimizer Panel
Optimizasyon kontrolü ve sonuç görüntüleme
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QSpinBox,
    QDoubleSpinBox, QFormLayout, QMessageBox, QSplitter
)
from PySide6.QtCore import Signal, Qt, QThread
import pandas as pd


class OptimizationWorker(QThread):
    """Arka plan optimizasyon thread'i"""
    
    progress = Signal(int, str)  # (yüzde, mesaj)
    result_ready = Signal(list)  # Sonuç listesi
    error = Signal(str)          # Hata mesajı
    
    def __init__(self, config: dict, data):
        super().__init__()
        self.config = config
        self.data = data
        self._is_running = True
    
    def run(self):
        """Optimizasyonu çalıştır"""
        try:
            from src.optimization.hybrid_group_optimizer import HybridGroupOptimizer, IndicatorCache
            
            self.progress.emit(10, "Cache oluşturuluyor...")
            cache = IndicatorCache(self.data)
            
            self.progress.emit(20, "Optimizer başlatılıyor...")
            optimizer = HybridGroupOptimizer()
            
            # TODO: Config'e göre parametre aralıklarını ayarla
            
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
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        """Optimizasyonu durdur"""
        self._is_running = False


class OptimizerPanel(QWidget):
    """Optimizasyon paneli"""
    
    # Signals
    optimization_complete = Signal(list)  # Sonuç listesi
    
    def __init__(self):
        super().__init__()
        self.config = {}
        self.data = None
        self.worker = None
        self._setup_ui()
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Splitter (Ayarlar | Sonuçlar)
        splitter = QSplitter(Qt.Horizontal)
        
        # Sol panel - Ayarlar
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Optimizer ayarları
        settings_group = self._create_settings_group()
        left_layout.addWidget(settings_group)
        
        # Kontrol butonları
        control_group = self._create_control_group()
        left_layout.addWidget(control_group)
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # Sağ panel - Sonuçlar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        results_group = self._create_results_group()
        right_layout.addWidget(results_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
    
    def _create_settings_group(self) -> QGroupBox:
        """Optimizer ayarları grubu"""
        group = QGroupBox("⚙️ Optimizer Ayarları")
        layout = QFormLayout(group)
        
        # Yöntem
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Hibrit Grup Optimizer",
            "Grid Search (Kaba)",
            "Bayesian Optimization"
        ])
        layout.addRow("Yöntem:", self.method_combo)
        
        # Paralel işlem sayısı
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(8)
        layout.addRow("Paralel İşlem:", self.workers_spin)
        
        # Top N sonuç
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(5, 100)
        self.topn_spin.setValue(20)
        layout.addRow("Top N Sonuç:", self.topn_spin)
        
        # Minimum işlem sayısı
        self.min_trades_spin = QSpinBox()
        self.min_trades_spin.setRange(10, 1000)
        self.min_trades_spin.setValue(100)
        layout.addRow("Min İşlem:", self.min_trades_spin)
        
        return group
    
    def _create_control_group(self) -> QGroupBox:
        """Kontrol butonları grubu"""
        group = QGroupBox("🎮 Kontrol")
        layout = QVBoxLayout(group)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Durum etiketi
        self.status_label = QLabel("Hazır")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Butonlar
        btn_row = QHBoxLayout()
        
        self.start_btn = QPushButton("▶️ Başlat")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self._start_optimization)
        btn_row.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Durdur")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_optimization)
        btn_row.addWidget(self.stop_btn)
        
        layout.addLayout(btn_row)
        
        return group
    
    def _create_results_group(self) -> QGroupBox:
        """Sonuçlar grubu"""
        group = QGroupBox("📊 Sonuçlar")
        layout = QVBoxLayout(group)
        
        # İstatistikler
        stats_row = QHBoxLayout()
        self.results_stats_label = QLabel("Henüz sonuç yok")
        stats_row.addWidget(self.results_stats_label)
        stats_row.addStretch()
        
        export_btn = QPushButton("📤 Sonuçları Dışa Aktar")
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
    
    def _start_optimization(self):
        """Optimizasyonu başlat"""
        if self.data is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce veri yükleyin.")
            return
        
        if not self.config:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce strateji parametrelerini ayarlayın.")
            return
        
        # UI güncelle
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Başlatılıyor...")
        
        # Worker başlat
        self.worker = OptimizationWorker(self.config, self.data)
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
    
    def _stop_optimization(self):
        """Optimizasyonu durdur"""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Durduruluyor...")
    
    def _on_progress(self, percent: int, message: str):
        """İlerleme güncellemesi"""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
    
    def _on_result(self, results: list):
        """Sonuç geldi"""
        self._display_results(results)
        self.optimization_complete.emit(results)
    
    def _on_error(self, error_msg: str):
        """Hata oluştu"""
        QMessageBox.critical(self, "Hata", f"Optimizasyon hatası: {error_msg}")
        self.status_label.setText("Hata!")
    
    def _on_finished(self):
        """Thread tamamlandı"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
    
    def _display_results(self, results: list):
        """Sonuçları tabloda göster"""
        if not results:
            self.results_stats_label.setText("Sonuç bulunamadı")
            return
        
        # Tablo oluştur
        cols = ['Sıra', 'Net Kar', 'İşlem', 'PF', 'Max DD']
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(len(results))
        
        for row_idx, result in enumerate(results):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(row_idx + 1)))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{result.get('net_profit', 0):.0f}"))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(str(result.get('trades', 0))))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(f"{result.get('pf', 0):.2f}"))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(f"{result.get('max_dd', 0):.0f}"))
        
        self.results_stats_label.setText(f"✅ {len(results)} sonuç bulundu")
    
    def _export_results(self):
        """Sonuçları dışa aktar (TODO)"""
        QMessageBox.information(self, "Bilgi", "Sonuç dışa aktarma yakında eklenecek.")
    
    def set_strategy_config(self, config: dict):
        """Strateji konfigürasyonunu ayarla"""
        self.config = config
    
    def set_data(self, df):
        """Veriyi ayarla"""
        self.data = df
