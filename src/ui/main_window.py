# -*- coding: utf-8 -*-
"""
IdealQuant - Main Window
Ana uygulama penceresi
"""

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QLabel, 
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QIcon

from .widgets.data_panel import DataPanel
from .widgets.strategy_panel import StrategyPanel
from .widgets.optimizer_panel import OptimizerPanel
from .widgets.validation_panel import ValidationPanel
from .widgets.export_panel import ExportPanel


class MainWindow(QMainWindow):
    """IdealQuant Ana Pencere"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IdealQuant - Algorithmic Trading Optimizer")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Stil dosyasını yükle
        self._load_stylesheet()
        
        # İkon yükle
        self._load_icon()
        
        # UI oluştur
        self._setup_ui()
        
        # Status bar
        self._setup_status_bar()
    
    def _load_icon(self):
        """Pencere ikonunu yükle"""
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
    
    def _load_stylesheet(self):
        """QSS stylesheet yükle (kayıtlı temayı kullan)"""
        settings = QSettings("IdealQuant", "Desktop")
        theme_index = settings.value("theme_index", 0, type=int)
        
        themes = {0: "dark_theme.qss", 1: "professional_theme.qss", 2: "sunset_theme.qss"}
        theme_file = themes.get(theme_index, "dark_theme.qss")
        style_path = Path(__file__).parent / "styles" / theme_file
        
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        
        self._saved_theme_index = theme_index
    
    def _setup_ui(self):
        """Ana UI bileşenlerini oluştur"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Ana layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # Paneller
        self.data_panel = DataPanel()
        self.strategy_panel = StrategyPanel()
        self.optimizer_panel = OptimizerPanel()
        self.validation_panel = ValidationPanel()
        self.export_panel = ExportPanel()
        
        # Tab'ları ekle
        self.tab_widget.addTab(self.data_panel, "Veri")
        self.tab_widget.addTab(self.strategy_panel, "Strateji")
        self.tab_widget.addTab(self.optimizer_panel, "Optimizer")
        self.tab_widget.addTab(self.validation_panel, "Validasyon")
        self.tab_widget.addTab(self.export_panel, "Export")
        
        main_layout.addWidget(self.tab_widget)
        
        # Panel bağlantıları
        self._connect_panels()
    
    def _create_header(self) -> QWidget:
        """Başlık alanı oluştur"""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 10)
        
        # Logo/Başlık
        title = QLabel("IdealQuant")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        
        subtitle = QLabel("Algorithmic Trading Optimizer v1.0")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        # Tema seçici
        from PySide6.QtWidgets import QComboBox
        layout.addWidget(QLabel("Tema:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Midnight (Koyu Mavi)", "Professional (Gri)", "Sunset (Turuncu)"])
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        self.theme_combo.setMinimumWidth(150)
        # Kayıtlı temayı seç (sinyal tetiklemeden)
        if hasattr(self, '_saved_theme_index'):
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(self._saved_theme_index)
            self.theme_combo.blockSignals(False)
        layout.addWidget(self.theme_combo)
        
        layout.addSpacing(20)
        
        # Hakkında butonu
        about_btn = QPushButton("Hakkinda")
        about_btn.clicked.connect(self._show_about)
        layout.addWidget(about_btn)
        
        return header
    
    def _change_theme(self, index: int):
        """Tema değiştir ve kaydet"""
        themes = {
            0: "dark_theme.qss",
            1: "professional_theme.qss",
            2: "sunset_theme.qss"
        }
        
        theme_file = themes.get(index, "dark_theme.qss")
        style_path = Path(__file__).parent / "styles" / theme_file
        
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
            
            # Temayı kaydet
            settings = QSettings("IdealQuant", "Desktop")
            settings.setValue("theme_index", index)
            
            self.status_label.setText(f"Tema değiştirildi: {self.theme_combo.currentText()}")
    
    def _setup_status_bar(self):
        """Status bar oluştur"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Durum label
        self.status_label = QLabel("Hazır")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Versiyon
        version_label = QLabel("v1.0.0")
        self.status_bar.addPermanentWidget(version_label)
    
    def _connect_panels(self):
        """Paneller arasındaki sinyalleri bağla"""
        # Data panel -> Strategy panel (mevcut)
        self.data_panel.data_loaded.connect(self.strategy_panel.set_data)
        
        # Data panel -> Optimizer & Validation panels
        self.data_panel.data_loaded.connect(self.optimizer_panel.set_data)
        self.data_panel.data_loaded.connect(self.validation_panel.set_data)
        self.data_panel.data_loaded.connect(lambda df: self.update_status(f"Veri yuklendi: {len(df)} bar"))
        
        # Data panel -> Optimizer panel (Süreç bağlantısı)
        self.data_panel.process_created.connect(self.optimizer_panel.set_process)
        self.data_panel.process_created.connect(lambda p: self.update_status(f"Süreç oluşturuldu: {p}"))
        
        # Strategy panel -> Optimizer panel (mevcut)
        self.strategy_panel.config_changed.connect(self.optimizer_panel.set_strategy_config)
        
        # Optimizer panel -> Validation panel
        self.optimizer_panel.optimization_complete.connect(self.validation_panel.set_optimization_results)
        self.optimizer_panel.optimization_complete.connect(lambda r: self.update_status(f"Optimizasyon tamamlandı: {len(r)} sonuç"))
        
        # Optimizer panel -> Export panel (mevcut)
        self.optimizer_panel.optimization_complete.connect(self.export_panel.set_results)
        
        # Validation panel -> Export panel (Yeni: Final seçildiğinde)
        self.validation_panel.validation_complete.connect(
            lambda pid, params: self.export_panel._refresh_processes()
        )
        self.validation_panel.validation_complete.connect(
            lambda pid, params: self.update_status(f"Validasyon tamamlandı: {pid}")
        )
        
        # Export panel -> Status update
        self.export_panel.export_complete.connect(
            lambda pid: self.update_status(f"Export tamamlandı: {pid}")
        )
    
    def _show_about(self):
        """Hakkında dialogu göster"""
        QMessageBox.about(
            self,
            "IdealQuant Hakkında",
            """<h2>IdealQuant</h2>
            <p>Algorithmic Trading Optimizer v4.1</p>
            <p>VIOP piyasası için optimize edilmiş alım-satım stratejileri.</p>
            <hr>
            <p><b>Özellikler:</b></p>
            <ul>
                <li>Strateji 1: Score-Based Gatekeeper</li>
                <li>Strateji 2: ARS Trend Takip v2</li>
                <li>Hibrit Grup Optimizasyonu</li>
                <li>IdealData Export</li>
            </ul>
            """
        )
    
    def update_status(self, message: str):
        """Status bar mesajını güncelle"""
        self.status_label.setText(message)
