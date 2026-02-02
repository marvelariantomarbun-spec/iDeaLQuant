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
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from .widgets.data_panel import DataPanel
from .widgets.strategy_panel import StrategyPanel
from .widgets.optimizer_panel import OptimizerPanel
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
        
        # UI oluştur
        self._setup_ui()
        
        # Status bar
        self._setup_status_bar()
    
    def _load_stylesheet(self):
        """QSS stylesheet yükle"""
        style_path = Path(__file__).parent / "styles" / "dark_theme.qss"
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
    
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
        self.export_panel = ExportPanel()
        
        # Tab'ları ekle
        self.tab_widget.addTab(self.data_panel, "📊 Veri")
        self.tab_widget.addTab(self.strategy_panel, "⚙️ Strateji")
        self.tab_widget.addTab(self.optimizer_panel, "🔬 Optimizer")
        self.tab_widget.addTab(self.export_panel, "📤 Export")
        
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
        
        subtitle = QLabel("Algorithmic Trading Optimizer v4.1")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        # Hakkında butonu
        about_btn = QPushButton("ℹ️ Hakkında")
        about_btn.clicked.connect(self._show_about)
        layout.addWidget(about_btn)
        
        return header
    
    def _setup_status_bar(self):
        """Status bar oluştur"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Durum label
        self.status_label = QLabel("Hazır")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Versiyon
        version_label = QLabel("v4.1.0")
        self.status_bar.addPermanentWidget(version_label)
    
    def _connect_panels(self):
        """Paneller arasındaki sinyalleri bağla"""
        # Data panel -> Strategy panel
        self.data_panel.data_loaded.connect(self.strategy_panel.set_data)
        
        # Strategy panel -> Optimizer panel
        self.strategy_panel.config_changed.connect(self.optimizer_panel.set_strategy_config)
        
        # Optimizer panel -> Export panel
        self.optimizer_panel.optimization_complete.connect(self.export_panel.set_results)
    
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
