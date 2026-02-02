# -*- coding: utf-8 -*-
"""
IdealQuant - Export Panel
IdealData export paneli
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QTextEdit, QFileDialog, QMessageBox, QFormLayout
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont


class ExportPanel(QWidget):
    """IdealData export paneli"""
    
    def __init__(self):
        super().__init__()
        self.results = []
        self.selected_params = {}
        self._setup_ui()
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Üst kısım - Ayarlar
        settings_group = self._create_settings_group()
        layout.addWidget(settings_group)
        
        # Parametre seçimi
        params_group = self._create_params_group()
        layout.addWidget(params_group)
        
        # Önizleme
        preview_group = self._create_preview_group()
        layout.addWidget(preview_group, 1)
        
        # Export butonları
        export_row = QHBoxLayout()
        export_row.addStretch()
        
        preview_btn = QPushButton("👁️ Önizleme")
        preview_btn.clicked.connect(self._generate_preview)
        export_row.addWidget(preview_btn)
        
        export_btn = QPushButton("📤 Export Et")
        export_btn.setObjectName("primaryButton")
        export_btn.clicked.connect(self._export_script)
        export_row.addWidget(export_btn)
        
        layout.addLayout(export_row)
    
    def _create_settings_group(self) -> QGroupBox:
        """Export ayarları grubu"""
        group = QGroupBox("⚙️ Export Ayarları")
        layout = QFormLayout(group)
        
        # Sembol
        self.symbol_edit = QLineEdit("VIP_X030T")
        layout.addRow("Sembol:", self.symbol_edit)
        
        # Periyot
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1", "5", "15", "60", "240"])
        self.period_combo.setCurrentText("15")
        layout.addRow("Periyot (dk):", self.period_combo)
        
        # Vade Tipi
        self.vade_combo = QComboBox()
        self.vade_combo.addItems(["ENDEKS", "SPOT"])
        layout.addRow("Vade Tipi:", self.vade_combo)
        
        # Strateji
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Strateji 1 - Gatekeeper",
            "Strateji 2 - ARS Trend v2",
            "Birleşik Robot"
        ])
        layout.addRow("Strateji:", self.strategy_combo)
        
        # Çıktı klasörü
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit("d:/Projects/IdealQuant/export")
        output_row.addWidget(self.output_edit)
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(browse_btn)
        layout.addRow("Çıktı Klasörü:", output_row)
        
        return group
    
    def _create_params_group(self) -> QGroupBox:
        """Parametre seçimi grubu"""
        group = QGroupBox("📊 Seçili Parametreler")
        layout = QVBoxLayout(group)
        
        self.params_label = QLabel("Optimizer'dan sonuç seçilmedi.\n\nÖnce Optimizer panelinden optimizasyon çalıştırın veya manuel parametre girin.")
        self.params_label.setWordWrap(True)
        layout.addWidget(self.params_label)
        
        return group
    
    def _create_preview_group(self) -> QGroupBox:
        """Kod önizleme grubu"""
        group = QGroupBox("📝 Kod Önizleme")
        layout = QVBoxLayout(group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 10))
        self.preview_text.setPlaceholderText("Önizleme için 'Önizleme' butonuna tıklayın...")
        layout.addWidget(self.preview_text)
        
        return group
    
    def _browse_output(self):
        """Çıktı klasörü seç"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Çıktı Klasörü Seç",
            self.output_edit.text()
        )
        if folder:
            self.output_edit.setText(folder)
    
    def _generate_preview(self):
        """Kod önizlemesi oluştur"""
        try:
            from src.export.idealdata_exporter import IdealDataExporter
            
            symbol = self.symbol_edit.text()
            period = int(self.period_combo.currentText())
            vade_tipi = self.vade_combo.currentText()
            strategy_idx = self.strategy_combo.currentIndex()
            
            exporter = IdealDataExporter(symbol, period)
            
            # Parametreleri al (varsa)
            params = self.selected_params or {}
            
            if strategy_idx == 0:
                code = exporter._generate_strategy1_code(params, vade_tipi)
            elif strategy_idx == 1:
                code = exporter._generate_strategy2_code(params, vade_tipi)
            else:
                code = exporter._generate_combined_robot(params, params, vade_tipi)
            
            self.preview_text.setPlainText(code)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Önizleme hatası: {str(e)}")
    
    def _export_script(self):
        """Script dosyasını oluştur"""
        try:
            from src.export.idealdata_exporter import IdealDataExporter
            
            symbol = self.symbol_edit.text()
            period = int(self.period_combo.currentText())
            vade_tipi = self.vade_combo.currentText()
            strategy_idx = self.strategy_combo.currentIndex()
            output_dir = Path(self.output_edit.text())
            
            # Klasör yoksa oluştur
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exporter = IdealDataExporter(symbol, period)
            params = self.selected_params or {}
            
            # Dosya adı
            strategy_names = ["Gatekeeper", "ARS_Trend_v2", "Combined"]
            filename = f"{symbol}_{period}DK_{strategy_names[strategy_idx]}.cs"
            filepath = output_dir / filename
            
            # Kod oluştur
            if strategy_idx == 0:
                code = exporter._generate_strategy1_code(params, vade_tipi)
            elif strategy_idx == 1:
                code = exporter._generate_strategy2_code(params, vade_tipi)
            else:
                code = exporter._generate_combined_robot(params, params, vade_tipi)
            
            # Dosyaya yaz
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            QMessageBox.information(
                self, 
                "Başarılı", 
                f"✅ Script başarıyla oluşturuldu:\n\n{filepath}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Export hatası: {str(e)}")
    
    def set_results(self, results: list):
        """Optimizer sonuçlarını al"""
        self.results = results
        
        if results:
            # En iyi sonucu seç
            best = results[0]
            self.selected_params = best
            
            # Label güncelle
            params_text = "En iyi sonuç seçildi:\n\n"
            for key, value in best.items():
                if key not in ['net_profit', 'trades', 'pf', 'max_dd']:
                    params_text += f"• {key}: {value}\n"
            
            self.params_label.setText(params_text)
