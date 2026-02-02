# -*- coding: utf-8 -*-
"""
IdealQuant - Data Panel
Veri yönetimi paneli
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLineEdit, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDateEdit, QSpinBox, QMessageBox
)
from PySide6.QtCore import Signal, Qt, QDate
import pandas as pd


class DataPanel(QWidget):
    """Veri yükleme ve önizleme paneli"""
    
    # Signals
    data_loaded = Signal(object)  # DataFrame gönderir
    
    def __init__(self):
        super().__init__()
        self.df = None
        self._setup_ui()
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Dosya seçimi grubu
        file_group = self._create_file_group()
        layout.addWidget(file_group)
        
        # Filtre grubu
        filter_group = self._create_filter_group()
        layout.addWidget(filter_group)
        
        # Önizleme tablosu
        preview_group = self._create_preview_group()
        layout.addWidget(preview_group, 1)  # Stretch
    
    def _create_file_group(self) -> QGroupBox:
        """Dosya seçimi grubu"""
        group = QGroupBox("📁 Veri Kaynağı")
        layout = QVBoxLayout(group)
        
        # CSV dosya seçimi
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("CSV Dosyası:"))
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("VIP_X030T_1dk_.csv gibi...")
        csv_row.addWidget(self.csv_path_edit, 1)
        browse_btn = QPushButton("Gözat...")
        browse_btn.clicked.connect(self._browse_csv)
        csv_row.addWidget(browse_btn)
        layout.addLayout(csv_row)
        
        # Yükle butonu
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        load_btn = QPushButton("📥 Veriyi Yükle")
        load_btn.setObjectName("primaryButton")
        load_btn.clicked.connect(self._load_data)
        btn_row.addWidget(load_btn)
        layout.addLayout(btn_row)
        
        return group
    
    def _create_filter_group(self) -> QGroupBox:
        """Filtre grubu"""
        group = QGroupBox("🔍 Filtreler")
        layout = QHBoxLayout(group)
        
        # Tarih aralığı
        layout.addWidget(QLabel("Başlangıç:"))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate(2024, 1, 1))
        layout.addWidget(self.start_date)
        
        layout.addWidget(QLabel("Bitiş:"))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        layout.addWidget(self.end_date)
        
        layout.addSpacing(20)
        
        # Son N satır
        layout.addWidget(QLabel("Son N satır:"))
        self.last_n_rows = QSpinBox()
        self.last_n_rows.setRange(0, 1000000)
        self.last_n_rows.setValue(0)
        self.last_n_rows.setSpecialValueText("Tümü")
        layout.addWidget(self.last_n_rows)
        
        layout.addStretch()
        
        # Filtrele butonu
        filter_btn = QPushButton("🔄 Filtrele")
        filter_btn.clicked.connect(self._apply_filter)
        layout.addWidget(filter_btn)
        
        return group
    
    def _create_preview_group(self) -> QGroupBox:
        """Önizleme tablosu grubu"""
        group = QGroupBox("📋 Veri Önizleme")
        layout = QVBoxLayout(group)
        
        # İstatistikler
        stats_row = QHBoxLayout()
        self.stats_label = QLabel("Veri yüklenmedi")
        stats_row.addWidget(self.stats_label)
        stats_row.addStretch()
        layout.addLayout(stats_row)
        
        # Tablo
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.preview_table)
        
        return group
    
    def _browse_csv(self):
        """CSV dosyası seç"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "CSV Dosyası Seç",
            str(Path.home()),
            "CSV Dosyaları (*.csv);;Tüm Dosyalar (*)"
        )
        if file_path:
            self.csv_path_edit.setText(file_path)
    
    def _load_data(self):
        """Veriyi yükle"""
        csv_path = self.csv_path_edit.text().strip()
        
        if not csv_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir CSV dosyası seçin.")
            return
        
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "Hata", f"Dosya bulunamadı: {csv_path}")
            return
        
        try:
            # CSV yükle
            df = pd.read_csv(
                csv_path, 
                sep=';', 
                decimal=',', 
                encoding='cp1254',
                header=0,
                low_memory=False
            )
            df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
            
            # Sayısal dönüşüm
            for c in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
            df.dropna(inplace=True)
            
            # DateTime oluştur
            df['DateTime'] = pd.to_datetime(
                df['Tarih'] + ' ' + df['Saat'], 
                format='%d.%m.%Y %H:%M:%S', 
                errors='coerce'
            )
            df = df.dropna(subset=['DateTime']).reset_index(drop=True)
            
            self.df = df
            self._update_preview()
            
            # Signal gönder
            self.data_loaded.emit(self.df)
            
            QMessageBox.information(
                self, 
                "Başarılı", 
                f"✅ {len(df):,} satır yüklendi.\n"
                f"Tarih aralığı: {df['DateTime'].min()} - {df['DateTime'].max()}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri yüklenirken hata: {str(e)}")
    
    def _apply_filter(self):
        """Filtreleri uygula"""
        if self.df is None:
            return
        
        # Tarih filtresi
        start = self.start_date.date().toPython()
        end = self.end_date.date().toPython()
        
        filtered = self.df[
            (self.df['DateTime'].dt.date >= start) & 
            (self.df['DateTime'].dt.date <= end)
        ]
        
        # Son N satır
        n = self.last_n_rows.value()
        if n > 0:
            filtered = filtered.tail(n)
        
        self._update_preview(filtered)
    
    def _update_preview(self, df=None):
        """Önizleme tablosunu güncelle"""
        if df is None:
            df = self.df
        
        if df is None:
            return
        
        # İstatistikler
        self.stats_label.setText(
            f"📊 Toplam: {len(df):,} satır | "
            f"📅 {df['DateTime'].min().strftime('%Y-%m-%d')} → {df['DateTime'].max().strftime('%Y-%m-%d')}"
        )
        
        # Tablo (son 100 satır)
        preview_df = df.tail(100)
        cols = ['DateTime', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Lot']
        
        self.preview_table.setRowCount(len(preview_df))
        self.preview_table.setColumnCount(len(cols))
        self.preview_table.setHorizontalHeaderLabels(cols)
        
        for row_idx, (_, row) in enumerate(preview_df.iterrows()):
            for col_idx, col in enumerate(cols):
                value = row[col]
                if col == 'DateTime':
                    text = str(value)[:19]
                elif isinstance(value, float):
                    text = f"{value:.2f}"
                else:
                    text = str(value)
                self.preview_table.setItem(row_idx, col_idx, QTableWidgetItem(text))
    
    def get_data(self) -> pd.DataFrame:
        """Yüklü veriyi döndür"""
        return self.df
