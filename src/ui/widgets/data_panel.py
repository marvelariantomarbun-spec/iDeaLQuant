# -*- coding: utf-8 -*-
"""
IdealQuant - Data Panel
Veri yönetimi paneli (CSV + IdealData Binary)
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLineEdit, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDateEdit, QSpinBox, QMessageBox, QComboBox,
    QRadioButton, QButtonGroup, QTabWidget
)
from PySide6.QtCore import Signal, Qt, QDate
import pandas as pd

from src.core.database import db


class DataPanel(QWidget):
    """Veri yükleme ve önizleme paneli"""
    
    # Signals
    data_loaded = Signal(object)  # DataFrame gönderir
    process_created = Signal(str)  # process_id gönderir
    
    # IdealData default path
    IDEAL_DATA_PATH = r"D:\\iDeal\\ChartData"
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.current_process_id = None
        self._setup_ui()
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Veri kaynağı seçimi (Tab)
        source_tabs = QTabWidget()
        
        # Tab 1: IdealData Binary
        ideal_tab = self._create_ideal_tab()
        source_tabs.addTab(ideal_tab, "IdealData")
        
        # Tab 2: CSV
        csv_tab = self._create_csv_tab()
        source_tabs.addTab(csv_tab, "CSV Dosyası")
        
        layout.addWidget(source_tabs)
        
        # Filtre grubu
        filter_group = self._create_filter_group()
        layout.addWidget(filter_group)
        
        # Önizleme tablosu
        preview_group = self._create_preview_group()
        layout.addWidget(preview_group, 1)  # Stretch
    
    def _create_ideal_tab(self) -> QWidget:
        """IdealData veri kaynağı tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ChartData Path
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("ChartData Yolu:"))
        self.ideal_path_edit = QLineEdit(self.IDEAL_DATA_PATH)
        path_row.addWidget(self.ideal_path_edit, 1)
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(self._browse_ideal_path)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)
        
        # Seçiciler
        select_row = QHBoxLayout()
        
        # Pazar
        select_row.addWidget(QLabel("Pazar:"))
        self.market_combo = QComboBox()
        self.market_combo.addItems(['VIP', 'IMKBH', 'IMKBX', 'FX', 'DOVIZ'])
        self.market_combo.currentTextChanged.connect(self._on_market_changed)
        select_row.addWidget(self.market_combo)
        
        # Periyot
        select_row.addWidget(QLabel("Periyot:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(['1', '5', '15', '60', 'G'])
        self.period_combo.currentTextChanged.connect(self._on_period_changed)
        select_row.addWidget(self.period_combo)
        
        # Sembol
        select_row.addWidget(QLabel("Sembol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setMinimumWidth(150)
        select_row.addWidget(self.symbol_combo)
        
        select_row.addStretch()
        
        # Sembolleri yükle butonu
        refresh_btn = QPushButton("Yenile")
        refresh_btn.clicked.connect(self._refresh_symbols)
        select_row.addWidget(refresh_btn)
        
        layout.addLayout(select_row)
        
        # Yükle butonu
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        load_btn = QPushButton("IdealData'dan Yükle")
        load_btn.setObjectName("primaryButton")
        load_btn.clicked.connect(self._load_ideal_data)
        btn_row.addWidget(load_btn)
        layout.addLayout(btn_row)
        
        # İlk yükleme
        self._refresh_symbols()
        
        return widget
    
    def _create_csv_tab(self) -> QWidget:
        """CSV veri kaynağı tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
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
        load_btn = QPushButton("CSV'den Yükle")
        load_btn.setObjectName("primaryButton")
        load_btn.clicked.connect(self._load_csv_data)
        btn_row.addWidget(load_btn)
        layout.addLayout(btn_row)
        
        layout.addStretch()
        
        return widget
    
    def _create_filter_group(self) -> QGroupBox:
        """Filtre grubu"""
        group = QGroupBox("Filtreler")
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
        filter_btn = QPushButton("Filtrele")
        filter_btn.clicked.connect(self._apply_filter)
        layout.addWidget(filter_btn)
        
        return group
    
    def _create_preview_group(self) -> QGroupBox:
        """Önizleme tablosu grubu"""
        group = QGroupBox("Veri Önizleme")
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
    
    def _browse_ideal_path(self):
        """IdealData klasörü seç"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "ChartData Klasörü Seç",
            self.ideal_path_edit.text()
        )
        if folder:
            self.ideal_path_edit.setText(folder)
            self._refresh_symbols()
    
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
    
    def _on_market_changed(self, market: str):
        """Pazar değiştiğinde"""
        self._refresh_symbols()
    
    def _on_period_changed(self, period: str):
        """Periyot değiştiğinde"""
        self._refresh_symbols()
    
    def _refresh_symbols(self):
        """Sembolleri yenile"""
        try:
            from src.data.ideal_parser import list_symbols
            
            chart_data = self.ideal_path_edit.text()
            market = self.market_combo.currentText()
            period = self.period_combo.currentText()
            
            symbols = list_symbols(chart_data, market, period)
            
            self.symbol_combo.clear()
            self.symbol_combo.addItems(symbols)
            
            # Varsayılan olarak X030-T seç (vadeli, akşam seansı dahil)
            if 'X030-T' in symbols:
                self.symbol_combo.setCurrentText('X030-T')
            elif 'X030' in symbols:
                self.symbol_combo.setCurrentText('X030')
            
        except Exception as e:
            print(f"Sembol yükleme hatası: {e}")
    
    def _load_ideal_data(self):
        """IdealData'dan veri yükle"""
        try:
            from src.data.ideal_parser import load_ideal_data
            
            chart_data = self.ideal_path_edit.text()
            market = self.market_combo.currentText()
            symbol = self.symbol_combo.currentText()
            period = self.period_combo.currentText()
            
            if not symbol:
                QMessageBox.warning(self, "Uyarı", "Lütfen bir sembol seçin.")
                return
            
            df = load_ideal_data(chart_data, market, symbol, period)
            
            if df is None or len(df) == 0:
                QMessageBox.warning(self, "Uyarı", f"{symbol} için veri bulunamadı.")
                return
            
            # Kolon isimlerini standartlaştır
            df = df.rename(columns={
                'Open': 'Acilis',
                'High': 'Yuksek',
                'Low': 'Dusuk',
                'Close': 'Kapanis',
                'Volume': 'Lot',
                'Amount': 'Hacim'
            })
            
            self.df = df
            self._update_preview()
            
            # Süreç oluştur
            full_symbol = f"{market}_{symbol}"
            process_id = db.create_process(
                symbol=full_symbol,
                period=f"{period}dk" if period.isdigit() else period,
                data_file=f"{symbol}.{period}",
                data_rows=len(df)
            )
            self.current_process_id = process_id
            
            # Signals gönder
            self.data_loaded.emit(self.df)
            self.process_created.emit(process_id)
            
            QMessageBox.information(
                self, 
                "Başarılı", 
                f"{len(df):,} bar yüklendi.\n"
                f"Süreç: {process_id}\n"
                f"Tarih: {df['DateTime'].min()} - {df['DateTime'].max()}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri yüklenirken hata: {str(e)}")
    
    def _load_csv_data(self):
        """CSV'den veri yükle"""
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
            
            # CSV dosya adından sembol ve periyot çıkar
            fname = Path(csv_path).stem
            parts = fname.split('_')
            symbol = parts[0] if parts else 'UNKNOWN'
            period = parts[1] if len(parts) > 1 else '1dk'
            
            # Süreç oluştur
            process_id = db.create_process(
                symbol=symbol,
                period=period,
                data_file=Path(csv_path).name,
                data_rows=len(df)
            )
            self.current_process_id = process_id
            
            # Signals gönder
            self.data_loaded.emit(self.df)
            self.process_created.emit(process_id)
            
            QMessageBox.information(
                self, 
                "Başarılı", 
                f"{len(df):,} satır yüklendi.\n"
                f"Süreç: {process_id}\n"
                f"Tarih: {df['DateTime'].min()} - {df['DateTime'].max()}"
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
            f"Toplam: {len(df):,} bar | "
            f"{df['DateTime'].min().strftime('%Y-%m-%d')} -> {df['DateTime'].max().strftime('%Y-%m-%d')}"
        )
        
        # Tablo (son 100 satır)
        preview_df = df.tail(100)
        cols = ['DateTime', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Lot']
        
        # Mevcut kolonları kullan
        available_cols = [c for c in cols if c in preview_df.columns]
        
        self.preview_table.setRowCount(len(preview_df))
        self.preview_table.setColumnCount(len(available_cols))
        self.preview_table.setHorizontalHeaderLabels(available_cols)
        
        for row_idx, (_, row) in enumerate(preview_df.iterrows()):
            for col_idx, col in enumerate(available_cols):
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
