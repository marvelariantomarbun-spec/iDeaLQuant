# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QTableWidget, QTableWidgetItem, QHeaderView, 
                               QMessageBox, QLabel, QAbstractItemView)
from PySide6.QtCore import Qt
from src.core.database import db

class DatabaseManager(QDialog):
    """Veritabanı ve süreç yönetim dialogu"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Veritabanı Yönetimi")
        self.resize(900, 500)
        
        self.init_ui()
        self.load_data()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Bilgi Etiketi
        info_label = QLabel("Kayıtlı süreçleri ve optimizasyon sonuçlarını buradan yönetebilirsiniz.")
        layout.addWidget(info_label)
        
        # Tablo
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Process ID", "Tarih", "Sembol", "Periyot", "Veri", "Durum"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.table)
        
        # Butonlar
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Yenile")
        refresh_btn.clicked.connect(self.load_data)
        btn_layout.addWidget(refresh_btn)
        
        delete_btn = QPushButton("🗑️ Seçilileri Sil")
        delete_btn.clicked.connect(self.delete_selected)
        btn_layout.addWidget(delete_btn)
        
        vacuum_btn = QPushButton("🧹 Veritabanını Sıkıştır (Vacuum)")
        vacuum_btn.clicked.connect(self.vacuum_db)
        btn_layout.addWidget(vacuum_btn)
        
        btn_layout.addStretch()
        
        clear_btn = QPushButton("⚠️ TÜM VERİTABANINI TEMİZLE")
        clear_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
        clear_btn.clicked.connect(self.clear_all_db)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
    def load_data(self):
        """Süreçleri listele"""
        try:
            processes = db.get_all_processes()
            self.table.setRowCount(len(processes))
            
            for i, p in enumerate(processes):
                # Process ID
                self.table.setItem(i, 0, QTableWidgetItem(p['process_id']))
                
                # Tarih
                created_at = p['created_at']
                # Eğer str ise truncate et, datetime ise formatla
                if isinstance(created_at, str) and len(created_at) > 19:
                    created_at = created_at[:19]
                self.table.setItem(i, 1, QTableWidgetItem(str(created_at)))
                
                # Diğer sütunlar
                self.table.setItem(i, 2, QTableWidgetItem(p['symbol']))
                self.table.setItem(i, 3, QTableWidgetItem(p['period']))
                self.table.setItem(i, 4, QTableWidgetItem(f"{p['data_rows']:,} bar"))
                self.table.setItem(i, 5, QTableWidgetItem(p['status']))
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri yüklenirken hata: {str(e)}")
            
    def delete_selected(self):
        """Seçili satırları sil"""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Uyarı", "Lütfen silinecek süreç(ler)i seçin.")
            return
            
        count = len(selected_rows)
        reply = QMessageBox.question(
            self, 
            "Onay", 
            f"Seçili {count} süreci ve bunlara bağlı TÜM optimizasyon sonuçlarını silmek istediğinize emin misiniz?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            deleted_count = 0
            # Ters sırayla sil ki indeksler kaymasın (gerçi burada ID alıp siliyoruz)
            ids_to_delete = []
            for row in selected_rows:
                process_id = self.table.item(row.row(), 0).text()
                ids_to_delete.append(process_id)
            
            for pid in ids_to_delete:
                if db.delete_process(pid):
                    deleted_count += 1
            
            QMessageBox.information(self, "Başarılı", f"{deleted_count} süreç silindi.")
            self.load_data()
            
    def vacuum_db(self):
        """Vacuum işlemi"""
        try:
            db.vacuum()
            QMessageBox.information(self, "Başarılı", "Veritabanı optimize edildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Vacuum hatası: {str(e)}")
            
    def clear_all_db(self):
        """Tüm veritabanını temizle"""
        reply = QMessageBox.question(
            self, 
            "KRİTİK UYARI", 
            "BU İŞLEM GERİ ALINAMAZ!\n\nTüm süreçler, optimizasyon sonuçları ve ayarlar silinecek.\nVeritabanı tamamen sıfırlansın mı?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # İkinci onay
            reply2 = QMessageBox.question(
                self, 
                "Son Onay", 
                "Gerçekten, kesin ve net olarak EMİN MİSİNİZ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply2 == QMessageBox.Yes:
                if db.clear_database():
                    QMessageBox.information(self, "Başarılı", "Veritabanı tertemiz oldu. Yeni bir başlangıç!")
                    self.load_data()
                else:
                    QMessageBox.critical(self, "Hata", "Veritabanı temizlenirken bir sorun oluştu.")
