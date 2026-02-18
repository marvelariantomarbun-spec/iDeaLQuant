# -*- coding: utf-8 -*-
"""
IdealQuant - Main Entry Point
Uygulamayı başlatır
"""

import sys
import os

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .main_window import MainWindow


def main():
    """Ana uygulama başlatıcı"""
    print("[DEBUG] App Starting...")
    # High DPI desteği
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("IdealQuant")
    app.setApplicationVersion("4.1.0")
    app.setOrganizationName("IdealQuant")
    
    # Varsayılan font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Ana pencere
    window = MainWindow()
    window.show()
    
    # Uygulama döngüsü
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
