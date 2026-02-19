# -*- coding: utf-8 -*-
"""
IdealQuant - Main Entry Point
Uygulamayı başlatır
"""

import sys
import os
import atexit
import multiprocessing

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .main_window import MainWindow


def _cleanup_workers():
    """Uygulama kapanirken tum alt islemleri temizle"""
    # Terminate all active child processes to prevent BrokenPipeError
    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=1)
        except Exception:
            pass


def main():
    """Ana uygulama başlatıcı"""
    multiprocessing.freeze_support()
    
    # Register cleanup handler
    atexit.register(_cleanup_workers)
    
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
    exit_code = app.exec()
    _cleanup_workers()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

