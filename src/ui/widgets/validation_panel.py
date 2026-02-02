# -*- coding: utf-8 -*-
"""
IdealQuant - Validation Panel
Monte Carlo, Walk-Forward, Stabilite Analizi
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QTextEdit, QTabWidget, QFormLayout,
    QMessageBox, QComboBox
)
from PySide6.QtCore import Signal, Qt, QThread
from PySide6.QtGui import QFont
import numpy as np
from typing import List, Dict, Any


class MonteCarloWorker(QThread):
    """Monte Carlo simülasyon thread'i"""
    
    progress = Signal(int)
    result = Signal(dict)
    error = Signal(str)
    
    def __init__(self, trades: List[float], n_simulations: int = 1000):
        super().__init__()
        self.trades = trades
        self.n_simulations = n_simulations
    
    def run(self):
        try:
            results = self._run_monte_carlo()
            self.result.emit(results)
        except Exception as e:
            self.error.emit(str(e))
    
    def _run_monte_carlo(self) -> dict:
        """Monte Carlo simülasyonu çalıştır"""
        trades = np.array(self.trades)
        n = len(trades)
        simulations = []
        
        for i in range(self.n_simulations):
            # İşlemleri rastgele sırala
            shuffled = np.random.permutation(trades)
            
            # Equity curve
            equity = np.cumsum(shuffled)
            
            # Max drawdown hesapla
            peak = np.maximum.accumulate(equity)
            dd = peak - equity
            max_dd = np.max(dd)
            
            # Final equity
            final_equity = equity[-1]
            
            simulations.append({
                'final_equity': final_equity,
                'max_dd': max_dd
            })
            
            if i % 100 == 0:
                self.progress.emit(int(i / self.n_simulations * 100))
        
        self.progress.emit(100)
        
        # İstatistikler
        final_equities = [s['final_equity'] for s in simulations]
        max_dds = [s['max_dd'] for s in simulations]
        
        return {
            'n_simulations': self.n_simulations,
            'original_profit': float(np.sum(trades)),
            'mean_profit': float(np.mean(final_equities)),
            'std_profit': float(np.std(final_equities)),
            'min_profit': float(np.min(final_equities)),
            'max_profit': float(np.max(final_equities)),
            'percentile_5': float(np.percentile(final_equities, 5)),
            'percentile_95': float(np.percentile(final_equities, 95)),
            'mean_max_dd': float(np.mean(max_dds)),
            'worst_max_dd': float(np.max(max_dds)),
            'prob_profitable': float(np.mean(np.array(final_equities) > 0) * 100),
        }


class ValidationPanel(QWidget):
    """Validasyon paneli: MC, WFA, Stabilite"""
    
    def __init__(self):
        super().__init__()
        self.optimization_results = []
        self.trades = []
        self._setup_ui()
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Monte Carlo Tab
        mc_tab = self._create_monte_carlo_tab()
        tabs.addTab(mc_tab, "Monte Carlo")
        
        # Walk-Forward Tab
        wfa_tab = self._create_wfa_tab()
        tabs.addTab(wfa_tab, "Walk-Forward")
        
        # Stabilite Tab
        stability_tab = self._create_stability_tab()
        tabs.addTab(stability_tab, "Parametre Stabilitesi")
        
        layout.addWidget(tabs)
    
    def _create_monte_carlo_tab(self) -> QWidget:
        """Monte Carlo tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Ayarlar
        settings_group = QGroupBox("Monte Carlo Ayarları")
        settings_layout = QFormLayout(settings_group)
        
        self.mc_simulations = QSpinBox()
        self.mc_simulations.setRange(100, 10000)
        self.mc_simulations.setValue(1000)
        self.mc_simulations.setSingleStep(100)
        settings_layout.addRow("Simülasyon Sayısı:", self.mc_simulations)
        
        layout.addWidget(settings_group)
        
        # Kontrol
        control_row = QHBoxLayout()
        
        self.mc_progress = QProgressBar()
        control_row.addWidget(self.mc_progress, 1)
        
        self.mc_run_btn = QPushButton("Çalıştır")
        self.mc_run_btn.setObjectName("primaryButton")
        self.mc_run_btn.clicked.connect(self._run_monte_carlo)
        control_row.addWidget(self.mc_run_btn)
        
        layout.addLayout(control_row)
        
        # Sonuçlar
        results_group = QGroupBox("Sonuçlar")
        results_layout = QVBoxLayout(results_group)
        
        self.mc_results_text = QTextEdit()
        self.mc_results_text.setReadOnly(True)
        self.mc_results_text.setFont(QFont("Consolas", 10))
        self.mc_results_text.setPlaceholderText(
            "Monte Carlo simülasyonu, işlemlerinizi rastgele sıralayarak\n"
            "farklı senaryolarda stratejinizin nasıl performans göstereceğini test eder.\n\n"
            "Bu, şansın etkisini ölçmenize yardımcı olur."
        )
        results_layout.addWidget(self.mc_results_text)
        
        layout.addWidget(results_group, 1)
        
        return widget
    
    def _create_wfa_tab(self) -> QWidget:
        """Walk-Forward Analysis tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Ayarlar
        settings_group = QGroupBox("Walk-Forward Ayarları")
        settings_layout = QFormLayout(settings_group)
        
        self.wfa_train_pct = QSpinBox()
        self.wfa_train_pct.setRange(50, 90)
        self.wfa_train_pct.setValue(70)
        self.wfa_train_pct.setSuffix("%")
        settings_layout.addRow("Eğitim Oranı:", self.wfa_train_pct)
        
        self.wfa_windows = QSpinBox()
        self.wfa_windows.setRange(2, 20)
        self.wfa_windows.setValue(5)
        settings_layout.addRow("Pencere Sayısı:", self.wfa_windows)
        
        layout.addWidget(settings_group)
        
        # Açıklama
        info_group = QGroupBox("Walk-Forward Analizi Nedir?")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Walk-Forward Analizi, stratejinizi geçmiş verinin bir kısmında optimize edip\n"
            "(eğitim), sonra kalan kısımda test eder. Bu işlem birden fazla pencerede tekrarlanır.\n\n"
            "Amaç: Stratejinin sadece optimize edildiği döneme değil,\n"
            "görülmemiş verilere de iyi uyum sağlayıp sağlamadığını görmek.\n\n"
            "WFA Efficiency = Test Performansı / Eğitim Performansı\n"
            "İyi bir strateji için bu oran > 0.5 olmalıdır."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # Kontrol
        control_row = QHBoxLayout()
        control_row.addStretch()
        
        wfa_run_btn = QPushButton("Walk-Forward Çalıştır")
        wfa_run_btn.setObjectName("primaryButton")
        wfa_run_btn.clicked.connect(self._run_wfa)
        control_row.addWidget(wfa_run_btn)
        
        layout.addLayout(control_row)
        
        # Sonuç tablosu
        self.wfa_table = QTableWidget()
        self.wfa_table.setColumnCount(5)
        self.wfa_table.setHorizontalHeaderLabels([
            'Pencere', 'Eğitim Kar', 'Test Kar', 'Efficiency', 'Durum'
        ])
        self.wfa_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.wfa_table, 1)
        
        return widget
    
    def _create_stability_tab(self) -> QWidget:
        """Parametre stabilitesi tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Açıklama
        info_group = QGroupBox("Parametre Stabilitesi Nedir?")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "Parametre stabilitesi, optimize edilmiş parametrelerin küçük değişikliklere\n"
            "ne kadar dayanıklı olduğunu ölçer.\n\n"
            "Yüksek stabilite = Parametreyi ±%10 değiştirdiğinizde performans çok değişmiyor\n"
            "Düşük stabilite = Küçük parametre değişiklikleri büyük performans dalgalanmalarına yol açıyor\n\n"
            "Yüksek stabilite skoru olan parametreler daha güvenilirdir."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # Sonuç tablosu
        self.stability_table = QTableWidget()
        self.stability_table.setColumnCount(5)
        self.stability_table.setHorizontalHeaderLabels([
            'Parametre', 'Optimal', '-10%', '+10%', 'Stabilite Skoru'
        ])
        self.stability_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.stability_table, 1)
        
        # Özet
        summary_row = QHBoxLayout()
        self.stability_summary = QLabel("Toplam stabilite skoru hesaplanmadı.")
        summary_row.addWidget(self.stability_summary)
        summary_row.addStretch()
        
        calc_btn = QPushButton("Stabilite Hesapla")
        calc_btn.setObjectName("primaryButton")
        calc_btn.clicked.connect(self._calculate_stability)
        summary_row.addWidget(calc_btn)
        
        layout.addLayout(summary_row)
        
        return widget
    
    def _run_monte_carlo(self):
        """Monte Carlo simülasyonu başlat"""
        if not self.trades:
            QMessageBox.warning(
                self, 
                "Uyarı", 
                "Önce optimizasyon çalıştırın ve sonuçları bu panele aktarın."
            )
            return
        
        self.mc_run_btn.setEnabled(False)
        self.mc_progress.setValue(0)
        
        self.mc_worker = MonteCarloWorker(
            trades=self.trades,
            n_simulations=self.mc_simulations.value()
        )
        self.mc_worker.progress.connect(self.mc_progress.setValue)
        self.mc_worker.result.connect(self._on_mc_result)
        self.mc_worker.error.connect(self._on_mc_error)
        self.mc_worker.finished.connect(lambda: self.mc_run_btn.setEnabled(True))
        self.mc_worker.start()
    
    def _on_mc_result(self, results: dict):
        """Monte Carlo sonuçları"""
        text = f"""
╔══════════════════════════════════════════════════════════╗
║              MONTE CARLO SİMÜLASYON SONUÇLARI            ║
╠══════════════════════════════════════════════════════════╣
║  Simülasyon Sayısı     :  {results['n_simulations']:>10,}                    ║
╠══════════════════════════════════════════════════════════╣
║  KÂRLILIK ANALİZİ                                        ║
╠══════════════════════════════════════════════════════════╣
║  Orijinal Kâr          :  {results['original_profit']:>15,.0f}               ║
║  Ortalama Kâr          :  {results['mean_profit']:>15,.0f}               ║
║  Std Sapma             :  {results['std_profit']:>15,.0f}               ║
║  Min Kâr               :  {results['min_profit']:>15,.0f}               ║
║  Max Kâr               :  {results['max_profit']:>15,.0f}               ║
╠══════════════════════════════════════════════════════════╣
║  GÜVEN ARALIĞI (%90)                                     ║
╠══════════════════════════════════════════════════════════╣
║  %5 Percentile         :  {results['percentile_5']:>15,.0f}               ║
║  %95 Percentile        :  {results['percentile_95']:>15,.0f}               ║
╠══════════════════════════════════════════════════════════╣
║  RİSK METRİKLERİ                                         ║
╠══════════════════════════════════════════════════════════╣
║  Ortalama Max DD       :  {results['mean_max_dd']:>15,.0f}               ║
║  En Kötü Max DD        :  {results['worst_max_dd']:>15,.0f}               ║
║  Kârlı Olma Olasılığı  :  {results['prob_profitable']:>14.1f}%               ║
╚══════════════════════════════════════════════════════════╝

YORUM:
"""
        # Yorum ekle
        if results['prob_profitable'] > 95:
            text += "✓ Strateji çok güçlü görünüyor. Kârlı olma olasılığı çok yüksek."
        elif results['prob_profitable'] > 80:
            text += "✓ Strateji iyi görünüyor ama bazı senaryolarda zarar edebilir."
        elif results['prob_profitable'] > 60:
            text += "⚠ Strateji riskli. Şansa bağlı sonuçlar mümkün."
        else:
            text += "✗ Strateji güvenilir değil. Parametre overfitting olabilir."
        
        self.mc_results_text.setPlainText(text)
    
    def _on_mc_error(self, error_msg: str):
        """Monte Carlo hatası"""
        QMessageBox.critical(self, "Hata", f"Monte Carlo hatası: {error_msg}")
    
    def _run_wfa(self):
        """Walk-Forward analizi"""
        QMessageBox.information(
            self, 
            "Bilgi", 
            "Walk-Forward analizi, optimizasyon sonuçları ile çalışır.\n"
            "Önce Optimizer'dan sonuç alın."
        )
    
    def _calculate_stability(self):
        """Stabilite hesapla"""
        QMessageBox.information(
            self, 
            "Bilgi", 
            "Stabilite analizi, optimizasyon sonuçları ile çalışır.\n"
            "Önce Optimizer'dan sonuç alın."
        )
    
    def set_trades(self, trades: List[float]):
        """İşlem listesini ayarla (Monte Carlo için)"""
        self.trades = trades
    
    def set_optimization_results(self, results: List[dict]):
        """Optimizasyon sonuçlarını ayarla"""
        self.optimization_results = results
