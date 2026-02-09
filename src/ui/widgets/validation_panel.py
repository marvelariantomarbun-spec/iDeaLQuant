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
    QMessageBox, QComboBox, QCheckBox, QAbstractItemView
)
from PySide6.QtCore import Signal, Qt, QThread
from PySide6.QtGui import QFont
import numpy as np
from typing import List, Dict, Any
import pandas as pd
import time

from src.core.database import db
from src.optimization.hybrid_group_optimizer import IndicatorCache, backtest_with_trades, load_data
from src.strategies.score_based import ScoreBasedStrategy
from src.strategies.ars_trend_v2 import ARSTrendStrategyV2
import pandas as pd


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


class WFAWorker(QThread):
    """Walk-Forward Analysis thread'i"""
    progress = Signal(int)
    result = Signal(dict)
    error = Signal(str)
    
    def __init__(self, cache, strategy_idx, params, costs, split_ratio=0.7):
        super().__init__()
        self.cache = cache
        self.strategy_idx = strategy_idx
        self.params = params
        self.costs = costs
        self.split_ratio = split_ratio
        
    def run(self):
        try:
            n = len(self.cache.closes)
            split_idx = int(n * self.split_ratio)
            
            # 1. In-Sample (IS)
            is_closes = self.cache.closes[:split_idx]
            # IS için sinyalleri tekrar üretmek yerine tam listeden bölüyoruz
            # Gerçek WFA'da IS'de optimize edilip OOS'da test edilir. 
            # Burada mevcut parametrelerin iki dönemdeki performansını karşılaştırıyoruz.
            
            if self.strategy_idx == 0:
                strategy = ScoreBasedStrategy.from_config_dict(self.cache, self.params)
            else:
                strategy = ARSTrendStrategyV2.from_config_dict(self.cache, self.params)
                
            signals, ex_long, ex_short = strategy.generate_all_signals()
            
            # IS Backtest
            is_pnl, is_trades, is_pf, is_dd = backtest_with_summary(
                is_closes, signals[:split_idx], ex_long[:split_idx], ex_short[:split_idx],
                self.costs['commission'], self.costs['slippage']
            )
            
            # OOS Backtest
            oos_closes = self.cache.closes[split_idx:]
            oos_pnl, oos_trades, oos_pf, oos_dd = backtest_with_summary(
                oos_closes, signals[split_idx:], ex_long[split_idx:], ex_short[split_idx:],
                self.costs['commission'], self.costs['slippage']
            )
            
            # Efficiency (Annualized Profit Ratio)
            is_days = max(1, split_idx / 500) # Yaklaşık gün (5dk veri farzıyla)
            oos_days = max(1, (n - split_idx) / 500)
            
            is_daily = is_pnl / is_days
            oos_daily = oos_pnl / oos_days
            
            efficiency = (oos_daily / is_daily * 100) if is_daily > 0 else 0
            
            self.result.emit({
                'is_pnl': is_pnl, 'is_trades': is_trades, 'is_pf': is_pf,
                'oos_pnl': oos_pnl, 'oos_trades': oos_trades, 'oos_pf': oos_pf,
                'efficiency': efficiency
            })
            
        except Exception as e:
            self.error.emit(str(e))

def backtest_with_summary(closes, signals, ex_long, ex_short, comm, slip):
    """Basit özet backtest helper"""
    pos, entry_price, gross_profit, gross_loss, trades = 0, 0.0, 0.0, 0.0, 0
    cost = comm + slip
    for i in range(len(closes)):
        if pos == 1 and ex_long[i]:
            pnl = (closes[i] - entry_price) - cost
            if pnl > 0: gross_profit += pnl
            else: gross_loss += abs(pnl)
            pos = 0; trades += 1
        elif pos == -1 and ex_short[i]:
            pnl = (entry_price - closes[i]) - cost
            if pnl > 0: gross_profit += pnl
            else: gross_loss += abs(pnl)
            pos = 0; trades += 1
        if pos == 0:
            if signals[i] == 1: pos = 1; entry_price = closes[i]
            elif signals[i] == -1: pos = -1; entry_price = closes[i]
            
    net = gross_profit - gross_loss
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 9.9
    return net, trades, pf, 0.0


class BatchAnalysisWorker(QThread):
    """Toplu analiz (WFA + Stabilite) thread'i"""
    # Progress: percent, message, elapsed_str, eta_str
    progress = Signal(int, str, str, str)
    result = Signal(int, float, float, float) # opt_id, wfa_score, stability_score, mc_prob
    finished_all = Signal()
    
    def __init__(self, comparison_data: list, process_costs: dict):
        super().__init__()
        self.comparison_data = comparison_data # [{'id': 1, 'params': {}, 'strategy_idx': 0}, ...]
        self.cache = None
        self.process_costs = process_costs
        self.is_running = True
        self.start_time = None
        
    def set_data(self, df):
        self.cache = IndicatorCache(df)
        
    def _format_time(self, seconds: float) -> str:
        """Saniyeyi MM:SS formatına çevir"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def run(self):
        self.start_time = time.time()
        total = len(self.comparison_data)
        
        for i, item in enumerate(self.comparison_data):
            if not self.is_running: break
            
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            
            # ETA hesapla
            eta_str = "--:--"
            if i > 0:
                avg_time = elapsed / i
                remaining_items = total - i
                remaining_time = avg_time * remaining_items
                eta_str = self._format_time(remaining_time)
            
            percent = int((i / total) * 100)
            
            opt_id = item['id']
            params = item['params']
            idx = item['strategy_index']
            
            self.progress.emit(percent, f"Analiz ediliyor: ID {opt_id}...", elapsed_str, eta_str)
            
            # 1. WFA Score (Efficiency)
            # Hızlı WFA: %70 In / %30 Out
            wfa_score = self._calc_wfa(idx, params)
            
            # 2. Stability Score
            # Merkez + 4 komşu
            stab_score = self._calc_stability(idx, params)
            
            # 3. Monte Carlo Score (1000 iterasyon)
            mc_prob = self._calc_mc(idx, params)
            
            self.result.emit(opt_id, wfa_score, stab_score, mc_prob)
            
        self.finished_all.emit()
        
    def _calc_wfa(self, idx, params):
        if not self.cache: return 0
        n = len(self.cache.closes)
        split = int(n * 0.7)
        
        # Strateji
        if idx == 0: s = ScoreBasedStrategy.from_config_dict(self.cache, params)
        else: s = ARSTrendStrategyV2.from_config_dict(self.cache, params)
        
        sig, ex_l, ex_s = s.generate_all_signals()
        
        # IS vs OOS
        is_pnl = backtest_with_summary(self.cache.closes[:split], sig[:split], ex_l[:split], ex_s[:split], 
                                      self.process_costs['commission'], self.process_costs['slippage'])[0]
        oos_pnl = backtest_with_summary(self.cache.closes[split:], sig[split:], ex_l[split:], ex_s[split:],
                                       self.process_costs['commission'], self.process_costs['slippage'])[0]
                                       
        if is_pnl <= 0: return 0
        
        # Time adjusts
        is_days = max(1, split/500)
        oos_days = max(1, (n-split)/500)
        
        eff = ((oos_pnl/oos_days) / (is_pnl/is_days)) * 100
        return max(0, eff)
        
    def _calc_stability(self, idx, params):
        if not self.cache: return 0
        
        base_pnl = self._run_bt(idx, params)
        if base_pnl == 0: return 0
        
        scores = []
        # Random 5 parametreyi %10 oynat
        import random
        keys = list([k for k,v in params.items() if isinstance(v, (int, float))])
        sample_keys = random.sample(keys, min(5, len(keys)))
        
        for k in sample_keys:
            val = params[k]
            # +10%
            p_up = params.copy(); p_up[k] = val * 1.10
            pnl_up = self._run_bt(idx, p_up)
            scores.append(abs(pnl_up - base_pnl) / abs(base_pnl))
            
            # -10%
            p_down = params.copy(); p_down[k] = val * 0.90
            pnl_down = self._run_bt(idx, p_down)
            scores.append(abs(pnl_down - base_pnl) / abs(base_pnl))
            
        avg_dev = sum(scores) / len(scores) if scores else 0
        return max(0, 100 - (avg_dev * 500)) # %10 sapma -> %50 puan kırma
        
    def _run_bt(self, idx, params):
        if idx == 0: s = ScoreBasedStrategy.from_config_dict(self.cache, params)
        else: s = ARSTrendStrategyV2.from_config_dict(self.cache, params)
        sig, ex_l, ex_s = s.generate_all_signals()
        return backtest_with_summary(self.cache.closes, sig, ex_l, ex_s, 
                                   self.process_costs['commission'], self.process_costs['slippage'])[0]

    def _calc_mc(self, idx, params):
        """Monte Carlo simülasyonu (1000 iterasyon)"""
        if not self.cache: return 0
        
        # Orijinal backtest ile işlem listesini al
        if idx == 0: s = ScoreBasedStrategy.from_config_dict(self.cache, params)
        else: s = ARSTrendStrategyV2.from_config_dict(self.cache, params)
        
        # Basit backtest - işlem listesi (pnl'ler) döndüren versiyon lazım
        # fast_backtest trade_pnls listesi döndürüyor, onu kullanalım
        sig, ex_l, ex_s = s.generate_all_signals()
        
        # fast_backtest'i doğrudan çağıralım (indikatör hesaplarını s içinde yaptı)
        # Import fast_backtest from hybrid_group_optimizer for simplicity if needed, 
        # but better use the common backend logic.
        from src.optimization.hybrid_group_optimizer import fast_backtest
        res = fast_backtest(self.cache.closes, sig, ex_l, ex_s, 
                           self.process_costs['commission'], self.process_costs['slippage'])
        # res = (net_profit, trades, pf, max_dd, sharpe) -> fast_backtest modifiye edilmedikçe liste dönmez.
        
        # Gerçek Monte Carlo Simülasyonu
        # 1. İşlem listesini (PnL) al
        trades_pnl = []
        
        # fast_backtest'i modifiye etmeden pnl listesini almak zor.
        # Bu yüzden basit bir döngü ile işlem sonuçlarını çıkaralım.
        # fast_backtest yerine backtest_with_trades yardımcı fonksiyonunu kullanalım (eğer varsa)
        # Yoksa manuel hesaplayalım:
        
        entry_price = 0.0
        pos = 0
        cost = self.process_costs['commission'] + self.process_costs['slippage']
        
        close_arr = self.cache.closes
        
        for i in range(len(close_arr)):
            # Long Exit
            if pos == 1 and ex_l[i]:
                pnl = (close_arr[i] - entry_price) - cost
                trades_pnl.append(pnl)
                pos = 0
            # Short Exit
            elif pos == -1 and ex_s[i]:
                pnl = (entry_price - close_arr[i]) - cost
                trades_pnl.append(pnl)
                pos = 0
            
            # Entry
            if pos == 0:
                if sig[i] == 1:
                    pos = 1
                    entry_price = close_arr[i]
                elif sig[i] == -1:
                    pos = -1
                    entry_price = close_arr[i]
        
        if not trades_pnl:
            return 0.0
            
        # 2. Simülasyon (1000 iterasyon)
        import random
        profitable_runs = 0
        n_sim = 1000
        initial_capital = 10000.0 # Varsayılan
        
        for _ in range(n_sim):
            # İşlemleri karıştır
            shuffled_trades = random.sample(trades_pnl, len(trades_pnl))
            
            # Kümülatif kar
            net_profit = sum(shuffled_trades)
            
            # Max DD kontrolü (Opsiyonel: Ruin olasılığı için)
            # equity = np.cumsum(shuffled_trades)
            # if np.min(equity) < -initial_capital: continue
            
            if net_profit > 0:
                profitable_runs += 1
                
        mc_prob = (profitable_runs / n_sim) * 100
        return mc_prob


class ValidationPanel(QWidget):
    """Validasyon paneli: MC, WFA, Stabilite"""
    
    validation_complete = Signal(str, dict)  # process_id, final_params
    
    def __init__(self):
        super().__init__()
        self.optimization_results = []
        self.trades = []
        self.current_process_id = None
        
        # Timer için
        from PySide6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_batch_timer)
        
        self._setup_ui()
    
    def _update_batch_timer(self):
        if not hasattr(self, 'start_time'): return
        
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        percent = getattr(self, 'current_percent', 0)
        eta = getattr(self, 'current_eta', '--:--')
        
        self.batch_status_label.setText(f"%{percent} Tamamlandı - Geçen: {elapsed_str} - Kalan: {eta}")

    def _format_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Süreç seçimi
        process_row = QHBoxLayout()
        process_row.addWidget(QLabel("Süreç:"))
        self.process_combo = QComboBox()
        self.process_combo.setMinimumWidth(250)
        self.process_combo.currentTextChanged.connect(self._on_process_changed)
        process_row.addWidget(self.process_combo)
        
        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.clicked.connect(self._refresh_processes)
        process_row.addWidget(refresh_btn)
        
        process_row.addStretch()
        layout.addLayout(process_row)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Karşılaştırma Tab (YENİ)
        compare_tab = self._create_comparison_tab()
        self.tabs.addTab(compare_tab, "Karşılaştırma")
        
        # Monte Carlo Tab
        mc_tab = self._create_monte_carlo_tab()
        self.tabs.addTab(mc_tab, "Monte Carlo")
        
        # Walk-Forward Tab
        wfa_tab = self._create_wfa_tab()
        self.tabs.addTab(wfa_tab, "Walk-Forward")
        
        # Stabilite Tab
        stability_tab = self._create_stability_tab()
        self.tabs.addTab(stability_tab, "Parametre Stabilitesi")
        
        layout.addWidget(self.tabs)
    
    def _create_comparison_tab(self) -> QWidget:
        """Optimizasyon sonuçları karşılaştırma tab'ı"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Açıklama
        info = QLabel(
            "Bu tablo, seçili süreç için tüm optimizasyon sonuçlarını gösterir.\n"
            "Her strateji için en iyi sonucu 'Final' olarak seçin."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Karşılaştırma tablosu
        self.compare_table = QTableWidget()
        self.compare_table.setColumnCount(13)
        self.compare_table.setHorizontalHeaderLabels([
            '✓', 'Strateji', 'Metod', 'Net Kar', 'Max DD', 'PF', 'Trade', 'Sharpe', 'Fitness', 'WFA', 'Stabil', 'MC %', 'Final'
        ])
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.compare_table.setColumnWidth(0, 30) # Checkbox kolonu dar olsun
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.compare_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.compare_table.cellClicked.connect(self._on_table_row_clicked)
        layout.addWidget(self.compare_table, 1)
        
        # Butonlar
        btn_row = QHBoxLayout()
        
        self.batch_analyze_btn = QPushButton("Seçilenleri Analiz Et (WFA + Stabilite)")
        self.batch_analyze_btn.setObjectName("primaryButton")
        self.batch_analyze_btn.clicked.connect(self._run_batch_analysis)
        btn_row.addWidget(self.batch_analyze_btn)
        
        btn_row.addStretch()
        
        # Batch Progress & Status
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setFixedHeight(15)
        layout.addWidget(self.batch_progress_bar)
        
        self.batch_status_label = QLabel("")
        self.batch_status_label.setAlignment(Qt.AlignCenter)
        self.batch_status_label.setStyleSheet("font-weight: bold; color: #E91E63; font-size: 13px;")
        self.batch_status_label.setVisible(False)
        layout.addWidget(self.batch_status_label)
        
        self.set_final_btn = QPushButton("Seçileni Final Yap")
        self.set_final_btn.setObjectName("primaryButton")
        self.set_final_btn.clicked.connect(self._set_as_final)
        btn_row.addWidget(self.set_final_btn)
        
        layout.addLayout(btn_row)
        
        return widget
    
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
        
        self.mc_run_btn = QPushButton("Monte Carlo Çalıştır")
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
        
        self.wfa_run_btn = QPushButton("Walk-Forward Çalıştır")
        self.wfa_run_btn.setObjectName("primaryButton")
        self.wfa_run_btn.clicked.connect(self._run_wfa)
        control_row.addWidget(self.wfa_run_btn)
        
        layout.addLayout(control_row)
        
        # Sonuçlar
        self.wfa_results_text = QTextEdit()
        self.wfa_results_text.setReadOnly(True)
        self.wfa_results_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.wfa_results_text, 1)
        
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
        
        # Kontrol
        control_row = QHBoxLayout()
        control_row.addStretch()
        
        self.stabil_run_btn = QPushButton("Analizi Başlat")
        self.stabil_run_btn.setObjectName("primaryButton")
        self.stabil_run_btn.clicked.connect(self._calculate_stability)
        control_row.addWidget(self.stabil_run_btn)
        
        layout.addLayout(control_row)
        
        # İlerleme
        self.stability_progress = QProgressBar()
        layout.addWidget(self.stability_progress)
        
        # Sonuçlar
        self.stability_results_text = QTextEdit()
        self.stability_results_text.setReadOnly(True)
        self.stability_results_text.setFont(QFont("Consolas", 11))
        layout.addWidget(self.stability_results_text, 1)
        
        return widget
        
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
        """Walk-Forward analizi başlat"""
        if not hasattr(self, 'trades') or not self.trades:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce Karşılaştırma tablosundan bir sonuç seçin.")
            return
            
        # Mevcut verileri al
        opt_id = self._selected_opt_id
        opt_result = db.get_optimization_result_by_id(opt_id)
        costs = db.get_process_costs(opt_result['process_id'])
        
        # Cache'i yeniden oluştur (Helper method yapalım)
        df = self._load_data_for_process(opt_result['process_id'])
        cache = IndicatorCache(df)
        
        self.wfa_run_btn.setEnabled(False)
        self.wfa_worker = WFAWorker(cache, opt_result['strategy_index'], opt_result['params'], costs)
        self.wfa_worker.result.connect(self._on_wfa_result)
        self.wfa_worker.error.connect(lambda e: QMessageBox.critical(self, "Hata", e))
        self.wfa_worker.finished.connect(lambda: self.wfa_run_btn.setEnabled(True))
        self.wfa_worker.start()

    def _on_wfa_result(self, res: dict):
        text = f"""
╔══════════════════════════════════════════════════════════╗
║              WALK-FORWARD ANALİZ SONUÇLARI               ║
╠══════════════════════════════════════════════════════════╣
║ METRİK             ║ IN-SAMPLE (70%)  ║ OUT-OF-SAMPLE (30%) ║
╠══════════════════════════════════════════════════════════╣
║ Net Kâr            ║ {res['is_pnl']:>15,.0f}  ║ {res['oos_pnl']:>17,.0f}  ║
║ İşlem Sayısı       ║ {res['is_trades']:>15,}  ║ {res['oos_trades']:>17,}  ║
║ Profit Factor      ║ {res['is_pf']:>15.2f}  ║ {res['oos_pf']:>17.2f}  ║
╠══════════════════════════════════════════════════════════╣
║ WFA VERİMLİLİĞİ    : %{res['efficiency']:>10.1f}                        ║
╚══════════════════════════════════════════════════════════╝

YORUM:
"""
        if res['efficiency'] > 80:
            text += "✓ Mükemmel! Strateji iki dönemde de benzer verimlilikte."
        elif res['efficiency'] > 50:
            text += "✓ İyi. Strateji dış veride kârlılığını koruyor."
        else:
            text += "⚠ Dikkat. Strateji dış veride ciddi performans kaybetti (Overfitting riski)."
            
        self.wfa_results_text.setPlainText(text)

    def _load_data_for_process(self, process_id):
        from src.data.ideal_parser import read_ideal_data
        from pathlib import Path
        import os
        
        # Süreç bilgilerini al
        proc = db.get_process(process_id)
        period_str = proc['period'].replace('dk', '')
        symbol = proc['symbol'].split('_')[-1] # Market_Symbol -> Symbol
        
        data_file = proc['data_file']
        
        # 1. Dosya var mı kontrol et
        found_file = self._find_data_file(data_file)
        
        # 2. Dosya yoksa Resample yapmayı dene (1dk veya 5dk'dan)
        if not found_file:
            print(f"Dosya bulunamadı: {data_file}. Resampling denenecek...")
            base_periods = ['1', '5'] # Öncelik sırasi
            
            for base_p in base_periods:
                # Hedef periyot base'den büyük olmalı
                if int(period_str) <= int(base_p):
                    continue
                    
                # Base dosya yolunu tahmin et
                from src.data.ideal_parser import PERIOD_MAP
                base_info = PERIOD_MAP.get(base_p)
                
                # Orijinal yolun yapısını korumaya çalış
                path_obj = Path(data_file)
                # D:\iDeal\ChartData\VIP\01\X030-T.01
                try:
                    market_dir = path_obj.parent.parent
                    base_dir = market_dir / base_info['folder']
                    base_ext = base_info['ext']
                    
                    # Sembol ismini koru (X030-T)
                    stem = path_obj.stem # X030-T
                    base_filename = f"{stem}{base_ext}"
                    base_path_guess = base_dir / base_filename
                    
                    found_base = self._find_data_file(str(base_path_guess))
                    
                    if found_base:
                        print(f"Base data bulundu: {found_base}. {period_str}dk'ya çevriliyor...")
                        df = read_ideal_data(found_base)
                        
                        # Resample
                        df.set_index('DateTime', inplace=True)
                        rule = f"{period_str}T"
                        
                        ohlc_dict = {
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum',
                            'Amount': 'sum',
                            'Lot': 'sum'
                        }
                        
                        df_res = df.resample(rule).agg(ohlc_dict).dropna()
                        df_res.reset_index(inplace=True)
                        
                        # Tipik fiyat
                        df_res['Tipik'] = (df_res['High'] + df_res['Low'] + df_res['Close']) / 3
                        return df_res
                        
                except Exception as e:
                    print(f"Resampling hatası ({base_p}dk): {e}")
                    continue
        
        # Dosya veya Base Dosya bulunamadıysa normal akış
        if found_file:
            data_file = found_file

        try:
            # Binary oku
            df = read_ideal_data(data_file)
            # Tipik fiyat hesapla
            df['Tipik'] = (df['High'] + df['Low'] + df['Close']) / 3
            return df
        except Exception as e:
            # Fallback for old CSV?
            try:
                df = pd.read_csv(data_file, sep=';', decimal=',', encoding='cp1254', header=None, low_memory=False)
                df.columns = ['Tarih', 'Saat', 'Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Ortalama', 'Hacim', 'Lot']
                for c in ['Acilis', 'Yuksek', 'Dusuk', 'Kapanis', 'Hacim', 'Lot']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df.dropna(inplace=True)
                df['Tipik'] = (df['Yuksek'] + df['Dusuk'] + df['Kapanis']) / 3
                df['DateTime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
                return df.dropna(subset=['DateTime']).reset_index(drop=True)
            except:
                raise e

    def _find_data_file(self, file_path: str) -> str:
        """Dosyayı bulmaya çalış (VIP' prefixi vb. için)"""
        import os
        from pathlib import Path
        
        if os.path.exists(file_path):
            return file_path
            
        path_obj = Path(file_path)
        directory = path_obj.parent
        filename = path_obj.name
        
        if directory.exists():
            for f in directory.iterdir():
                # Tam eşleşme veya son ek eşleşmesi (VIP'X30 vs X30)
                if f.name == filename or f.name.endswith(filename) or filename in f.name:
                    return str(f)
        
        return None

    def _calculate_stability(self):
        """Stabilite puanı hesapla"""
        if not hasattr(self, 'trades') or not self.trades:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sonuç seçin.")
            return
            
        opt_id = self._selected_opt_id
        opt_result = db.get_optimization_result_by_id(opt_id)
        params = opt_result['params']
        costs = db.get_process_costs(opt_result['process_id'])
        df = self._load_data_for_process(opt_result['process_id'])
        cache = IndicatorCache(df)
        
        # Sinyalleri üret
        def get_profit(p):
            if opt_result['strategy_index'] == 0:
                s = ScoreBasedStrategy.from_config_dict(cache, p)
            else:
                s = ARSTrendStrategyV2.from_config_dict(cache, p)
            sig, ex_l, ex_s = s.generate_all_signals()
            return backtest_with_summary(cache.closes, sig, ex_l, ex_s, costs['commission'], costs['slippage'])[0]
            
        base_profit = get_profit(params)
        variations = []
        
        # Sayısal parametreleri %5 oynat
        for k, v in params.items():
            if isinstance(v, (int, float)) and "period" not in k: # Periyotlar tam sayı olmalı
                test_params = params.copy()
                test_params[k] = v * 1.05
                variations.append(get_profit(test_params))
                test_params[k] = v * 0.95
                variations.append(get_profit(test_params))
                
        if not variations:
            stability = 100
        else:
            diffs = [abs((v - base_profit) / base_profit) if base_profit != 0 else 0 for v in variations]
            avg_diff = np.mean(diffs)
            stability = max(0, 100 - (avg_diff * 200)) # %10 değişim -> %20 puan kaybı
            
        self.stability_progress.setValue(int(stability))
        self.stability_results_text.setPlainText(f"Parametre Stabilite Skoru: {stability:.1f}\n\n100 üzerinden değerlendirme.\nYüksek puan, parametrelerdeki ufak değişimlerin kârı bozmadığını gösterir.")
    
    def set_trades(self, trades: List[float]):
        """İşlem listesini ayarla (Monte Carlo için)"""
        self.trades = trades
    
    def set_optimization_results(self, results: List[dict]):
        """Optimizasyon sonuçlarını ayarla (eski uyumluluk için)"""
        self.optimization_results = results
    
    # =========================================================================
    # SÜREÇ YÖNETİMİ
    # =========================================================================
    
    def _refresh_processes(self):
        """Süreç listesini yenile (sadece optimize edilmişler)"""
        self.process_combo.clear()
        processes = db.get_all_processes()
        
        # Sadece optimized veya validated süreçleri göster
        valid_processes = [p for p in processes if p['status'] in ('optimized', 'validated', 'exported')]
        
        if not valid_processes:
            self.process_combo.addItem("(Optimizasyon yapılmış süreç yok)")
            return
        
        for proc in valid_processes:
            status_icon = "✓" if proc['status'] == 'validated' else "○"
            display = f"{status_icon} {proc['process_id']}"
            self.process_combo.addItem(display, proc['process_id'])
        
        # İlkini seç
        if valid_processes:
            self.current_process_id = valid_processes[0]['process_id']
            self._load_comparison_data()
    
    def _on_process_changed(self, text: str):
        """Süreç seçimi değiştiğinde"""
        idx = self.process_combo.currentIndex()
        if idx >= 0:
            self.current_process_id = self.process_combo.itemData(idx)
            self._load_comparison_data()
    
    def _load_comparison_data(self):
        """Karşılaştırma tablosunu DB'den doldur"""
        if not self.current_process_id:
            return
        
        results = db.get_optimization_results(self.current_process_id)
        
        if not results:
            self.compare_table.setRowCount(0)
            return
        
        self.compare_table.setRowCount(len(results))
        
        for row_idx, result in enumerate(results):
            # 0. Checkbox
            chk_widget = QWidget()
            chk_layout = QHBoxLayout()
            chk_layout.setContentsMargins(0,0,0,0)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk = QCheckBox()
            chk.setProperty("opt_id", result['id'])
            chk_layout.addWidget(chk)
            chk_widget.setLayout(chk_layout)
            self.compare_table.setCellWidget(row_idx, 0, chk_widget)
            
            # 1. Strateji
            strategy_name = "S1" if result['strategy_index'] == 0 else "S2"
            self.compare_table.setItem(row_idx, 1, QTableWidgetItem(strategy_name))
            
            # 2. Metod
            self.compare_table.setItem(row_idx, 2, QTableWidgetItem(result['method'].capitalize()))
            
            # 3. Net Kar
            self.compare_table.setItem(row_idx, 3, QTableWidgetItem(f"{result['net_profit']:,.0f}"))
            
            # 4. Max DD
            self.compare_table.setItem(row_idx, 4, QTableWidgetItem(f"{result['max_drawdown']:,.0f}"))
            
            # 5. PF
            self.compare_table.setItem(row_idx, 5, QTableWidgetItem(f"{result['profit_factor']:.2f}"))
            
            # 6. Trade
            self.compare_table.setItem(row_idx, 6, QTableWidgetItem(str(result.get('total_trades', 0))))
            
            # 7. Sharpe
            self.compare_table.setItem(row_idx, 7, QTableWidgetItem(f"{result.get('sharpe', 0):.2f}"))
            
            # 8. Fitness
            fitness = result.get('fitness', 0)
            fit_item = QTableWidgetItem(f"{fitness:,.0f}")
            if fitness > 0:
                fit_item.setForeground(Qt.darkGreen)
            else:
                fit_item.setForeground(Qt.red)
            self.compare_table.setItem(row_idx, 8, fit_item)
            
            # 9. WFA (Boş)
            self.compare_table.setItem(row_idx, 9, QTableWidgetItem("-"))
            
            # 10. Stabil (Boş)
            self.compare_table.setItem(row_idx, 10, QTableWidgetItem("-"))
            
            # 11. MC % (Boş)
            self.compare_table.setItem(row_idx, 11, QTableWidgetItem("-"))
            
            # 12. Final durumu
            validations = db.get_validation_results(self.current_process_id)
            is_final = any(v['optimization_id'] == result['id'] and v['is_final'] for v in validations)
            final_text = "✓ Final" if is_final else ""
            self.compare_table.setItem(row_idx, 12, QTableWidgetItem(final_text))
            
    def _on_table_row_clicked(self, row, col):
        """Satıra tıklandığında detayları güncelle"""
        # Checkbox kolonuna (0) tıklanırsa işlem yapma
        if col == 0: return
        
        # Opt ID'yi bul (Checkbox widget'ından alabiliriz veya row index'ten db listesine gidebiliriz)
        # En güvenlisi checkbox widget'ındaki property
        widget = self.compare_table.cellWidget(row, 0)
        if widget:
            # Widget yapısı: QWidget -> Layout -> Checkbox
            # Layout içindeki ilk item'ı al
            layout = widget.layout()
            if layout and layout.count() > 0:
                chk = layout.itemAt(0).widget()
                opt_id = chk.property("opt_id")
                self._select_for_final(opt_id)

    def _run_batch_analysis(self):
        """Seçili satırlar için toplu analiz başlat"""
        print("DEBUG: _run_batch_analysis called!")  # Debug
        selected_ids = []
        for row in range(self.compare_table.rowCount()):
            widget = self.compare_table.cellWidget(row, 0)
            if widget:
                layout = widget.layout()
                chk = layout.itemAt(0).widget()
                if chk.isChecked():
                    selected_ids.append(chk.property("opt_id"))
                    self.compare_table.setItem(row, 9, QTableWidgetItem("...")) # WFA loading
                    self.compare_table.setItem(row, 10, QTableWidgetItem("...")) # Stabil loading
                    
        if not selected_ids:
            QMessageBox.warning(self, "Uyarı", "Lütfen en az bir strateji seçin.")
            return

        # Verileri hazırla
        data_to_process = []
        for oid in selected_ids:
            res = db.get_optimization_result_by_id(oid)
            data_to_process.append(res)
            
        costs = db.get_process_costs(self.current_process_id)
        df = self._load_data_for_process(self.current_process_id)
        
        # Worker başlat
        self.batch_worker = BatchAnalysisWorker(data_to_process, costs)
        self.batch_worker.set_data(df)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.result.connect(self._on_batch_result)
        self.batch_worker.finished_all.connect(self._on_batch_finished)
        
        self.batch_analyze_btn.setEnabled(False)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)
        self.batch_status_label.setVisible(True)
        self.batch_status_label.setText("Analiz başlatılıyor...")
        
        self.start_time = time.time()
        self.timer.start(1000)
        
        self.batch_worker.start()
        
    def _on_batch_progress(self, percent: int, message: str, elapsed: str, eta: str):
        """İlerleme durumunu güncelle"""
        self.current_percent = percent
        self.current_eta = eta
        self.batch_progress_bar.setValue(percent)
        self._update_batch_timer()
        
    def _on_batch_result(self, opt_id, wfa_score, stab_score, mc_prob):
        """Tablodaki satırı güncelle"""
        for row in range(self.compare_table.rowCount()):
            widget = self.compare_table.cellWidget(row, 0)
            if widget:
                layout = widget.layout()
                chk = layout.itemAt(0).widget()
                if chk.property("opt_id") == opt_id:
                    # WFA
                    wfa_item = QTableWidgetItem(f"%{wfa_score:.0f}")
                    if wfa_score >= 80: wfa_item.setForeground(Qt.darkGreen)
                    elif wfa_score < 50: wfa_item.setForeground(Qt.red)
                    self.compare_table.setItem(row, 9, wfa_item)
                    
                    # Stabilite
                    stab_item = QTableWidgetItem(f"{stab_score:.0f}")
                    if stab_score >= 80: stab_item.setForeground(Qt.darkGreen)
                    elif stab_score < 50: stab_item.setForeground(Qt.red)
                    self.compare_table.setItem(row, 10, stab_item)
                    
                    # Monte Carlo
                    mc_item = QTableWidgetItem(f"%{mc_prob:.0f}")
                    if mc_prob >= 90: mc_item.setForeground(Qt.darkGreen)
                    elif mc_prob < 70: mc_item.setForeground(Qt.red)
                    self.compare_table.setItem(row, 11, mc_item)
                    break
                    
    def _on_batch_finished(self):
        self.timer.stop()
        final_time = self._format_time(time.time() - self.start_time)
        
        self.batch_progress_bar.setVisible(False)
        self.batch_status_label.setText(f"Toplu Analiz Tamamlandı (Süre: {final_time})")
        self.batch_analyze_btn.setText("Seçilenleri Analiz Et")
        self.batch_analyze_btn.setEnabled(True)
        QMessageBox.information(self, "Tamamlandı", f"Toplu analiz tamamlandı.\nToplam Süre: {final_time}")
    
    def _select_for_final(self, optimization_id: int):
        """Seçili satırı işaretle ve verilerini yükle"""
        self._selected_opt_id = optimization_id
        
        # Verileri arka planda hesapla
        self._calculate_trades_for_selected(optimization_id)

    def _calculate_trades_for_selected(self, optimization_id: int):
        """Seçilen optimizasyon için işlemleri hesapla (MC için)"""
        try:
            # 1. Optimizasyon sonucunu al
            opt_result = db.get_optimization_result_by_id(optimization_id)
            if not opt_result: return
            
            process_id = opt_result['process_id']
            strategy_index = opt_result['strategy_index']
            params = opt_result['params']
            
            # 2. Maliyetleri ve Veri Dosyasını al
            costs = db.get_process_costs(process_id)
            data_file = db.get_process_data_file(process_id)
            
            if not data_file:
                QMessageBox.warning(self, "Hata", "Veri dosyası bulunamadı.")
                return

            # 3. Veriyi yükle
            df = self._load_data_for_process(process_id)
            
            if df is None or len(df) == 0:
                QMessageBox.critical(self, "Hata", "Veri yüklenemedi.")
                return
            
            cache = IndicatorCache(df)
            
            # 4. Strateji çalıştır
            if strategy_index == 0:
                strategy = ScoreBasedStrategy.from_config_dict(cache, params)
            else:
                strategy = ARSTrendStrategyV2.from_config_dict(cache, params)
                
            signals, ex_long, ex_short = strategy.generate_all_signals()
            trades = backtest_with_trades(
                cache.closes, signals, ex_long, ex_short,
                commission=costs['commission'],
                slippage=costs['slippage']
            )
            
            self.trades = trades
            self.optimization_results = [opt_result] # MC ve WFA için mevcut sonucu set et
            
            # Bilgi ver
            valid_trades_count = len(trades)
            self.tabs.setTabText(1, f"Monte Carlo ({valid_trades_count} İşlem)")
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri hesaplama hatası: {str(e)}")
    
    def _set_as_final(self):
        """Seçili optimizasyonu final olarak işaretle"""
        if not hasattr(self, '_selected_opt_id') or not self._selected_opt_id:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir satır seçin.")
            return
        
        # Final olarak kaydet
        opt_result = db.get_optimization_result_by_id(self._selected_opt_id)
        if opt_result:
            db.set_final_selection(self._selected_opt_id, opt_result['params'])
            
            # Süreç durumunu güncelle
            db.update_process_status(self.current_process_id, 'validated')
            
            QMessageBox.information(
                self, 
                "Başarılı", 
                f"Strateji {opt_result['strategy_index'] + 1} / {opt_result['method']} final olarak seçildi."
            )
            
            # Tabloyu yenile
            self._load_comparison_data()
            
            # Signal gönder
            self.validation_complete.emit(self.current_process_id, opt_result['params'])
    
    def set_data(self, df: pd.DataFrame):
        """Dışarıdan veri setini güncelle (DataPanel'den gelen filtrelenmiş veri)"""
        self.df = df
        if df is not None:
            # Stats label'ı güncelle (eğer varsa veya log bas)
            print(f"[VAL] Yeni veri seti alindi: {len(df)} bar")

    def set_process(self, process_id: str):
        """Dışarıdan süreç ayarla"""
        self.current_process_id = process_id
        self._refresh_processes()
        
        # Combo'da ilgili süreci seç
        for i in range(self.process_combo.count()):
            if self.process_combo.itemData(i) == process_id:
                self.process_combo.setCurrentIndex(i)
                break

