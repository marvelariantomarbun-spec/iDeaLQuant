# -*- coding: utf-8 -*-
"""
Canlı İzleme Ekranı (Live Monitor Window)
Optimizasyon sırasında açılan ayrı pencere.
Dinamik parametre sütunları + sıralanabilir tablo + dark theme.
"""
import time
from PySide6.QtWidgets import (
    QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHBoxLayout, QWidget, QLabel, QHeaderView, QStatusBar,
    QAbstractItemView, QPushButton, QComboBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QBrush

# --- Metrik sütun tanımları (sabit) ---
METRIC_COLUMNS = [
    ('_rank',      'Sıra',     60),
    ('net_profit', 'Net Kar',   100),
    ('pf',         'PF',        65),
    ('max_dd',     'Max DD',    90),
    ('trades',     'İşlem',     65),
    ('sharpe',     'Sharpe',    70),
    ('fitness',    'Fitness',   90),
    ('test_net',   'Test Kar',  90),
    ('test_pf',    'Test PF',   70),
]

# Parametre grup tanımlarını dışarıdan import et
from src.ui.widgets.optimizer_panel import (
    STRATEGY1_PARAM_GROUPS,
    STRATEGY2_PARAM_GROUPS,
    STRATEGY3_PARAM_GROUPS,
    STRATEGY4_PARAM_GROUPS,
)

STRATEGY_PARAM_GROUPS = [
    STRATEGY1_PARAM_GROUPS,
    STRATEGY2_PARAM_GROUPS,
    STRATEGY3_PARAM_GROUPS,
    STRATEGY4_PARAM_GROUPS,
]

STRATEGY_NAMES = [
    "Strateji 1 — Gatekeeper",
    "Strateji 2 — ARS Trend v2",
    "Strateji 3 — Paradise",
    "Strateji 4 — TOMA + Momentum",
]


class NumericTableItem(QTableWidgetItem):
    """Sayısal sıralama için özel QTableWidgetItem"""
    def __init__(self, display_text: str, sort_value: float):
        super().__init__(display_text)
        self._sort_value = sort_value

    def __lt__(self, other):
        if isinstance(other, NumericTableItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class LiveMonitorWindow(QMainWindow):
    """Optimizasyon sırasında canlı sonuç izleme penceresi"""

    def __init__(self, strategy_index: int = 0, method: str = "Hibrit Grup", parent=None):
        super().__init__(parent)
        self.strategy_index = strategy_index
        self.method = method
        self._all_results = []           # Tüm gelen sonuçlar (birleşik)
        self._param_columns = []         # [(key, label), ...]
        self._is_final = False
        self._update_count = 0

        self.setWindowTitle("IdealQuant — Canlı İzleme")
        self.setMinimumSize(1200, 700)
        self._apply_dark_theme()
        self._build_ui()
        self._build_columns()

    # ==================================================================
    # THEME
    # ==================================================================
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QTableWidget {
                background: #16213e;
                color: #e0e0e0;
                gridline-color: #2a2a4a;
                border: none;
                font-size: 12px;
                alternate-background-color: #1a1a3e;
            }
            QTableWidget::item:selected {
                background: #0f3460;
                color: #fff;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2c2c54, stop:1 #1e1e3f);
                color: #ff9800;
                border: 1px solid #333;
                padding: 4px 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QLabel { color: #ccc; }
            QStatusBar { background: #0d0d1a; color: #aaa; font-size: 11px; }
            QPushButton {
                background: #2c2c54; color: #ff9800; border: 1px solid #444;
                padding: 4px 12px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background: #3a3a6a; }
        """)

    # ==================================================================
    # UI BUILD
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 2)
        main_layout.setSpacing(4)

        # --- Header bar ---
        header = QHBoxLayout()

        self.title_label = QLabel(f"⭐ {STRATEGY_NAMES[self.strategy_index]} — {self.method}")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #ff9800;")
        header.addWidget(self.title_label)

        header.addStretch()

        self.count_label = QLabel("Sonuç: 0")
        self.count_label.setStyleSheet("color: #aaa; font-size: 12px;")
        header.addWidget(self.count_label)

        self.best_label = QLabel("")
        self.best_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")
        header.addWidget(self.best_label)

        self.status_indicator = QLabel("⏳ Bekleniyor...")
        self.status_indicator.setStyleSheet("color: #ff9800; font-size: 12px;")
        header.addWidget(self.status_indicator)

        main_layout.addLayout(header)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.setSortingEnabled(True)
        # İlk tıklamada descending olsun
        self.table.horizontalHeader().setSortIndicator(-1, Qt.DescendingOrder)

        main_layout.addWidget(self.table)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Optimizasyon bekleniyor...")

    # ==================================================================
    # COLUMN MANAGEMENT
    # ==================================================================
    def _build_param_columns(self, strategy_index: int):
        """Stratejiye göre parametre sütunlarını oluştur"""
        groups = STRATEGY_PARAM_GROUPS[strategy_index]
        columns = []
        for group_id, group_info in groups.items():
            for p_name, p_info in group_info.get('params', {}).items():
                columns.append((p_name, p_info.get('label', p_name)))
        return columns

    def _build_columns(self):
        """Metrik + parametre sütunlarını tabloya uygula"""
        self._param_columns = self._build_param_columns(self.strategy_index)

        all_cols = [col[2] if len(col) > 2 else col[1] for col in METRIC_COLUMNS]
        headers = [col[1] for col in METRIC_COLUMNS] + [pc[1] for pc in self._param_columns]
        widths = [col[2] for col in METRIC_COLUMNS] + [75] * len(self._param_columns)

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        for i, w in enumerate(widths):
            self.table.setColumnWidth(i, w)

    def set_strategy(self, strategy_index: int, method: str = None):
        """Strateji değişince sütunları yeniden oluştur"""
        self.strategy_index = strategy_index
        if method:
            self.method = method
        self.title_label.setText(f"⭐ {STRATEGY_NAMES[self.strategy_index]} — {self.method}")
        self._all_results.clear()
        self._build_columns()
        self.table.setRowCount(0)
        self._is_final = False

    # ==================================================================
    # DATA UPDATE
    # ==================================================================
    def update_results(self, results: list, method: str = None, is_final: bool = False):
        """Sonuçları tabloya yaz (partial veya final)"""
        if not results:
            return

        if method:
            self.method = method
            self.title_label.setText(f"⭐ {STRATEGY_NAMES[self.strategy_index]} — {self.method}")

        self._is_final = is_final
        self._update_count += 1

        # Partial ise merge et, final ise üzerine yaz
        if is_final:
            self._all_results = list(results)
        else:
            # Partial: Yeni sonuçları ekle (basit merge — duplicate'ler olabilir)
            self._all_results = list(results)

        self._populate_table()

        # Status güncelle
        n = len(self._all_results)
        best_net = max((r.get('net_profit', 0) for r in self._all_results), default=0)
        self.count_label.setText(f"Sonuç: {n:,}")
        if best_net > 0:
            self.best_label.setText(f"En İyi: {best_net:,.0f}")
        
        if is_final:
            self.status_indicator.setText("✅ Tamamlandı")
            self.status_indicator.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")
            self.status_bar.showMessage(f"{self.method} tamamlandı — {n:,} sonuç")
        else:
            dots = "." * ((self._update_count % 3) + 1)
            self.status_indicator.setText(f"⏳ Canlı{dots}")
            self.status_indicator.setStyleSheet("color: #ff9800; font-size: 12px;")
            self.status_bar.showMessage(f"Güncelleme #{self._update_count} — {n:,} sonuç")

    def _populate_table(self):
        """Tablo satırlarını doldur"""
        self.table.setSortingEnabled(False)  # Populate sırasında disable et
        
        results = self._all_results
        n_metric = len(METRIC_COLUMNS)
        n_param = len(self._param_columns)
        total_cols = n_metric + n_param

        self.table.setRowCount(len(results))

        for row_idx, result in enumerate(results):
            params = result.get('params', result)  # S1/S2/S3 nested, S4 flat

            # --- Metrik sütunlar ---
            # Sıra
            self.table.setItem(row_idx, 0, NumericTableItem(str(row_idx + 1), float(row_idx + 1)))

            # Net Kar
            net = result.get('net_profit', 0)
            item = NumericTableItem(f"{net:,.0f}", float(net))
            item.setForeground(QColor("#4CAF50") if net > 0 else QColor("#f44336"))
            self.table.setItem(row_idx, 1, item)

            # PF
            pf = result.get('pf', 0)
            item = NumericTableItem(f"{pf:.2f}", float(pf))
            if pf >= 2.0:
                item.setForeground(QColor("#4CAF50"))
            elif pf >= 1.5:
                item.setForeground(QColor("#8BC34A"))
            self.table.setItem(row_idx, 2, item)

            # Max DD
            dd = result.get('max_dd', 0)
            item = NumericTableItem(f"{dd:,.0f}", float(dd))
            item.setForeground(QColor("#ff9800"))
            self.table.setItem(row_idx, 3, item)

            # İşlem
            trades = result.get('trades', 0)
            self.table.setItem(row_idx, 4, NumericTableItem(str(trades), float(trades)))

            # Sharpe
            sh = result.get('sharpe', 0)
            item = NumericTableItem(f"{sh:.2f}", float(sh))
            if sh > 0.5:
                item.setForeground(QColor("#4CAF50"))
            self.table.setItem(row_idx, 5, item)

            # Fitness
            fit = result.get('fitness', 0)
            item = NumericTableItem(f"{fit:,.0f}", float(fit))
            if fit > 0:
                item.setForeground(QColor("#4CAF50"))
            else:
                item.setForeground(QColor("#f44336"))
            self.table.setItem(row_idx, 6, item)

            # Test Kar
            test_net = result.get('test_net', None)
            if test_net is not None:
                item = NumericTableItem(f"{test_net:,.0f}", float(test_net))
                item.setForeground(QColor("#4CAF50") if test_net > 0 else QColor("#f44336"))
            else:
                item = NumericTableItem("-", -999999.0)
                item.setForeground(QColor("#555"))
            self.table.setItem(row_idx, 7, item)

            # Test PF
            test_pf = result.get('test_pf', None)
            if test_pf is not None:
                item = NumericTableItem(f"{test_pf:.2f}", float(test_pf))
            else:
                item = NumericTableItem("-", -999999.0)
                item.setForeground(QColor("#555"))
            self.table.setItem(row_idx, 8, item)

            # --- Parametre sütunları ---
            for p_idx, (p_key, p_label) in enumerate(self._param_columns):
                col = n_metric + p_idx
                val = params.get(p_key, None)
                if val is not None:
                    if isinstance(val, float):
                        display = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
                    else:
                        display = str(val)
                    item = NumericTableItem(display, float(val) if isinstance(val, (int, float)) else 0.0)
                    item.setForeground(QColor("#b0bec5"))
                else:
                    item = NumericTableItem("-", 0.0)
                    item.setForeground(QColor("#444"))
                self.table.setItem(row_idx, col, item)

        self.table.setSortingEnabled(True)  # Tekrar aktif et

    def clear(self):
        """Tabloyu temizle"""
        self._all_results.clear()
        self.table.setRowCount(0)
        self._is_final = False
        self._update_count = 0
        self.count_label.setText("Sonuç: 0")
        self.best_label.setText("")
        self.status_indicator.setText("⏳ Bekleniyor...")
        self.status_indicator.setStyleSheet("color: #ff9800; font-size: 12px;")
