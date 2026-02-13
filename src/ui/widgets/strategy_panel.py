# -*- coding: utf-8 -*-
"""
IdealQuant - Strategy Panel
Strateji seçimi ve parametre düzenleme
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QScrollArea, QFormLayout, QMessageBox
)
from PySide6.QtCore import Signal
import json
from pathlib import Path


class StrategyPanel(QWidget):
    """Strateji seçimi ve parametre düzenleme paneli"""
    
    # Signals
    config_changed = Signal(dict)  # Strateji config gönderir
    
    # Varsayılan parametreler
    STRATEGY1_DEFAULTS = {
        'min_score': 3,
        'exit_score': 3,
        'ars_period': 3,
        'ars_k': 0.01,
        'adx_period': 17,
        'adx_threshold': 25.0,
        'macdv_short': 13,
        'macdv_long': 28,
        'macdv_signal': 8,
        'netlot_threshold': 20.0,
        'yatay_ars_bars': 10,
        'ars_mesafe_threshold': 0.25,
        'bb_period': 20,
        'bb_std': 2.0,
        'bb_width_multiplier': 0.8,
        'bb_avg_period': 50,
        'filter_score_threshold': 2,
    }
    
    STRATEGY2_DEFAULTS = {
        'ars_ema_period': 3,
        'ars_atr_period': 10,
        'ars_atr_mult': 0.5,
        'ars_min_band': 0.002,
        'ars_max_band': 0.015,
        'momentum_period': 5,
        'breakout_period': 10,
        'mfi_period': 14,
        'mfi_hhv_period': 14,
        'volume_hhv_period': 14,
        'atr_exit_period': 14,
        'atr_sl_mult': 2.0,
        'atr_tp_mult': 5.0,
        'atr_trail_mult': 2.0,
        'exit_confirm_bars': 2,
        'exit_confirm_mult': 1.0,
    }
    
    STRATEGY3_DEFAULTS = {
        'ema_period': 21,
        'dsma_period': 50,
        'ma_period': 20,
        'hh_period': 25,
        'vol_hhv_period': 14,
        'mom_period': 60,
        'mom_alt': 98.0,
        'mom_ust': 102.0,
        'atr_period': 14,
        'atr_sl': 2.0,
        'atr_tp': 4.0,
        'atr_trail': 2.5,
    }
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.param_widgets = {}
        self._setup_ui()
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Strateji seçimi
        select_group = self._create_strategy_select()
        layout.addWidget(select_group)
        
        # Parametre düzenleme (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        scroll.setWidget(self.params_container)
        layout.addWidget(scroll, 1)
        
        # Alt butonlar
        btn_row = QHBoxLayout()
        
        reset_btn = QPushButton("Varsayilana Don")
        reset_btn.clicked.connect(self._reset_to_defaults)
        btn_row.addWidget(reset_btn)
        
        load_btn = QPushButton("Preset Yukle")
        load_btn.clicked.connect(self._load_preset)
        btn_row.addWidget(load_btn)
        
        save_btn = QPushButton("Preset Kaydet")
        save_btn.clicked.connect(self._save_preset)
        btn_row.addWidget(save_btn)
        
        btn_row.addStretch()
        
        apply_btn = QPushButton("Uygula")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self._apply_config)
        btn_row.addWidget(apply_btn)
        
        layout.addLayout(btn_row)
        
        # Manuel Backtest bölümü
        backtest_group = QGroupBox("Manuel Backtest")
        backtest_layout = QVBoxLayout(backtest_group)
        
        # Backtest butonu
        backtest_btn_row = QHBoxLayout()
        self.backtest_btn = QPushButton("Backtest Calistir")
        self.backtest_btn.clicked.connect(self._run_manual_backtest)
        backtest_btn_row.addWidget(self.backtest_btn)
        backtest_btn_row.addStretch()
        backtest_layout.addLayout(backtest_btn_row)
        
        # Sonuç etiketi
        self.backtest_result_label = QLabel("Veri yukleyip parametreleri ayarladiktan sonra backtest calistirin.")
        self.backtest_result_label.setWordWrap(True)
        self.backtest_result_label.setStyleSheet("padding: 10px; background-color: #1e1e2e; border-radius: 5px;")
        backtest_layout.addWidget(self.backtest_result_label)
        
        layout.addWidget(backtest_group)
        
        # Varsayılan stratejiyi yükle
        self._on_strategy_changed(0)
    
    def _create_strategy_select(self) -> QGroupBox:
        """Strateji seçimi grubu"""
        group = QGroupBox("🎯 Strateji Seçimi")
        layout = QHBoxLayout(group)
        
        layout.addWidget(QLabel("Strateji:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Strateji 1 - Score-Based Gatekeeper",
            "Strateji 2 - ARS Trend Takip v2",
            "Strateji 3 - Paradise"
        ])
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        layout.addWidget(self.strategy_combo, 1)
        
        layout.addWidget(QLabel("Vade Tipi:"))
        self.vade_combo = QComboBox()
        self.vade_combo.addItems(["ENDEKS", "SPOT"])
        layout.addWidget(self.vade_combo)
        
        return group
    
    def _on_strategy_changed(self, index: int):
        """Strateji değiştiğinde parametreleri güncelle"""
        # Mevcut widget'ları temizle
        for widget in self.param_widgets.values():
            widget.setParent(None)
        self.param_widgets.clear()
        
        # Layout'taki widget'ları temizle
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Yeni parametreleri oluştur
        if index == 0:
            self._create_strategy1_params()
        elif index == 1:
            self._create_strategy2_params()
        else:
            self._create_strategy3_params()
    
    def _create_strategy1_params(self):
        """Strateji 1 parametrelerini oluştur"""
        defaults = self.STRATEGY1_DEFAULTS
        
        # Skor Ayarları
        score_group = QGroupBox("📊 Skor Ayarları")
        score_layout = QFormLayout(score_group)
        self._add_spin('min_score', "Min Onay Skoru:", 1, 4, defaults['min_score'], score_layout)
        self._add_spin('exit_score', "Çıkış Hassasiyeti:", 1, 4, defaults['exit_score'], score_layout)
        self.params_layout.addWidget(score_group)
        
        # ARS Ayarları
        ars_group = QGroupBox("📈 ARS Parametreleri")
        ars_layout = QFormLayout(ars_group)
        self._add_spin('ars_period', "ARS Periyot:", 2, 20, defaults['ars_period'], ars_layout)
        self._add_double_spin('ars_k', "ARS K:", 0.001, 0.1, defaults['ars_k'], ars_layout, 3)
        self.params_layout.addWidget(ars_group)
        
        # ADX Ayarları
        adx_group = QGroupBox("📉 ADX Parametreleri")
        adx_layout = QFormLayout(adx_group)
        self._add_spin('adx_period', "ADX Periyot:", 5, 30, defaults['adx_period'], adx_layout)
        self._add_double_spin('adx_threshold', "ADX Eşik:", 10, 50, defaults['adx_threshold'], adx_layout)
        self.params_layout.addWidget(adx_group)
        
        # MACD-V Ayarları
        macdv_group = QGroupBox("📊 MACD-V Parametreleri")
        macdv_layout = QFormLayout(macdv_group)
        self._add_spin('macdv_short', "Kısa Periyot:", 5, 20, defaults['macdv_short'], macdv_layout)
        self._add_spin('macdv_long', "Uzun Periyot:", 15, 50, defaults['macdv_long'], macdv_layout)
        self._add_spin('macdv_signal', "Sinyal Periyot:", 5, 20, defaults['macdv_signal'], macdv_layout)
        self.params_layout.addWidget(macdv_group)
        
        # NetLot Ayarları
        netlot_group = QGroupBox("💰 NetLot Parametreleri")
        netlot_layout = QFormLayout(netlot_group)
        self._add_double_spin('netlot_threshold', "NetLot Eşik:", 0, 100, defaults['netlot_threshold'], netlot_layout)
        self.params_layout.addWidget(netlot_group)
        
        # Yatay Filtre
        filter_group = QGroupBox("🔲 Yatay Filtre")
        filter_layout = QFormLayout(filter_group)
        self._add_spin('yatay_ars_bars', "ARS Bar Sayısı:", 5, 30, defaults['yatay_ars_bars'], filter_layout)
        self._add_double_spin('ars_mesafe_threshold', "ARS Mesafe Eşik:", 0.1, 1.0, defaults['ars_mesafe_threshold'], filter_layout)
        self._add_spin('bb_period', "BB Periyot:", 10, 50, defaults['bb_period'], filter_layout)
        self._add_double_spin('bb_std', "BB StdDev:", 1.0, 3.0, defaults['bb_std'], filter_layout)
        self._add_double_spin('bb_width_multiplier', "BB Genişlik Çarpanı:", 0.5, 1.5, defaults['bb_width_multiplier'], filter_layout)
        self._add_spin('filter_score_threshold', "Filtre Skor Eşik:", 1, 4, defaults['filter_score_threshold'], filter_layout)
        self.params_layout.addWidget(filter_group)
        
        self.params_layout.addStretch()
    
    def _create_strategy2_params(self):
        """Strateji 2 parametrelerini oluştur"""
        defaults = self.STRATEGY2_DEFAULTS
        
        # ARS Ayarları
        ars_group = QGroupBox("📈 ARS Parametreleri")
        ars_layout = QFormLayout(ars_group)
        self._add_spin('ars_ema_period', "EMA Periyot:", 2, 20, defaults['ars_ema_period'], ars_layout)
        self._add_spin('ars_atr_period', "ATR Periyot:", 5, 30, defaults['ars_atr_period'], ars_layout)
        self._add_double_spin('ars_atr_mult', "ATR Çarpan:", 0, 2, defaults['ars_atr_mult'], ars_layout)
        self._add_double_spin('ars_min_band', "Min Band:", 0.001, 0.05, defaults['ars_min_band'], ars_layout, 3)
        self._add_double_spin('ars_max_band', "Max Band:", 0.005, 0.1, defaults['ars_max_band'], ars_layout, 3)
        self.params_layout.addWidget(ars_group)
        
        # Giriş Filtreleri
        entry_group = QGroupBox("🎯 Giriş Filtreleri")
        entry_layout = QFormLayout(entry_group)
        self._add_spin('momentum_period', "Momentum Periyot:", 3, 20, defaults['momentum_period'], entry_layout)
        self._add_spin('breakout_period', "Breakout Periyot:", 5, 30, defaults['breakout_period'], entry_layout)
        self._add_spin('mfi_period', "MFI Periyot:", 7, 30, defaults['mfi_period'], entry_layout)
        self._add_spin('mfi_hhv_period', "MFI HHV Periyot:", 7, 30, defaults['mfi_hhv_period'], entry_layout)
        self._add_spin('volume_hhv_period', "Volume HHV Periyot:", 7, 30, defaults['volume_hhv_period'], entry_layout)
        self.params_layout.addWidget(entry_group)
        
        # ATR Çıkış
        exit_group = QGroupBox("🚪 ATR Çıkış Parametreleri")
        exit_layout = QFormLayout(exit_group)
        self._add_spin('atr_exit_period', "ATR Periyot:", 7, 30, defaults['atr_exit_period'], exit_layout)
        self._add_double_spin('atr_sl_mult', "Stop Loss Çarpan:", 1, 5, defaults['atr_sl_mult'], exit_layout)
        self._add_double_spin('atr_tp_mult', "Take Profit Çarpan:", 2, 10, defaults['atr_tp_mult'], exit_layout)
        self._add_double_spin('atr_trail_mult', "Trailing Çarpan:", 1, 5, defaults['atr_trail_mult'], exit_layout)
        self._add_spin('exit_confirm_bars', "Onay Bar Sayısı:", 1, 5, defaults['exit_confirm_bars'], exit_layout)
        self._add_double_spin('exit_confirm_mult', "Onay Mesafe Çarpanı:", 0.5, 3, defaults['exit_confirm_mult'], exit_layout)
        self.params_layout.addWidget(exit_group)
        
        self.params_layout.addStretch()

    def _create_strategy3_params(self):
        """Strateji 3 (Paradise) parametrelerini oluştur"""
        defaults = self.STRATEGY3_DEFAULTS
        
        # Trend
        trend_group = QGroupBox("📈 Trend Filtreleri")
        trend_layout = QFormLayout(trend_group)
        self._add_spin('ema_period', "EMA Periyot:", 5, 100, defaults['ema_period'], trend_layout)
        self._add_spin('dsma_period', "DSMA Periyot:", 10, 200, defaults['dsma_period'], trend_layout)
        self._add_spin('ma_period', "MA Periyot:", 5, 100, defaults['ma_period'], trend_layout)
        self.params_layout.addWidget(trend_group)
        
        # Breakout
        bo_group = QGroupBox("🚀 Breakout Ayarları")
        bo_layout = QFormLayout(bo_group)
        self._add_spin('hh_period', "HH/LL Periyot:", 5, 100, defaults['hh_period'], bo_layout)
        self._add_spin('vol_hhv_period', "Hacim HHV Periyot:", 5, 50, defaults['vol_hhv_period'], bo_layout)
        self.params_layout.addWidget(bo_group)
        
        # Momentum
        mom_group = QGroupBox("⚡ Momentum Bandı")
        mom_layout = QFormLayout(mom_group)
        self._add_spin('mom_period', "Momentum Periyot:", 10, 200, defaults['mom_period'], mom_layout)
        self._add_double_spin('mom_alt', "Alt Eşik (100 - X):", 90.0, 100.0, defaults['mom_alt'], mom_layout)
        self._add_double_spin('mom_ust', "Üst Eşik (100 + X):", 100.0, 110.0, defaults['mom_ust'], mom_layout)
        self.params_layout.addWidget(mom_group)
        
        # Risk / Çıkış
        risk_group = QGroupBox("🛡️ Risk & Çıkış")
        risk_layout = QFormLayout(risk_group)
        self._add_spin('atr_period', "ATR Periyot:", 5, 50, defaults['atr_period'], risk_layout)
        self._add_double_spin('atr_sl', "ATR Stop Loss:", 0.5, 10.0, defaults['atr_sl'], risk_layout)
        self._add_double_spin('atr_tp', "ATR Take Profit:", 1.0, 20.0, defaults['atr_tp'], risk_layout)
        self._add_double_spin('atr_trail', "ATR Trailing Stop:", 0.5, 10.0, defaults['atr_trail'], risk_layout)
        self.params_layout.addWidget(risk_group)
        
        self.params_layout.addStretch()
    
    def _add_spin(self, name: str, label: str, min_val: int, max_val: int, default: int, layout: QFormLayout):
        """Integer SpinBox ekle"""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        self.param_widgets[name] = spin
        layout.addRow(label, spin)
    
    def _add_double_spin(self, name: str, label: str, min_val: float, max_val: float, default: float, layout: QFormLayout, decimals: int = 2):
        """Double SpinBox ekle"""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1 if decimals <= 2 else 0.001)
        spin.setValue(default)
        self.param_widgets[name] = spin
        layout.addRow(label, spin)
    
    def _reset_to_defaults(self):
        """Varsayılan değerlere dön"""
        index = self.strategy_combo.currentIndex()
        if index == 0:
            defaults = self.STRATEGY1_DEFAULTS
        elif index == 1:
            defaults = self.STRATEGY2_DEFAULTS
        else:
            defaults = self.STRATEGY3_DEFAULTS
        
        for name, widget in self.param_widgets.items():
            if name in defaults:
                widget.setValue(defaults[name])
    
    def _load_preset(self):
        """Preset yükle"""
        from PySide6.QtWidgets import QFileDialog
        import json
        
        preset_dir = Path(__file__).parent.parent.parent.parent / "presets"
        preset_dir.mkdir(exist_ok=True)
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Preset Yükle", str(preset_dir), "JSON Files (*.json)"
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    preset = json.load(f)
                
                # Strateji seç
                if 'strategy' in preset:
                    self.strategy_combo.setCurrentIndex(preset['strategy'] - 1)
                
                # Vade tipi
                if 'vade_tipi' in preset:
                    self.vade_combo.setCurrentText(preset['vade_tipi'])
                
                # Parametreleri yükle
                for name, value in preset.items():
                    if name in self.param_widgets:
                        self.param_widgets[name].setValue(value)
                
                QMessageBox.information(self, "Başarılı", f"Preset yüklendi: {Path(filepath).stem}")
            except Exception as e:
                QMessageBox.warning(self, "Hata", f"Preset yüklenemedi: {e}")
    
    def _save_preset(self):
        """Preset kaydet"""
        from PySide6.QtWidgets import QFileDialog
        import json
        
        preset_dir = Path(__file__).parent.parent.parent.parent / "presets"
        preset_dir.mkdir(exist_ok=True)
        
        # Varsayılan isim
        idx = self.strategy_combo.currentIndex()
        strategy_name = "strateji1" if idx == 0 else ("strateji2" if idx == 1 else "paradise")
        default_name = f"{strategy_name}_preset.json"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Preset Kaydet", str(preset_dir / default_name), "JSON Files (*.json)"
        )
        
        if filepath:
            try:
                config = self.get_config()
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Başarılı", f"Preset kaydedildi: {Path(filepath).name}")
            except Exception as e:
                QMessageBox.warning(self, "Hata", f"Preset kaydedilemedi: {e}")
    
    def _apply_config(self):
        """Konfigürasyonu uygula"""
        config = self.get_config()
        self.config_changed.emit(config)
        QMessageBox.information(self, "Uygulama", f"✅ {len(config)} parametre uygulandı.")
    
    def get_config(self) -> dict:
        """Mevcut konfigürasyonu döndür"""
        config = {
            'strategy': self.strategy_combo.currentIndex() + 1,
            'vade_tipi': self.vade_combo.currentText(),
        }
        
        for name, widget in self.param_widgets.items():
            config[name] = widget.value()
        
        return config
    
    def set_data(self, df):
        """Veri set et (DataPanel'den sinyal)"""
        self.df = df
    
    def _run_manual_backtest(self):
        """Manuel backtest calistir"""
        if self.df is None or len(self.df) == 0:
            QMessageBox.warning(self, "Uyari", "Lutfen once veri yukleyin.")
            return
        
        try:
            config = self.get_config()
            strategy_idx = config['strategy']
            
            self.backtest_result_label.setText("Backtest calisiyor...")
            self.backtest_btn.setEnabled(False)
            
            # Gerekli verileri hazirla
            df = self.df
            opens = df['Acilis'].tolist() if 'Acilis' in df.columns else df['Open'].tolist()
            highs = df['Yuksek'].tolist() if 'Yuksek' in df.columns else df['High'].tolist()
            lows = df['Dusuk'].tolist() if 'Dusuk' in df.columns else df['Low'].tolist()
            closes = df['Kapanis'].tolist() if 'Kapanis' in df.columns else df['Close'].tolist()
            typical = df['Tipik'].tolist() if 'Tipik' in df.columns else [(h+l+c)/3 for h,l,c in zip(highs,lows,closes)]
            dates = df['DateTime'].tolist() if 'DateTime' in df.columns else None
            
            if strategy_idx == 1:
                from src.strategies.score_based import ScoreBasedStrategy
                strategy = ScoreBasedStrategy.from_config_dict(
                    {'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes, 'typical': typical, 'dates': dates},
                    config,
                    dates
                )
            elif strategy_idx == 3:
                from src.strategies.paradise_strategy import ParadiseStrategy
                strategy = ParadiseStrategy.from_config_dict(
                    {'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes, 'typical': typical, 'dates': dates},
                    config,
                    dates
                )
            else:
                from src.strategies.ars_trend_v2 import ARSTrendStrategyV2
                strategy = ARSTrendStrategyV2.from_config_dict(
                    {'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes, 'typical': typical, 'dates': dates},
                    config,
                    dates
                )
            
            # Sinyalleri uret
            signals, exits_long, exits_short = strategy.generate_all_signals()
            
            # Basit backtest
            import numpy as np
            position = 0  # 0=flat, 1=long, -1=short
            entry_price = 0
            trades = []
            
            for i in range(len(closes)):
                if position == 0:
                    if signals[i] == 1:
                        position = 1
                        entry_price = closes[i]
                    elif signals[i] == -1:
                        position = -1
                        entry_price = closes[i]
                elif position == 1:
                    if exits_long[i] or signals[i] == -1:
                        pnl = closes[i] - entry_price
                        trades.append(pnl)
                        position = 0 if exits_long[i] else -1
                        entry_price = closes[i] if signals[i] == -1 else 0
                elif position == -1:
                    if exits_short[i] or signals[i] == 1:
                        pnl = entry_price - closes[i]
                        trades.append(pnl)
                        position = 0 if exits_short[i] else 1
                        entry_price = closes[i] if signals[i] == 1 else 0
            
            # Metrikleri hesapla
            if len(trades) == 0:
                self.backtest_result_label.setText("Hic islem bulunamadi.")
                self.backtest_btn.setEnabled(True)
                return
            
            trades_arr = np.array(trades)
            total_profit = np.sum(trades_arr)
            win_count = np.sum(trades_arr > 0)
            loss_count = np.sum(trades_arr < 0)
            win_rate = win_count / len(trades_arr) * 100
            
            gross_profit = np.sum(trades_arr[trades_arr > 0]) if win_count > 0 else 0
            gross_loss = abs(np.sum(trades_arr[trades_arr < 0])) if loss_count > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Max drawdown
            equity = np.cumsum(trades_arr)
            peak = np.maximum.accumulate(equity)
            drawdown = peak - equity
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
            
            result_text = f"""
BACKTEST SONUCLARI
==================
Toplam Islem: {len(trades)}
Kazanan: {win_count} | Kaybeden: {loss_count}
Win Rate: {win_rate:.1f}%

Net Kar: {total_profit:.0f} puan
Profit Factor: {profit_factor:.2f}
Max Drawdown: {max_dd:.0f} puan

Ortalama Kar: {np.mean(trades_arr):.1f} puan/islem
"""
            
            self.backtest_result_label.setText(result_text.strip())
            self.backtest_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Backtest hatasi: {str(e)}")
            self.backtest_btn.setEnabled(True)

