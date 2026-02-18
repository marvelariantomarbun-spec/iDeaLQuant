# -*- coding: utf-8 -*-
"""
Preset Manager — Parametre aralık/adım preset yönetimi
Sembol + Periyot + Strateji bazlı kayıt/yükleme.
Son kaydedilen → bir sonraki oturumda varsayılan.
"""
import json
import os
import time


class PresetManager:
    """Sembol+Periyot+Strateji bazlı parametre preset yönetimi"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Proje koku: src/optimization/../../ = proje root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.preset_dir = os.path.join(base_dir, "presets")
        os.makedirs(self.preset_dir, exist_ok=True)

    def _make_key(self, symbol: str, period: str, strategy_idx: int) -> str:
        """Dosya adı için güvenli key oluştur"""
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
        safe_period = period.replace(" ", "")
        return f"{safe_symbol}_{safe_period}_S{strategy_idx + 1}"

    def _get_path(self, key: str) -> str:
        return os.path.join(self.preset_dir, f"{key}.json")

    def save_preset(self, symbol: str, period: str, strategy_idx: int,
                    strategy_name: str, param_ranges: dict) -> str:
        """
        Parametre aralıklarını kaydet.
        param_ranges: {param_name: {min, max, step, active}, ...}
        Returns: Dosya yolu
        """
        key = self._make_key(symbol, period, strategy_idx)
        preset = {
            'symbol': symbol,
            'period': period,
            'strategy_index': strategy_idx,
            'strategy_name': strategy_name,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'params': param_ranges
        }
        path = self._get_path(key)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(preset, f, ensure_ascii=False, indent=2)
        return path

    def load_preset(self, symbol: str, period: str, strategy_idx: int) -> dict | None:
        """
        Preset yükle.
        Returns: preset dict veya None
        """
        key = self._make_key(symbol, period, strategy_idx)
        path = self._get_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[PRESET] Okuma hatasi: {e}")
            return None

    def has_preset(self, symbol: str, period: str, strategy_idx: int) -> bool:
        """Preset var mı?"""
        key = self._make_key(symbol, period, strategy_idx)
        return os.path.exists(self._get_path(key))

    def delete_preset(self, symbol: str, period: str, strategy_idx: int) -> bool:
        """Preset sil"""
        key = self._make_key(symbol, period, strategy_idx)
        path = self._get_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_presets(self) -> list:
        """Tüm presetleri listele"""
        presets = []
        if not os.path.exists(self.preset_dir):
            return presets
        for fname in os.listdir(self.preset_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(self.preset_dir, fname), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        presets.append({
                            'file': fname,
                            'symbol': data.get('symbol', '?'),
                            'period': data.get('period', '?'),
                            'strategy': data.get('strategy_name', '?'),
                            'created': data.get('created', '?'),
                            'param_count': len(data.get('params', {}))
                        })
                except:
                    pass
        return presets
