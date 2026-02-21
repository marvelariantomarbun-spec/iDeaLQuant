# -*- coding: utf-8 -*-
"""
CheckpointManager — Universal disk-based checkpoint for all optimizers.
Atomik JSON yazma ile crash-safe.
"""

import json
import os
import time
import glob


def _json_safe(obj):
    """JSON serialize edilemeyen tipleri dönüştür (numpy int/float vb.)."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class CheckpointManager:
    """
    Universal disk-based checkpoint manager.
    
    Her optimizasyon işlemi bir job_id ile tanımlanır.
    Checkpoint'ler JSON dosyası olarak kaydedilir.
    Yazma atomiktir (temp dosyaya yaz → rename) — yarım kalmış dosya olmaz.
    
    job_id formatı: {strategy}_{method}_{process_id}
    Örnek: s4_hybrid_VIP_X030_T_1dk_20260220
    """
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Proje kökü/checkpoints/
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_dir = os.path.join(project_root, 'checkpoints')
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _path(self, job_id: str) -> str:
        """Checkpoint dosya yolu."""
        safe_id = job_id.replace('/', '_').replace('\\', '_')
        return os.path.join(self.base_dir, f"{safe_id}.json")
    
    def save(self, job_id: str, data: dict):
        """
        Checkpoint'i diske yaz (atomik: temp'e yaz → rename).
        Crash olsa bile eski checkpoint bozulmaz.
        """
        path = self._path(job_id)
        tmp = path + ".tmp"
        try:
            payload = {
                'job_id': job_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': time.time(),
                **data
            }
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=_json_safe, ensure_ascii=False)
            os.replace(tmp, path)  # Atomik replace
        except Exception as e:
            print(f"[CHECKPOINT] Kayit hatasi ({job_id}): {e}")
            # Temp dosya kaldıysa sil
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except:
                pass
    
    def load(self, job_id: str) -> dict | None:
        """Checkpoint yükle. Yoksa None döndür."""
        path = self._path(job_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CHECKPOINT] Okuma hatasi ({job_id}): {e}")
            return None
    
    def delete(self, job_id: str):
        """Başarılı tamamlanmada checkpoint sil."""
        path = self._path(job_id)
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[CHECKPOINT] Silindi: {job_id}")
        except Exception as e:
            print(f"[CHECKPOINT] Silme hatasi ({job_id}): {e}")
    
    def list_all(self) -> list:
        """
        Tüm checkpoint'leri listele.
        Returns: [{'job_id': ..., 'timestamp': ..., 'file': ..., 'size_kb': ...}, ...]
        """
        results = []
        for fp in glob.glob(os.path.join(self.base_dir, '*.json')):
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results.append({
                    'job_id': data.get('job_id', os.path.basename(fp)),
                    'timestamp': data.get('timestamp', '?'),
                    'file': fp,
                    'size_kb': round(os.path.getsize(fp) / 1024, 1),
                    'data': data
                })
            except Exception:
                pass
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return results
    
    def delete_all(self):
        """Tüm checkpoint'leri sil."""
        for fp in glob.glob(os.path.join(self.base_dir, '*.json')):
            try:
                os.remove(fp)
            except:
                pass
    
    @staticmethod
    def make_job_id(strategy_index: int, method: str, process_id: str = None) -> str:
        """Standart job_id oluştur."""
        s_map = {0: 's1', 1: 's2', 2: 's3', 3: 's4'}
        m_map = {
            'Hibrit Grup': 'hybrid',
            'Genetik': 'genetic',
            'Bayesian': 'bayesian',
        }
        s = s_map.get(strategy_index, f's{strategy_index}')
        m = m_map.get(method, method.lower().replace(' ', '_'))
        p = (process_id or 'unknown').replace(' ', '_')[:40]
        return f"{s}_{m}_{p}"
