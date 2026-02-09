# -*- coding: utf-8 -*-
"""
IdealQuant - Database Module
SQLite veritabanı bağlantısı ve süreç yönetimi tabloları
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os


def get_db_path() -> Path:
    """Veritabanı dosyası yolunu döndür"""
    # Kullanıcı home dizininde .idealquant klasörü
    home = Path.home()
    db_dir = home / ".idealquant"
    db_dir.mkdir(exist_ok=True)
    return db_dir / "processes.db"


class Database:
    """SQLite veritabanı yöneticisi"""
    
    _instance: Optional['Database'] = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db_path = get_db_path()
        self._create_tables()
        self._migrate_tables()
        self._initialized = True
    
    def _get_connection(self) -> sqlite3.Connection:
        """Yeni connection döndür (thread-safe için her çağrıda yeni)"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _create_tables(self):
        """Veritabanı tablolarını oluştur"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Süreçler tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                period TEXT NOT NULL,
                data_file TEXT NOT NULL,
                data_rows INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                commission REAL DEFAULT 0,
                slippage REAL DEFAULT 0
            )
        """)
        
        # Optimizasyon sonuçları tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT NOT NULL,
                strategy_index INTEGER NOT NULL,
                method TEXT NOT NULL,
                params TEXT NOT NULL,
                net_profit REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                fitness REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (process_id) REFERENCES processes(process_id),
                UNIQUE(process_id, strategy_index, method)
            )
        """)
        
        # Validasyon sonuçları tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_id INTEGER NOT NULL,
                wfa_efficiency REAL DEFAULT 0,
                monte_carlo_prob REAL DEFAULT 0,
                stability_score REAL DEFAULT 0,
                is_final INTEGER DEFAULT 0,
                final_params TEXT,
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (optimization_id) REFERENCES optimization_results(id)
            )
        """)
        
        # Grup optimizasyon sonuçları tablosu (Hibrit Optimizer için)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS group_optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_id TEXT NOT NULL,
                strategy_index INTEGER NOT NULL,
                group_name TEXT NOT NULL,
                rank INTEGER DEFAULT 0,
                params TEXT NOT NULL,
                net_profit REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (process_id) REFERENCES processes(process_id)
            )
        """)
        
        conn.commit()
    
        conn.commit()
        conn.close()

    def _migrate_tables(self):
        """Eksik kolonları ekle (Migration)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # processes tablosuna commission ve slippage ekle
            cursor.execute("PRAGMA table_info(processes)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'commission' not in columns:
                cursor.execute("ALTER TABLE processes ADD COLUMN commission REAL DEFAULT 0")
                
            if 'slippage' not in columns:
                cursor.execute("ALTER TABLE processes ADD COLUMN slippage REAL DEFAULT 0")
                
            conn.commit()
        except Exception as e:
            print(f"Migration error: {e}")
        finally:
            conn.close()
    
    # =========================================================================
    # PROCESS CRUD
    # =========================================================================
    
    def create_process(self, symbol: str, period: str, data_file: str, 
                       data_rows: int = 0) -> str:
        """Yeni süreç oluştur, process_id döndür"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_id = f"{symbol}_{period}_{timestamp}"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO processes (process_id, symbol, period, data_file, data_rows, status, commission, slippage)
            VALUES (?, ?, ?, ?, ?, 'pending', 0, 0)
        """, (process_id, symbol, period, data_file, data_rows))
        
        conn.commit()
        conn.close()
        
        return process_id
    
    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Süreç detaylarını getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM processes WHERE process_id = ?", (process_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_processes(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Tüm süreçleri listele (opsiyonel status filtresi)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute(
                "SELECT * FROM processes WHERE status = ? ORDER BY created_at DESC", 
                (status,)
            )
        else:
            cursor.execute("SELECT * FROM processes ORDER BY created_at DESC")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_process_status(self, process_id: str, status: str):
        """Süreç durumunu güncelle"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE processes 
            SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE process_id = ?
        """, (status, process_id))
        
        conn.commit()
        conn.close()
    
    def delete_process(self, process_id: str):
        """Süreci ve ilişkili tüm kayıtları sil"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Önce validation_results sil (optimization_id üzerinden)
        cursor.execute("""
            DELETE FROM validation_results 
            WHERE optimization_id IN (
                SELECT id FROM optimization_results WHERE process_id = ?
            )
        """, (process_id,))
        
        # Grup optimizasyon sonuçlarını sil
        cursor.execute("DELETE FROM group_optimization_results WHERE process_id = ?", (process_id,))
        
        # Sonra optimization_results sil
        cursor.execute("DELETE FROM optimization_results WHERE process_id = ?", (process_id,))
        
        # En son processes sil
        cursor.execute("DELETE FROM processes WHERE process_id = ?", (process_id,))
        
        conn.commit()
        conn.close()

    def update_process_costs(self, process_id: str, commission: float, slippage: float):
        """Süreç için komisyon ve kayma değerlerini güncelle"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE processes SET commission = ?, slippage = ?, updated_at = CURRENT_TIMESTAMP
            WHERE process_id = ?
        """, (commission, slippage, process_id))
        conn.commit()
        conn.close()

    def get_process_costs(self, process_id: str) -> Dict[str, float]:
        """Sürecin maliyet ayarlarını getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT commission, slippage FROM processes WHERE process_id = ?", (process_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'commission': row['commission'], 'slippage': row['slippage']}
        return {'commission': 0.0, 'slippage': 0.0}

    def get_process_data_file(self, process_id: str) -> Optional[str]:
        """Sürecin veri dosyası yolunu getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT data_file FROM processes WHERE process_id = ?", (process_id,))
        row = cursor.fetchone()
        conn.close()
        return row['data_file'] if row else None
    
    # =========================================================================
    # OPTIMIZATION RESULTS CRUD
    # =========================================================================
    
    def save_optimization_result(self, process_id: str, strategy_index: int,
                                  method: str, params: Dict[str, Any],
                                  result: Dict[str, Any]) -> int:
        """Optimizasyon sonucu kaydet (UPSERT), id döndür"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        params_json = json.dumps(params, ensure_ascii=False)
        
        # UPSERT (varsa güncelle, yoksa ekle)
        cursor.execute("""
            INSERT INTO optimization_results 
                (process_id, strategy_index, method, params, net_profit, 
                 max_drawdown, profit_factor, total_trades, win_rate, fitness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(process_id, strategy_index, method) 
            DO UPDATE SET
                params = excluded.params,
                net_profit = excluded.net_profit,
                max_drawdown = excluded.max_drawdown,
                profit_factor = excluded.profit_factor,
                total_trades = excluded.total_trades,
                win_rate = excluded.win_rate,
                fitness = excluded.fitness,
                created_at = CURRENT_TIMESTAMP
        """, (
            process_id, strategy_index, method, params_json,
            result.get('net_profit', 0),
            result.get('max_drawdown', 0),
            result.get('profit_factor', 0),
            result.get('total_trades', 0),
            result.get('win_rate', 0),
            result.get('fitness', 0)
        ))
        
        conn.commit()
        
        # ID'yi al
        cursor.execute("""
            SELECT id FROM optimization_results 
            WHERE process_id = ? AND strategy_index = ? AND method = ?
        """, (process_id, strategy_index, method))
        
        row = cursor.fetchone()
        result_id = row['id'] if row else 0
        
        conn.close()
        
        # Süreç durumunu güncelle
        self.update_process_status(process_id, 'optimized')
        
        return result_id
    
    def get_optimization_results(self, process_id: str) -> List[Dict[str, Any]]:
        """Bir sürece ait tüm optimizasyon sonuçlarını getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM optimization_results 
            WHERE process_id = ? 
            ORDER BY strategy_index, method
        """, (process_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            d = dict(row)
            d['params'] = json.loads(d['params'])
            results.append(d)
        
        return results
    
    def get_optimization_result_by_id(self, opt_id: int) -> Optional[Dict[str, Any]]:
        """ID ile optimizasyon sonucu getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM optimization_results WHERE id = ?", (opt_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            d = dict(row)
            d['params'] = json.loads(d['params'])
            return d
        return None
    
    # =========================================================================
    # VALIDATION RESULTS CRUD
    # =========================================================================
    
    def save_validation_result(self, optimization_id: int, 
                                wfa_efficiency: float = 0,
                                monte_carlo_prob: float = 0,
                                stability_score: float = 0,
                                is_final: bool = False,
                                final_params: Optional[Dict] = None,
                                notes: str = "") -> int:
        """Validasyon sonucu kaydet"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        final_params_json = json.dumps(final_params) if final_params else None
        
        cursor.execute("""
            INSERT INTO validation_results 
                (optimization_id, wfa_efficiency, monte_carlo_prob, 
                 stability_score, is_final, final_params, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            optimization_id, wfa_efficiency, monte_carlo_prob,
            stability_score, 1 if is_final else 0, final_params_json, notes
        ))
        
        conn.commit()
        result_id = cursor.lastrowid
        
        # Eğer final seçildiyse, süreç durumunu güncelle
        if is_final:
            # optimization_result'tan process_id al
            cursor.execute(
                "SELECT process_id FROM optimization_results WHERE id = ?", 
                (optimization_id,)
            )
            row = cursor.fetchone()
            if row:
                self.update_process_status(row['process_id'], 'validated')
        
        conn.close()
        return result_id
    
    def get_validation_results(self, process_id: str) -> List[Dict[str, Any]]:
        """Bir sürece ait tüm validasyon sonuçlarını getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT v.*, o.strategy_index, o.method, o.params, o.net_profit
            FROM validation_results v
            JOIN optimization_results o ON v.optimization_id = o.id
            WHERE o.process_id = ?
            ORDER BY v.is_final DESC, o.strategy_index, o.method
        """, (process_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            d = dict(row)
            d['params'] = json.loads(d['params'])
            if d['final_params']:
                d['final_params'] = json.loads(d['final_params'])
            results.append(d)
        
        return results
    
    def get_final_params(self, process_id: str) -> Dict[int, Dict[str, Any]]:
        """Final olarak işaretlenmiş parametreleri getir (strateji bazlı)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT o.strategy_index, o.params, v.final_params
            FROM validation_results v
            JOIN optimization_results o ON v.optimization_id = o.id
            WHERE o.process_id = ? AND v.is_final = 1
        """, (process_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            strategy_idx = row['strategy_index']
            params = json.loads(row['final_params'] or row['params'])
            result[strategy_idx] = params
        
        return result
    
    def set_final_selection(self, optimization_id: int, 
                            final_params: Optional[Dict] = None):
        """Bir optimizasyon sonucunu final olarak işaretle"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Önce aynı strateji için diğer final'leri kaldır
        cursor.execute("""
            UPDATE validation_results 
            SET is_final = 0
            WHERE optimization_id IN (
                SELECT o2.id FROM optimization_results o1
                JOIN optimization_results o2 
                    ON o1.process_id = o2.process_id 
                    AND o1.strategy_index = o2.strategy_index
                WHERE o1.id = ?
            )
        """, (optimization_id,))
        
        # Bu optimizasyon için validation kaydı var mı kontrol et
        cursor.execute(
            "SELECT id FROM validation_results WHERE optimization_id = ?",
            (optimization_id,)
        )
        existing = cursor.fetchone()
        
        final_params_json = json.dumps(final_params) if final_params else None
        
        if existing:
            # Varsa güncelle
            cursor.execute("""
                UPDATE validation_results 
                SET is_final = 1, final_params = ?
                WHERE optimization_id = ?
            """, (final_params_json, optimization_id))
        else:
            # Yoksa yeni kayıt oluştur
            cursor.execute("""
                INSERT INTO validation_results (optimization_id, is_final, final_params)
                VALUES (?, 1, ?)
            """, (optimization_id, final_params_json))
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # GROUP OPTIMIZATION RESULTS CRUD (Hibrit Optimizer)
    # =========================================================================
    
    def save_group_result(self, process_id: str, strategy_index: int,
                          group_name: str, rank: int, params: Dict[str, Any],
                          result: Dict[str, Any]) -> int:
        """Grup optimizasyon sonucu kaydet"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        params_json = json.dumps(params, ensure_ascii=False)
        
        cursor.execute("""
            INSERT INTO group_optimization_results 
                (process_id, strategy_index, group_name, rank, params,
                 net_profit, max_drawdown, profit_factor, total_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            process_id, strategy_index, group_name, rank, params_json,
            result.get('net_profit', 0),
            result.get('max_drawdown', result.get('max_dd', 0)),
            result.get('profit_factor', result.get('pf', 0)),
            result.get('total_trades', result.get('trades', 0))
        ))
        
        conn.commit()
        result_id = cursor.lastrowid
        conn.close()
        
        return result_id
    
    def save_group_results_batch(self, process_id: str, strategy_index: int,
                                  group_name: str, results: List[Dict]) -> int:
        """Grup sonuçlarını toplu kaydet (En iyi N tane)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Önce bu grup için eski sonuçları sil
        cursor.execute("""
            DELETE FROM group_optimization_results 
            WHERE process_id = ? AND strategy_index = ? AND group_name = ?
        """, (process_id, strategy_index, group_name))
        
        # Yeni sonuçları ekle (sıralı)
        for rank, result in enumerate(results, 1):
            params = result.get('params', result)
            params_json = json.dumps(params, ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO group_optimization_results 
                    (process_id, strategy_index, group_name, rank, params,
                     net_profit, max_drawdown, profit_factor, total_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                process_id, strategy_index, group_name, rank, params_json,
                result.get('net_profit', 0),
                result.get('max_drawdown', result.get('max_dd', 0)),
                result.get('profit_factor', result.get('pf', 0)),
                result.get('total_trades', result.get('trades', 0))
            ))
        
        conn.commit()
        count = len(results)
        conn.close()
        
        return count
    
    def get_group_results(self, process_id: str, strategy_index: int = None,
                          group_name: str = None) -> List[Dict[str, Any]]:
        """Grup optimizasyon sonuçlarını getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM group_optimization_results WHERE process_id = ?"
        params = [process_id]
        
        if strategy_index is not None:
            query += " AND strategy_index = ?"
            params.append(strategy_index)
        
        if group_name:
            query += " AND group_name = ?"
            params.append(group_name)
        
        query += " ORDER BY group_name, rank"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            d = dict(row)
            d['params'] = json.loads(d['params'])
            results.append(d)
        
        return results
    
    def get_best_group_params(self, process_id: str, strategy_index: int) -> Dict[str, Dict]:
        """Her grup için en iyi (rank=1) parametreleri getir"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT group_name, params FROM group_optimization_results 
            WHERE process_id = ? AND strategy_index = ? AND rank = 1
        """, (process_id, strategy_index))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            result[row['group_name']] = json.loads(row['params'])
        
        return result
    
    def clear_group_results(self, process_id: str, strategy_index: int = None):
        """Grup sonuçlarını temizle"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if strategy_index is not None:
            cursor.execute("""
                DELETE FROM group_optimization_results 
                WHERE process_id = ? AND strategy_index = ?
            """, (process_id, strategy_index))
        else:
            cursor.execute("""
                DELETE FROM group_optimization_results WHERE process_id = ?
            """, (process_id,))
        
        conn.commit()
        conn.close()

    def delete_process(self, process_id: str) -> bool:
        """Süreci ve ilişkili tüm verileri sil"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 1. Validation Results (Optimization ID üzerinden bağlı)
            cursor.execute("""
                DELETE FROM validation_results 
                WHERE optimization_id IN (
                    SELECT id FROM optimization_results WHERE process_id = ?
                )
            """, (process_id,))
            
            # 2. Optimization Results
            cursor.execute("DELETE FROM optimization_results WHERE process_id = ?", (process_id,))
            
            # 3. Group Optimization Results
            cursor.execute("DELETE FROM group_optimization_results WHERE process_id = ?", (process_id,))
            
            # 4. Process
            cursor.execute("DELETE FROM processes WHERE process_id = ?", (process_id,))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Delete process error: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()

    def clear_database(self) -> bool:
        """Tüm veritabanını temizle (Tabloları boşalt)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Tabloları sırayla sil (Foreign Key sırasına dikkat ederek)
            cursor.execute("DELETE FROM validation_results")
            cursor.execute("DELETE FROM group_optimization_results")
            cursor.execute("DELETE FROM optimization_results")
            cursor.execute("DELETE FROM processes")
            
            # Auto-increment sayaçlarını sıfırla
            cursor.execute("DELETE FROM sqlite_sequence")
            
            conn.commit()
            self.vacuum()  # Alanı geri kazan
            return True
            
        except Exception as e:
            print(f"Clear database error: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()

    def vacuum(self):
        """Veritabanını optimize et (VACUUM)"""
        conn = self._get_connection()
        try:
            conn.execute("VACUUM")
        except Exception as e:
            print(f"Vacuum error: {e}")
        finally:
            conn.close()


# Singleton instance
db = Database()
    
