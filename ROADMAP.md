# 🗺️ IdealQuant - Yol Haritası v2.0

## 🎯 Ana Hedef
IdealData backtest sonuçları ile **%100 uyumlu** harici backtest + optimizasyon + robust parametre seçim sistemi.

**Deadline:** Pazar Geceyarısı (2 Şubat 00:00)

---

## 📋 Faz Durumları

| Faz | Durum | Açıklama | Öncelik |
|-----|-------|----------|---------|
| Faz 0 | ✅ | Proje Kurulumu | - |
| Faz 1 | ✅ | IdealData Uyumu | - |
| Faz 2 | ✅ | Optimizasyon Motoru | - |
| Faz 3 | ✅ | Robust Parametre | - |
| Faz 4 | ✅ | IdealData Entegrasyonu | - |
| Faz 5 | ✅ | **v4.1 Sistem Hizalaması** | 🔴 Kritik |
| Faz 6 | ✅ | Desktop UI (PySide6) | - |
| Faz 7 | ✅ | Veritabanı Entegrasyonu | - |
| Faz 8 | 🔄 | Agent Dokümantasyonu | 🔴 Sürekli |
| Faz 9 | 🔜 | Canlı Test & S5 Araştırma | 🟡 Düşük |

---

## ✅ FAZ 0-2: TAMAMLANDI

<details>
<summary>Detaylar için tıkla</summary>

### Faz 0: Proje Kurulumu
- [x] Proje klasörü, Git repo, temel yapı

### Faz 1: IdealData Uyumu
- [x] Veri okuma %100 uyum
- [x] İndikatörler %99+ uyum
- [x] Sinyal eşleşme %97.8
- [x] P&L eşleşme %97

### Faz 2: Optimizasyon Motoru
- [x] 32-thread paralel işleme
- [x] 3-aşamalı optimizasyon (Satellite-Drone-Stability)
- [x] Hibrit Grid Optimizer
- [x] Genetik Algoritma
- [x] **Bayesian Optimizer (Optuna)** ← YENİ
- [x] **Optimizer Audit & Bug Fixes** (Feb 11) ← YENİ
- [x] **Advanced Fitness Modeling** ← YENİ
  - Stricter Selection (Min PF 1.5)
  - "Sweet Spot" Bonus (PF 1.5-2.5)
  - Equity Smoothness (R²) Reward
  - Anti-Overtrading Logic

### Kalibrasyon (✅ TAMAMLANDI)
| Gösterge | Max Fark |
|----------|----------|
| ARS | ~0.01 |
| Momentum, HHV/LLV | 0.00 |
| Volume HHV/LLV | 0.00 |
| MFI | 0.005 |
| ATR | 0.0001 |
| OBV / ADL | 0.00 (Kümülatif fix) |
| Aroon / Stoch | 0.00 (Formül fix) |
| ARS_Dynamic | 0.00 (Yuvarlama fix) |

</details>

### Strateji Validasyonu (✅ TAMAMLANDI)
- [x] Strateji 1 Python Portu: `score_based.py` (Gatekeeper)
- [x] Strateji 2 Python Portu: `ars_trend_v2.py` (Trend)
- [x] Strateji 3 Python Portu: `paradise_strategy.py` (HH/LL Breakout + Momentum)
- [x] Strateji 4 Python Portu: `toma_strategy.py` (TOMA + Momentum)
- [x] IdealData Kaynak Kodları: `S1`, `S2`, `Paradise`, `TOMA_S4`
- [x] **v4.2 Uyumu:** Tüm stratejiler (S1-S4) senkronize edildi, cache desteği ve C# export eklendi.
- [x] **Numba Optimizasyonu:** Tüm backtest motorları `jit` ile 100x hızlandırıldı.

---

## 🔄 FAZ 3: Robust Parametre Seçimi [AKTİF]

> [!IMPORTANT]
> Bu faz overfitting'i tespit edip güvenli parametreleri belirler.

### 3.1 Walk-Forward Analiz ✅
- [x] `src/robust/walk_forward.py` oluşturuldu
- [x] In-sample / Out-of-sample bölme
- [x] Rolling window implementasyonu
- [x] WFA skoru hesaplama

### 3.2 Monte Carlo Simülasyonu ✅
- [x] `src/robust/monte_carlo.py` oluşturuldu
- [x] Trade shuffle (1000 simülasyon)
- [x] %95 Confidence interval
- [x] Risk of Ruin hesaplama

---

## ✅ FAZ 4: IdealData Entegrasyonu [TAMAMLANDI]

### 4.1 Binary Parser ✅
- [x] `src/data/ideal_parser.py` - .01 dosyalarını okur
- [x] 32-byte record format çözüldü
- [x] Tüm periyotlar destekleniyor (1dk, 5dk, 60dk, G)

### 4.2 Kod Export ✅
- [x] `src/export/idealdata_exporter.py`
- [x] Strateji 1 + 2 kod üretimi
- [x] Birleşik robot kodu
- [x] Sistematik dosya isimlendirme
- [ ] Isı haritası

---

## ⏸️ FAZ 4: IdealData Dosya Yapısı

> CSV'ye gerek kalmadan direkt binary okuma.

### 4.1 Binary Analiz
- [ ] IdealData dosya formatı reverse engineering
- [ ] `src/engine/ideal_reader.py` oluştur
- [ ] OHLCV direkt okuma

---

## ⏸️ FAZ 5: Veritabanı Entegrasyonu

### 5.1 SQLite Şema
- [ ] `src/database/` modül oluştur
- [ ] OHLCV tabloları
- [ ] Optimizasyon sonuç tabloları
- [ ] CRUD operasyonları

---

## ⏸️ FAZ 6: Validation Modülü

### 6.1 İndikatör Karşılaştırma
- [ ] `src/validation/` modül oluştur
- [ ] Otomatik indikatör doğrulama
- [ ] Backtest karşılaştırma raporları

---

## 🔄 FAZ 7: Agent Dokümantasyonu [SÜREKLİ]

### 7.1 AI Kuralları
- [x] `CLAUDE.md` güncellendi (Gemini 3 Pro uyumu)
- [ ] Yeni workflow dosyaları

---

## ⏸️ FAZ 8: Uygulama Arayüzü [SON ADIM]

> [!IMPORTANT]
> AI'ya ihtiyaç duymadan tek başına kullanılabilen uygulama.

### 8.1 CLI (Command Line Interface)
- [ ] `python -m idealquant optimize --strategy X`
- [ ] `python -m idealquant wfa --strategy X`
- [ ] `python -m idealquant mc --simulations 1000`

### 8.2 Web UI (Streamlit)
- [ ] Parametre grid tanımlama (slider'larla)
- [ ] Tek tıkla optimizasyon
- [ ] İnteraktif sonuç grafikleri
- [ ] Walk-Forward & Monte Carlo dashboard

---

## 📅 Zaman Çizelgesi

```
Cuma       00:55  ─┬─ FAZ 3 Başlangıç (Walk-Forward)
              ↓   │
Cumartesi  12:00  ─┼─ FAZ 3 Monte Carlo
              ↓   │
Cumartesi  18:00  ─┼─ FAZ 3 Stabilite
              ↓   │
Cumartesi  24:00  ─┼─ FAZ 4 IdealData Decompile
              ↓   │
Pazar      12:00  ─┼─ FAZ 5 Veritabanı
              ↓   │
Pazar      18:00  ─┼─ FAZ 6 Validation
              ↓   │
Pazar      24:00  ─┴─ DEADLINE ✓
```

---

## 🔗 İlgili Dosyalar

- [Implementation Plan](../.gemini/antigravity/brain/current/implementation_plan.md)
- [Günlük](DEVLOG.md)
- [AI Kuralları](.agent/CLAUDE.md)
- [Workflows](.agent/workflows/)
