# IdealQuant AI Assistant Rules

## 🎯 Temel Prensipler
Opus 4.5 (Thinking) seviyesinde titiz, planlı ve güvenli çalışmak esastır. Hata yapma lüksümüz yok.

---

## 📋 AKTİF FAZ KONTROLÜ

> [!IMPORTANT]
> Bu bölümü her işe başlamadan önce kontrol et!

| Faz | Durum | Ne Yapılabilir |
|-----|-------|----------------|
| Faz 1-2 | ✅ TAMAMLANDI | Sadece bug fix |
| Faz 3 | 🔄 AKTİF | Walk-Forward, Monte Carlo, Stabilite |
| Faz 4 | ⏸️ BEKLEMEDE | IdealData Decompile |
| Faz 5 | ⏸️ BEKLEMEDE | Veritabanı |
| Faz 6 | ⏸️ BEKLEMEDE | Validation Modülü |
| Faz 7 | 🔄 SÜREKLİ | Dokümantasyon |

**Şu an aktif görev:** `task.md` dosyasını kontrol et!

---

## ⛔ KESİNLİKLE YASAK OLANLAR (Strictly Forbidden)

1. **İzinsiz Mantık Değişimi:** Strateji sinyal mantığını (entry/exit koşulları) user onayı olmadan asla değiştirme.
2. **Parametre Sabitleme:** Kod içinde parametreleri hardcode etme (`3` yerine `self.config.ars_period` kullan).
3. **Manuel Optimizasyon:** Parametreleri kafana göre değiştirme, daima `smart_optimizer` sonuçlarını kullan.
4. **Veri Manipülasyonu:** `data/` klasöründeki hiçbir dosyayı silme veya değiştirme (yeni dosya ekle hariç).
5. **Eksik Test:** Bir kodu değiştirdikten sonra ilgili testi (`tests/`) çalıştırmadan "tamam" deme.
6. **Faz Atlama:** Aktif olmayan fazlardaki işlere başlama (yukarıdaki tabloya bak).
7. **ROADMAP Uyumsuzluğu:** `ROADMAP.md` ile çelişen iş yapma.

---

## ✅ ZORUNLU KURALLAR (Must Do)

1. **DEVLOG Kaydı:** Anlamlı her işin sonunda `DEVLOG.md` dosyasını güncelle.
2. **Task Güncellemesi:** İş başlamadan/bitince `task.md` güncelle.
3. **Sonuç Saklama:** Optimizasyon/test çıktılarını ASLA sadece ekrana basma, `results/` klasörüne kaydet.
4. **Test Çalıştırma:** Kod değişikliği sonrası: `python -m pytest tests/ -v`
5. **Plan Kontrolü:** `implementation_plan.md` dosyasını oku ve takip et.

---

## 📁 Dosya ve Klasör Yapısı

```
IdealQuant/
├── src/
│   ├── engine/         # Veri okuma, backtest core
│   ├── indicators/     # Core indikatör kütüphanesi
│   ├── strategies/     # Her strateji kendi dosyasında
│   ├── optimization/   # Grid search, genetic algo
│   ├── robust/         # Walk-forward, Monte Carlo, Stabilite
│   ├── database/       # SQLite repo (Faz 5)
│   └── validation/     # İndikatör/backtest karşılaştırma (Faz 6)
├── tests/              # PyTest testleri
├── results/            # CSV, JSON raporlar
├── data/               # OHLCV verileri (DOKUNMA!)
└── .agent/             # AI kuralları ve workflow'lar
```

---

## 🔄 Workflow Referansları

| Görev | Workflow Dosyası |
|-------|------------------|
| Strateji Optimize Etme | `.agent/workflows/optimize-strategy.md` |
| Yeni Strateji Ekleme | `.agent/workflows/add-new-strategy.md` |
| İndikatör Doğrulama | `.agent/workflows/validate-indicator.md` |
| Walk-Forward Analiz | `.agent/workflows/walk-forward.md` |
| Monte Carlo Simülasyonu | `.agent/workflows/monte-carlo.md` |
| Veritabanı İşlemleri | `.agent/workflows/database-ops.md` |

---

## 🧩 İndikatör Standartları

- Tüm indikatörler `src/indicators/core.py` içinde olmalı.
- Numba `@jit` ile hızlandırılmalı.
- IdealData ile %100 uyumlu olmalı (Wilder smoothing vb. dikkat).

---

## 🤖 Gemini 3 Pro Özel Talimatlar

> [!CAUTION]
> Bu bölüm Gemini 3 Pro için zorunlu kurallardır!

### Her İşe Başlamadan Önce:
1. `ROADMAP.md` oku - mevcut faz durumunu kontrol et
2. `task.md` oku - aktif görevi anla
3. `implementation_plan.md` oku - ne yapılacağını anla
4. Bu dosyayı (`CLAUDE.md`) oku - kuralları hatırla

### İş Sırasında:
1. Sadece **AKTİF** fazlardaki işleri yap
2. Her değişiklikten sonra test çalıştır
3. `DEVLOG.md` güncelle
4. Sonuçları `results/` klasörüne kaydet

### Asla Yapma:
1. ❌ Strateji entry/exit mantığını değiştirme
2. ❌ `data/` klasörüne dokunma
3. ❌ Beklemedeki fazlara başlama
4. ❌ Test çalıştırmadan "tamam" deme
5. ❌ Sonuçları sadece ekrana yazdırma

---

## 📊 SON OTURUM ÖZETİ (2026-01-31)

> [!IMPORTANT]
> Yeni sohbete başlarken önce `implementation_plan.md` dosyasını oku!

### 🎯 Karar Alınan Konular

**Strateji 1 (Gatekeeper) - 20 parametre, 6 grup:**
1. ARS (4p), ADX (3p), MACDV (3p), BB (4p) → Bağımsız
2. Hacim (2p), Skor (4p) → Kademeli

**Strateji 2 (Trend Takip) - 21 parametre, 4 grup:**
1. ARS Dinamik (5p), Breakout/Momentum (4p), MFI+Hacim (5p) → Bağımsız
2. Çıkış ATR + Çift Teyit (6p) → Kademeli

**Önemli Kararlar:**
- ✅ Grid Search + Genetik Algoritma PARALEL çalışacak
- ✅ RSI çıkarıldı, MFI Breakout eklendi
- ✅ Çoklu HHV/LLV (3 farklı periyot)
- ✅ ATR bazlı çıkış (SL, TP, Trailing)
- ✅ Trend dönüşü çift teyit (çoklu bar + ARS mesafesi)

### 🔧 Kalibrasyon Gerekli
- [ ] MFI (IdealData vs Python)
- [ ] Hacim HHV/LLV
- [ ] ATR (çıkış için)

### 📋 Sonraki Adımlar
1. Kalibrasyon
2. Optimizer kodları (Grid + GA)
3. WFA + Monte Carlo
4. Streamlit UI

> [!NOTE]
> Detaylar için: `implementation_plan.md` ve `task.md` dosyalarını oku!
