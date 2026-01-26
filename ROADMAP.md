# 🗺️ IdealQuant - Yol Haritası

## 🎯 Ana Hedef
IdealData backtest sonuçları ile **%100 uyumlu** harici backtest + optimizasyon + robust parametre seçim sistemi.

---

## 📋 Faz Durumları

| Faz | Durum | Açıklama |
|-----|-------|----------|
| Faz 0 | ✅ | Proje Kurulumu |
| Faz 1 | 🔴 | IdealData Uyumu (KRİTİK) |
| Faz 2 | ⏳ | Optimizasyon Motoru |
| Faz 3 | ⏳ | Robust Parametre Seçici |

---

## ✅ FAZ 0: Proje Kurulumu [TAMAMLANDI]
- [x] Proje klasörü oluşturuldu
- [x] `src/engine/data.py` - OHLCV veri yapıları
- [x] `src/indicators/core.py` - 15+ indikatör
- [x] Git repo başlatıldı

---

## 🔴 FAZ 1: IdealData Uyumu (KRİTİK)

> [!IMPORTANT]
> Bu faz tamamlanmadan optimizasyona geçilmemeli. Her adımda %100 uyum doğrulaması şart.

### Adım 1.1: Veri Uyumu
- [ ] IdealData'dan CSV export (F_XU030, 1dk, 1 hafta)
- [ ] Python'da veri okuma testi
- [ ] Bar-by-bar karşılaştırma
- [ ] **DOĞRULAMA:** %100 eşleşme

### Adım 1.2: İndikatör Uyumu
- [ ] SMA(20) test ve doğrulama
- [ ] EMA(20) test ve doğrulama
- [ ] RSI(14) test ve doğrulama
- [ ] ATR(14) test ve doğrulama
- [ ] ARS test ve doğrulama
- [ ] **DOĞRULAMA:** <%1 fark

### Adım 1.3: Sinyal Uyumu
- [ ] Basit strateji yazılması (C > SMA → AL)
- [ ] IdealData'dan sinyal export
- [ ] Sinyal karşılaştırma
- [ ] **DOĞRULAMA:** Tüm sinyaller aynı bar

### Adım 1.4: P&L Uyumu
- [ ] Backtest engine yazılması
- [ ] İşlem simülasyonu
- [ ] Komisyon/slippage modeli
- [ ] **DOĞRULAMA:** <%1 toplam P&L farkı

---

## 🟡 FAZ 2: Optimizasyon Motoru

### Adım 2.1: Grid Search
- [ ] Parametre grid tanımı
- [ ] Brute force arama
- [ ] Sonuç sıralama

### Adım 2.2: Paralel İşleme
- [ ] Multiprocessing entegrasyonu
- [ ] 32 thread desteği
- [ ] İlerleme takibi

### Adım 2.3: Sonuç Kaydı
- [ ] CSV export
- [ ] SQLite opsiyonu
- [ ] Top-N filtreleme

---

## 🟢 FAZ 3: Robust Parametre Seçici

### Adım 3.1: Walk-Forward Analiz
- [ ] In-sample / Out-of-sample bölme
- [ ] Rolling window
- [ ] WFA skoru hesaplama

### Adım 3.2: Parametre Stabilite
- [ ] Komşu parametre analizi
- [ ] Stabilite skoru
- [ ] Isı haritası görselleştirme

### Adım 3.3: Overfitting Tespiti
- [ ] Monte Carlo simülasyonu (opsiyonel)
- [ ] Overfitting raporu
- [ ] Risk uyarıları

---

## 📅 Tahmini Süre

| Faz | Süre | Öncelik |
|-----|------|---------|
| Faz 1 | ~1 hafta | 🔴 Kritik |
| Faz 2 | 2-3 gün | 🟡 Önemli |
| Faz 3 | 2-3 gün | 🟢 Nice-to-have |

---

## 🔗 İlgili Dosyalar
- [Günlük](DEVLOG.md) - Günlük geliştirme notları
- [README](README.md) - Proje açıklaması
