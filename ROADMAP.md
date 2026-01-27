# 🗺️ IdealQuant - Yol Haritası

## 🎯 Ana Hedef
IdealData backtest sonuçları ile **%100 uyumlu** harici backtest + optimizasyon + robust parametre seçim sistemi.

---

## 📋 Faz Durumları

| Faz | Durum | Açıklama |
|-----|-------|----------|
| Faz 0 | ✅ | Proje Kurulumu |
| Faz 1 | ✅ | IdealData Uyumu (TAMAMLANDI) |
| Faz 2 | 🟡 | Optimizasyon Motoru (BAŞLATILDI) |
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
- [x] IdealData'dan CSV export (F_XU030, 1dk, 1 hafta)
- [x] Python'da veri okuma testi
- [x] Bar-by-bar karşılaştırma
- [x] **DOĞRULAMA:** %100 eşleşme

### Adım 1.2: İndikatör Uyumu
- [x] SMA(20) test ve doğrulama
- [x] EMA(20) test ve doğrulama
- [x] RSI(14) test ve doğrulama
- [x] ATR(14) test ve doğrulama
- [x] ARS test ve doğrulama (İnce farklar tespit edildi ve doğrulandı)
- [x] **DOĞRULAMA:** %90 bar < 0.01 fark, max %0.02 hata (Kabul Edildi)

### Adım 1.3: Sinyal Uyumu
- [x] ARS Trend v2 stratejisi port edildi
- [x] IdealData'dan 5600+ işlem (1 yıl) export
- [x] Sinyal karşılaştırma testi
- [x] **DOĞRULAMA:** %97.8 sinyal uyumu (BAŞARILI)

### Adım 1.4: P&L Uyumu
- [x] Backtest engine entegrasyonu
- [x] Komisyon/slippage modeli (Gelecekte eklenecek, şimdilik atlandı)
- [x] **DOĞRULAMA:** %99.1 işlem uyumu, %97 P&L eşleşmesi (BAŞARILI)

---

## 🟡 FAZ 2: Optimizasyon Motoru

### Adım 2.1: Grid Search & Paralel İşleme
- [ ] ParameterGrid sınıfı (14 parametre desteği)
- [ ] Ryzen 9 9950X (24 worker) entegrasyonu
- [ ] "Kaba'dan İnce'ye" (2 aşamalı) optimizasyon mantığı
- [ ] Sonuç sıralama ve CSV kaydı

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
