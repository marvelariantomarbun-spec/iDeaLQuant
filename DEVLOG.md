# 📓 IdealQuant - Geliştirme Günlüğü

---

## 2026-01-25 (Cumartesi)

### ✅ Yapılanlar
- Proje dokümantasyonu güncellendi
- `ROADMAP.md` proje klasörüne eklendi
- `DEVLOG.md` günlük dosyası oluşturuldu

### 📌 Mevcut Durum
- **Aktif Faz:** Faz 1 - IdealData Uyumu
- **Sıradaki Adım:** Adım 1.1 - Veri Uyumu Doğrulama
- **Bekleyen:** IdealData'dan CSV export

### 🎯 Yarın için Plan
- [ ] IdealData'dan F_XU030 verisi export
- [ ] Veri okuma testi
- [ ] Bar-by-bar karşılaştırma

---

## 2026-01-24 (Cuma)

### ✅ Yapılanlar
- `src/engine/data.py` tamamlandı
  - OHLCV veri yapıları
  - IdealData CSV okuyucu
  - Liste() fonksiyonu
- `src/indicators/core.py` tamamlandı
  - Moving Averages: SMA, EMA, WMA, HullMA
  - Oscillators: RSI, Momentum, Stochastic
  - Volatility: ATR, Bollinger Bands
  - Trend: ADX
  - Custom: ARS, ARS_Dynamic, Qstick, RVI

### 📌 Notlar
- Numba import edildi ama henüz JIT optimizasyonu yok
- ATR ve RSI Wilder smoothing kullanıyor (IdealData uyumlu)

---

## 2026-01-23 (Perşembe)

### ✅ Yapılanlar
- Proje başlatıldı
- Git repo oluşturuldu
- Temel klasör yapısı kuruldu
- Yol haritası planlandı

### 📌 Karar
- IdealData ile %100 uyum öncelikli
- Optimizasyona geçmeden önce doğrulama şart

---

<!-- 
ŞABLON - Yeni gün için kopyala:

## 2026-XX-XX (Gün)

### ✅ Yapılanlar
- 

### 🐛 Hatalar / Sorunlar
- 

### 📌 Notlar
- 

### 🎯 Yarın için Plan
- [ ] 

-->
