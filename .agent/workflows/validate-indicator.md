---
description: Python indikatör fonksiyonları ile IdealData sonuçlarını doğrulama
---

# İndikatör Validation Workflow

1. **Verify Verisi Oluştur**
   - IdealData'da ilgili indikatörü hesaplayan ve dosyaya yazan bir sistem oluştur.
   - Örn: `Sistem.YaziDosyasinaYaz("Test_RSI_14.txt", RSI)`

2. **Python Karşılaştırma Scripti**
   - `tests/find_matching_indicator.py` şablonunu kullan.
   - IdealData'dan çıkan dosyayı ve ham veri dosyasını oku.
   - Python fonksiyonunu çalıştır ve iki seti karşılaştır.

3. **Tolerans Kontrolü**
   - Hata payı (epsilon) < 1e-4 olmalı.
   - Eğer fark varsa: "Wilder Smoothing" mi "Simple/Exp MA" mı kullanıldığını kontrol et.

4. **Kütüphaneye Ekle**
   - Onaylanan fonksiyonu `src/indicators/core.py` içine taşı.
   - `@jit(nopython=True)` decorator ekle (hız için).
