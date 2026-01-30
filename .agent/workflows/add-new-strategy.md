---
description: Yeni bir alım-satım stratejisi ekleme adımları
---

# Yeni Strateji Ekleme Workflow

1. **Python Implementasyonu**
   - `src/strategies/` altında yeni `.py` dosyası oluştur.
   - `StrategyConfig` dataclass'ı tanımla.
   - `indicators/core.py`'den gerekli indikatörleri import et.
   - `get_signal` metodunu implement et.

2. **IdealData Scripti**
   - Aynı mantığı içeren `.txt` dosyasını oluştur.
   - İndikatör formüllerinin birebir aynı olduğundan emin ol.

3. **Test Oluşturma**
   - `tests/test_new_strategy.py` oluştur.
   - Basit bir senaryo ile sinyal üretip üretmediğini, hata vermediğini test et.

4. **Sisteme Kayıt**
   - Eğer engine içinde bir registry varsa oraya ekle (şu an manuel import ediliyor).

5. **Dokümantasyon**
   - Stratejinin mantığını `DEVLOG.md` veya ayrı bir not dosyasına yaz.
