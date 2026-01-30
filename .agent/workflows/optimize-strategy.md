---
description: Bir stratejinin parametrelerini optimize etme süreci
---

# Strateji Optimizasyon Workflow

1. **Veri Kontrolü**
   - Hedef sembol için verinin `data/` altında olduğundan emin ol.
   - Yoksa: IdealData'dan 1DK export iste.

2. **Optimizer Konfigürasyonu**
   - İlgili stratejinin optimizer dosyasını aç (örn: `src/optimization/smart_optimizer.py`).
   - `PARAM_GRID` değişkenini kontrol et ve gerekirse güncelle.
   - `search_space` mantıklı sınırlar içinde mi?

3. **Optimizasyonu Başlat**
   // turbo
   ```powershell
   python -m src.optimization.smart_optimizer
   ```
   *(Eğer sembol parametresi eklendiyse: `python -m src.optimization.smart_optimizer --symbol THYAO`)*

4. **Sonuçları Analiz Et**
   - `results/` klasörüne düşen CSV dosyasını kontrol et.
   - En iyi 20 sonucu incele (NP, DD, PF kriterlerine göre).
   - "Stabilite" (Stage 3) raporuna bak.

5. **Parametreleri Uygula**
   - Seçilen parametreleri strateji dosyasına (`src/strategies/StrategyName.py`) ve IdealData scriptine (`.txt`) işle.
   - `DEVLOG.md` dosyasına sonucu not düş.
