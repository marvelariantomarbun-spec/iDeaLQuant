# IdealQuant

Trading Strategy Backtesting, Optimization, and Robust Parameter Selection Suite.

> 📍 **Hızlı Bağlantılar:** [Yol Haritası](ROADMAP.md) | [Geliştirme Günlüğü](DEVLOG.md)

## 🎯 Features

### 1. Backtest Engine
- IdealData compatible backtest
- Bar-by-bar simulation
- Multiple indicator support (ARS, RSI, QQEF, RVI, Qstick, etc.)
- Commission and slippage modeling

### 2. Optimization Engine
- Grid Search optimization
- Parallel processing (uses all CPU cores)
- 10-100x faster than IdealData

### 3. Robust Parameter Selector
- Walk-Forward Analysis
- Parameter Stability Scoring
- Monte Carlo Simulation
- Overfitting Detection

## 📁 Project Structure

```
IdealQuant/
├── src/
│   ├── engine/          # Backtest engine
│   ├── indicators/      # Technical indicators
│   ├── optimization/    # Optimization algorithms
│   ├── robust/          # Robust parameter selection
│   └── ui/              # Streamlit GUI
├── data/                # OHLCV data (CSV)
├── tests/               # Unit tests
├── reference/           # Reference code (IdealOptimizer)
└── requirements.txt
```

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run GUI
streamlit run src/ui/app.py
```

## 📊 Usage

1. Export OHLCV data from IdealData as CSV
2. Place in `data/` folder
3. Run optimization
4. Review Walk-Forward results
5. Select robust parameters
