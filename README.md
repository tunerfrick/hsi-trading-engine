# hsi-trading-engine
Python trading engine for Hang Seng Index stocks with SMA crossover strategy and Markowitz portfolio optimization


## Features

- **Automated HSI Constituent Fetching**: Dynamically scrapes top Hang Seng Index stocks from Yahoo Finance
- **SMA Crossover Strategy**: Implements simple moving average crossover signals (customizable windows)
- **Markowitz Portfolio Optimization**: Uses covariance matrix and linear algebra for optimal asset allocation
- **Transaction Costs**: Realistic modeling with 0.1% commission and 0.05% slippage
- **Comprehensive Risk Metrics**:
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Calmar Ratio
- **Live Monitoring**: Paper trading mode for real-time signal monitoring
- **Visualization**: Equity curve and drawdown charts

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/tunerfrick/hsi-trading-engine.git
cd hsi-trading-engine

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Backtest

```python
from hsi_trading_engine import EnhancedTradingEngine

# Initialize engine
engine = EnhancedTradingEngine(
    initial_cash=100000,
    short_window=10,
    long_window=30
)

# Fetch data and run backtest
engine.fetch_data(period='1y')
engine.generate_signals()
portfolio, metrics = engine.backtest(use_markowitz=True)
engine.plot(portfolio, metrics)
```

### Live Monitoring

```python
# Live paper trading monitor
engine_live = EnhancedTradingEngine(short_window=5, long_window=20)
engine_live.fetch_data(period='1y', live=True)
engine_live.generate_signals()
engine_live.monitor_live()
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|--------|
| `initial_cash` | Starting portfolio value (HKD) | 100000 |
| `short_window` | Short SMA period (days) | 10 |
| `long_window` | Long SMA period (days) | 30 |
| `use_markowitz` | Enable Markowitz optimization | True |

## Project Structure

```
hsi-trading-engine/
├── hsi_trading_engine.py    # Main trading engine code
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## Performance Metrics

The engine calculates the following metrics:

- **Total Return (%)**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown (%)**: Largest peak-to-trough decline

## Technical Details

### Strategy
- **Signal Generation**: Buy when short SMA > long SMA, sell when short SMA < long SMA
- **Position Sizing**: Markowitz optimization allocates based on covariance matrix
- **Costs**: 0.5% commission per trade + 0.05% slippage

### Data Source
- Yahoo Finance via `yfinance` library
- HSI constituents dynamically fetched

## Limitations

- Historical backtest only (no live trading execution)
- Assumes perfect order fills at close prices
- Does not account for survivorship bias
- Limited to top 20 HSI constituents by default

## Future Enhancements

- [ ] Add more strategies (RSI, MACD, Bollinger Bands)
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Integration with broker APIs
- [ ] Real-time data streaming

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Author

**vikram(duh)** - physics × aida student, polyu  

## Acknowledgments

- Markowitz Portfolio Theory
- Yahoo Finance API
- Python quant finance community
