# Trading Strategies

A collection of algorithmic trading strategies implemented in Python for backtesting and analysis.

## Current Strategies

### SMA Crossover Strategy
- **File**: `strategies/sma-crossover.py`
- **Description**: Simple Moving Average crossover strategy using 50-day and 200-day SMAs
- **Features**:
  - Golden Cross (50 SMA crosses above 200 SMA) - Buy signal
  - Death Cross (50 SMA crosses below 200 SMA) - Sell signal
  - Interactive candlestick charts with Plotly
  - Portfolio backtesting with $10,000 initial capital
  - Visual markers for entry/exit points

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run any strategy directly:
```bash
python strategies/sma-crossover.py
```

The script will:
- Download historical data from Yahoo Finance
- Calculate technical indicators
- Generate interactive charts
- Display backtest results

## Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance data download
- `plotly` - Interactive charting

## Future Strategies

Planned implementations:
- RSI divergence strategy
- Bollinger Bands mean reversion
- MACD momentum strategy
- Volume-weighted average price (VWAP)
- Pairs trading
- Options strategies

## Data Sources

- Yahoo Finance (via yfinance library)
- Historical price data from 2020-present

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions. 