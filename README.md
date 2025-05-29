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

### MACD Strategy
- **File**: `strategies/macd-strategy.py`
- **Description**: Moving Average Convergence Divergence momentum strategy using exponential moving averages
- **Features**:
  - MACD line crossover signals (12, 26, 9 periods)
  - Signal line and histogram analysis
  - Zero line crossover confirmation
  - Multiple chart panels showing price, MACD, and histogram
  - EMA trend confirmation with 12 and 26 period lines

### Mean Reversion (Z-Score) Strategy
- **File**: `strategies/mean-reversion-zscore.py`
- **Description**: Statistical mean reversion strategy using Z-Score to identify overbought/oversold conditions
- **Features**:
  - Z-Score calculation with 20-day rolling window
  - Buy signals at Z-Score ≤ -1.5 (oversold)
  - Sell signals at Z-Score ≥ 1.5 (overbought)
  - Bollinger Bands integration for additional confirmation
  - Volume confirmation using moving average filter

### RSI Divergence Strategy
- **File**: `strategies/rsi-divergence.py`
- **Description**: Relative Strength Index divergence strategy to identify potential trend reversals
- **Features**:
  - Bullish divergence detection (price lower lows, RSI higher lows)
  - Bearish divergence detection (price higher highs, RSI lower highs)
  - RSI overbought (>70) and oversold (<30) level confirmation
  - Peak and trough analysis for divergence identification
  - Combined signals for higher probability trades

### Bollinger Bands Strategy
- **File**: `strategies/bollinger-bands.py`
- **Description**: Volatility-based mean reversion strategy using Bollinger Bands (20-period, 2 std dev)
- **Features**:
  - Buy signals when price touches lower band (oversold)
  - Sell signals when price touches upper band (overbought)
  - Bollinger Band position calculation for precise entry/exit
  - Adaptive volatility bands that adjust to market conditions
  - Visual band representation with price action

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
python strategies/macd-strategy.py
python strategies/mean-reversion-zscore.py
python strategies/rsi-divergence.py
python strategies/bollinger-bands.py
```

The scripts will:
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
- Volume-weighted average price (VWAP)
- Pairs trading
- Options strategies
- Momentum strategies (Williams %R, Stochastic)
- Breakout strategies (Donchian Channels)

## Data Sources

- Yahoo Finance (via yfinance library)
- Historical price data from 2020-present
- Default ticker: JPM (configurable in each strategy)

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions. 