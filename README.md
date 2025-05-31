# Trading Strategies

A collection of algorithmic trading strategies implemented in Python for backtesting and analysis.

## Configuration

### Ticker Symbol Setup
All strategies use a centralized configuration system for the ticker symbol:

1. **Quick Setup**: Run the setup script to change the ticker for all strategies:
   ```bash
   python setup_ticker.py
   ```

2. **Manual Setup**: Edit the `config.txt` file directly:
   ```bash
   echo "AAPL" > config.txt
   ```

3. **Default**: If no configuration is found, strategies default to AAPL.

The configuration system includes:
- `config.txt` - Simple text file containing the ticker symbol
- `config.py` - Python module that reads the configuration
- Automatic fallback to parent directory for strategies in subfolders

### Data Source Configuration (Optional)
For enhanced reliability, you can configure Alpha Vantage as a fallback data source:

1. **Get a free API key** from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. **Copy the environment template**:
   ```bash
   cp env.example .env
   ```
3. **Add your API key** to the `.env` file:
   ```bash
   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
   ```

## Data Fetching System

The project uses a robust, multi-source data fetching system with automatic fallbacks:

### Data Sources (in order of priority):
1. **Yahoo Finance** (Primary) - Free, no API key required
2. **Alpha Vantage** (Fallback) - Requires free API key
3. **Sample Data** (Last Resort) - Generated realistic data for testing

### Features:
- **Automatic Fallback**: If Yahoo Finance is rate-limited, automatically tries Alpha Vantage
- **Rate Limit Handling**: Intelligent retry logic with exponential backoff
- **Consistent Format**: All data sources return the same format for seamless integration
- **Error Recovery**: Graceful handling of API failures and network issues

### Rate Limiting Protection:
- Yahoo Finance: Automatic retry with increasing delays
- Alpha Vantage: Respects API rate limits (5 calls/minute for free tier)
- Sample Data: Always available as final fallback

## Current Strategies

### SMA Crossover Strategy
- **File**: `strategies/sma-crossover.py`
- **Description**: Simple Moving Average crossover strategy using 50-day and 200-day SMAs
- **Features**:
  - Golden Cross (50 SMA crosses above 200 SMA) - Buy signal
  - Death Cross (50 SMA crosses below 200 SMA) - Sell signal
  - Interactive candlestick charts with Plotly
  - Portfolio backtesting with configurable initial capital
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
4. Configure your ticker symbol:
   ```bash
   python setup_ticker.py
   ```
5. (Optional) Set up Alpha Vantage API key for enhanced reliability:
   ```bash
   cp env.example .env
   # Edit .env file with your API key
   ```

## Usage

### Setting Up Ticker Symbol
Before running strategies, set your desired ticker symbol:
```bash
python setup_ticker.py
```

### Testing Data Sources
Test all data sources to ensure they're working:
```bash
python data_fetcher.py
```

### Running Strategies
Run any strategy directly:
```bash
python strategies/sma-crossover.py
python strategies/macd-strategy.py
python strategies/mean-reversion-zscore.py
python strategies/rsi-divergence.py
python strategies/bollinger-bands.py
```

The scripts will:
- Read ticker symbol from configuration
- Attempt to download data from Yahoo Finance
- Automatically fallback to Alpha Vantage if needed
- Use sample data as last resort
- Calculate technical indicators
- Generate interactive charts
- Display backtest results

### Configuration Options
The `config.py` module provides these configurable parameters:
- `TICKER` - Stock symbol (from config.txt)
- `START_DATE` - Backtest start date (default: '2020-01-01')
- `END_DATE` - Backtest end date (default: '2025-01-01')
- `INITIAL_CAPITAL` - Starting capital (default: $10,000)

## Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance data download
- `alpha_vantage` - Alpha Vantage API client
- `requests` - HTTP requests for API calls
- `plotly` - Interactive charting

## Future Strategies

Planned implementations:
- Volume-weighted average price (VWAP)
- Pairs trading
- Options strategies
- Momentum strategies (Williams %R, Stochastic)
- Breakout strategies (Donchian Channels)

## Data Sources

### Primary: Yahoo Finance
- Free, no API key required
- Real-time and historical data
- Comprehensive coverage of global markets

### Fallback: Alpha Vantage
- Free tier: 5 API calls per minute, 500 calls per day
- Premium tiers available for higher limits
- Get your free API key: https://www.alphavantage.co/support/#api-key

### Last Resort: Sample Data
- Realistic generated data for testing
- Always available when APIs fail
- Consistent with actual market patterns

## Troubleshooting

### Rate Limiting Issues
If you encounter "Too Many Requests" errors:
1. Wait a few minutes for rate limits to reset
2. Set up Alpha Vantage API key for fallback
3. The system will automatically use sample data if all sources fail

### API Key Setup
If Alpha Vantage isn't working:
1. Verify your API key is correct in the `.env` file
2. Check you haven't exceeded daily limits (500 calls/day for free tier)
3. Ensure the `.env` file is in the project root directory

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions. 