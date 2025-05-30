import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKER, START_DATE, END_DATE, INITIAL_CAPITAL


def calculate_ema(prices, span):
    """
    Calculate Exponential Moving Average
    EMA = (Price * (2 / (span + 1))) + (Previous EMA * (1 - (2 / (span + 1))))
    """
    return prices.ewm(span=span).mean()


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD Line
    Histogram = MACD Line - Signal Line
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# Load data
print("Downloading data...")
df = yf.download(f'{TICKER}', start=START_DATE, end=END_DATE)

# Check if data was downloaded successfully
if df is None or df.empty:
    raise ValueError("Failed to download data")

# Flatten the MultiIndex columns
df.columns = df.columns.droplevel(1)
df = df.reset_index()

# Calculate MACD
df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df['Close'])

# Calculate additional EMAs for trend confirmation
df['EMA_12'] = calculate_ema(df['Close'], 12)
df['EMA_26'] = calculate_ema(df['Close'], 26)

# Generate trading signals
# Buy when MACD crosses above Signal line (bullish crossover)
df['MACD_Cross_Up'] = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))

# Sell when MACD crosses below Signal line (bearish crossover)
df['MACD_Cross_Down'] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))

# Additional confirmation: MACD above/below zero line
df['MACD_Above_Zero'] = df['MACD'] > 0
df['MACD_Below_Zero'] = df['MACD'] < 0

# Strong signals: MACD crossover + zero line confirmation
df['Strong_Buy'] = df['MACD_Cross_Up'] & df['MACD_Above_Zero']
df['Strong_Sell'] = df['MACD_Cross_Down'] & df['MACD_Below_Zero']

# Alternative: Use any crossover (more signals, potentially more noise)
df['Buy_Signal'] = df['MACD_Cross_Up']
df['Sell_Signal'] = df['MACD_Cross_Down']

# Create subplot figure
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(f'{TICKER} Price with MACD Strategy', 'MACD', 'MACD Histogram'),
    row_heights=[0.5, 0.25, 0.25]
)

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f'{TICKER}'
    ),
    row=1, col=1
)

# Add EMAs to price chart
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['EMA_12'],
        name='EMA 12',
        line=dict(color='blue', width=1)
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['EMA_26'],
        name='EMA 26',
        line=dict(color='red', width=1)
    ),
    row=1, col=1
)

# Add buy signals
buy_signals = df[df['Buy_Signal']]
fig.add_trace(
    go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Close'],
        mode='markers+text',
        marker=dict(symbol='triangle-up', size=12, color='green'),
        text='BUY',
        textposition='bottom center',
        name='MACD Buy Signal'
    ),
    row=1, col=1
)

# Add sell signals
sell_signals = df[df['Sell_Signal']]
fig.add_trace(
    go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Close'],
        mode='markers+text',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        text='SELL',
        textposition='top center',
        name='MACD Sell Signal'
    ),
    row=1, col=1
)

# Add MACD and Signal lines
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['MACD'],
        name='MACD Line',
        line=dict(color='blue')
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Signal'],
        name='Signal Line',
        line=dict(color='red')
    ),
    row=2, col=1
)

# Add zero line to MACD
fig.add_shape(
    type="line",
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
    y0=0, y1=0,
    line=dict(dash="dash", color="gray"),
    row=2, col=1
)

# Add MACD Histogram
colors = ['green' if x >= 0 else 'red' for x in df['Histogram']]
fig.add_trace(
    go.Bar(
        x=df['Date'],
        y=df['Histogram'],
        name='MACD Histogram',
        marker_color=colors,
        opacity=0.7
    ),
    row=3, col=1
)

fig.update_layout(
    title=f'{TICKER} - MACD Strategy',
    xaxis_rangeslider_visible=False,
    height=900
)

fig.show()

# Backtest the strategy
initial_capital = INITIAL_CAPITAL
df['Position'] = 0
current_position = 0

# Generate position signals
for i in range(len(df)):
    if df['Buy_Signal'].iloc[i] and current_position == 0:
        current_position = 1  # Enter long position
    elif df['Sell_Signal'].iloc[i] and current_position == 1:
        current_position = 0  # Exit long position
    
    df.loc[df.index[i], 'Position'] = current_position

# Shift position by 1 to avoid look-ahead bias
df['Position'] = df['Position'].shift(1).fillna(0)

# Calculate portfolio performance
df['Holdings'] = 0.0
df['Cash'] = float(initial_capital)
shares_held = 0

for i in range(1, len(df)):
    prev_position = df['Position'].iloc[i-1]
    curr_position = df['Position'].iloc[i]
    
    if curr_position != prev_position:
        if curr_position == 1:  # Buy signal
            shares_held = int(df['Cash'].iloc[i-1] // df['Close'].iloc[i])
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] - (shares_held * df['Close'].iloc[i])
        else:  # Sell signal
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] + (shares_held * df['Close'].iloc[i-1])
            shares_held = 0
    
    # Update holdings value
    df.loc[df.index[i], 'Holdings'] = shares_held * df['Close'].iloc[i]
    if i < len(df) - 1:
        if curr_position == prev_position:
            df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

# Calculate total portfolio value
df['Portfolio_Value'] = df['Holdings'] + df['Cash']

# Performance metrics
total_return = (df['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
num_trades = df['Position'].diff().abs().sum()

print(f"\n=== MACD Strategy Performance ===")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
print(f"Number of trades: {num_trades}")

# Calculate additional metrics
df['Strategy_Returns'] = df['Portfolio_Value'].pct_change()
strategy_volatility = df['Strategy_Returns'].std() * np.sqrt(252) * 100  # Annualized
sharpe_ratio = (total_return / 100) / (strategy_volatility / 100) if strategy_volatility > 0 else 0

print(f"Strategy Volatility (annualized): {strategy_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Create portfolio performance chart
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Portfolio_Value'], 
    name='MACD Strategy',
    line=dict(color='blue')
))

# Add buy & hold comparison
initial_shares = initial_capital / df['Close'].iloc[0]
df['Buy_Hold_Value'] = initial_shares * df['Close']

fig2.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Buy_Hold_Value'], 
    name='Buy & Hold',
    line=dict(color='gray', dash='dash')
))

fig2.update_layout(
    title=f'{TICKER} - MACD Strategy vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    xaxis_rangeslider_visible=False
)

fig2.show() 