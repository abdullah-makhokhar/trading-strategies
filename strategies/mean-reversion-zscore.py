import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to import config and data fetcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKER, START_DATE, END_DATE, INITIAL_CAPITAL
from data_fetcher import fetch_stock_data

# Trading costs configuration
TRANSACTION_COST_RATE = 0.003  # 0.3% per trade (includes commission + spread)
SLIPPAGE_RATE = 0.001  # 0.1% average slippage

def apply_trading_costs(price, is_buy=True):
    """
    Apply realistic trading costs including transaction fees and slippage
    
    Args:
        price: The market price
        is_buy: True for buy orders, False for sell orders
    
    Returns:
        Adjusted price after costs
    """
    # Random slippage between 0% and 2x average slippage
    slippage = np.random.uniform(0, 2 * SLIPPAGE_RATE)
    
    if is_buy:
        # For buys: pay transaction cost + slippage (higher price)
        adjusted_price = price * (1 + TRANSACTION_COST_RATE + slippage)
    else:
        # For sells: pay transaction cost + slippage (lower price)
        adjusted_price = price * (1 - TRANSACTION_COST_RATE - slippage)
    
    return adjusted_price


def calculate_zscore(prices, window=20):
    """
    Calculate Z-Score for mean reversion
    Z-Score = (Current Price - Rolling Mean) / Rolling Standard Deviation
    
    Z-Score interpretation:
    > +2: Extremely overbought (sell signal)
    > +1: Overbought
    -1 to +1: Normal range
    < -1: Oversold  
    < -2: Extremely oversold (buy signal)
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    zscore = (prices - rolling_mean) / rolling_std
    return zscore, rolling_mean, rolling_std


def calculate_bollinger_position(prices, window=20, std_dev=2):
    """
    Calculate position within Bollinger Bands for additional confirmation
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    # Position: 0 = at lower band, 1 = at upper band
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position, upper_band, lower_band


# Load data using centralized fetcher with fallback options
print("Downloading data...")
df = fetch_stock_data(TICKER, START_DATE, END_DATE)

# Check if data was downloaded successfully
if df is None or df.empty:
    raise ValueError("Failed to download data from all sources")

# Data is already in the correct format from our fetcher
# No need to flatten MultiIndex columns or reset index

# Calculate Z-Score and related metrics
window = 20  # 20-day lookback period
df['ZScore'], df['Rolling_Mean'], df['Rolling_Std'] = calculate_zscore(df['Close'], window)

# Calculate Bollinger Band position for additional confirmation
df['BB_Position'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_position(df['Close'], window)

# Calculate additional technical indicators
df['SMA_50'] = df['Close'].rolling(window=50).mean()  # Longer-term trend
df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()  # Volume confirmation

# Generate trading signals based on Z-Score
# Conservative approach: Only trade at extreme levels
df['Extreme_Oversold'] = df['ZScore'] <= -2.0  # Very strong buy signal
df['Extreme_Overbought'] = df['ZScore'] >= 2.0  # Very strong sell signal

# Moderate approach: Trade at moderate levels
df['Oversold'] = df['ZScore'] <= -1.5
df['Overbought'] = df['ZScore'] >= 1.5

# Entry signals: Cross into extreme territory
df['Buy_Signal'] = (df['ZScore'] <= -1.5) & (df['ZScore'].shift(1) > -1.5)
df['Sell_Signal'] = (df['ZScore'] >= 1.5) & (df['ZScore'].shift(1) < 1.5)

# Exit signals: Return to normal range
df['Exit_Long'] = (df['ZScore'] >= 0.5) & (df['ZScore'].shift(1) < 0.5)
df['Exit_Short'] = (df['ZScore'] <= -0.5) & (df['ZScore'].shift(1) > -0.5)

# Volume confirmation: Only trade if volume is above average
df['Volume_Confirmed'] = df['Volume'] > df['Volume_SMA']

# Strong signals: Z-Score + Volume confirmation
df['Strong_Buy'] = df['Buy_Signal'] & df['Volume_Confirmed']
df['Strong_Sell'] = df['Sell_Signal'] & df['Volume_Confirmed']

# Create subplot figure
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(f'{TICKER} Price with Mean Reversion Strategy', 'Z-Score', 'Volume'),
    row_heights=[0.5, 0.3, 0.2]
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

# Add rolling mean
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Rolling_Mean'],
        name=f'Rolling Mean ({window}d)',
        line=dict(color='blue', width=2)
    ),
    row=1, col=1
)

# Add Bollinger Bands for reference
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['BB_Upper'],
        name='Upper Band',
        line=dict(color='red', dash='dash', width=1),
        opacity=0.7
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['BB_Lower'],
        name='Lower Band',
        line=dict(color='red', dash='dash', width=1),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        opacity=0.7
    ),
    row=1, col=1
)

# Add buy signals
buy_signals = df[df['Strong_Buy']]
fig.add_trace(
    go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Close'],
        mode='markers+text',
        marker=dict(symbol='triangle-up', size=12, color='green'),
        text='BUY',
        textposition='bottom center',
        name='Z-Score Buy Signal'
    ),
    row=1, col=1
)

# Add sell signals
sell_signals = df[df['Strong_Sell']]
fig.add_trace(
    go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Close'],
        mode='markers+text',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        text='SELL',
        textposition='top center',
        name='Z-Score Sell Signal'
    ),
    row=1, col=1
)

# Add Z-Score
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['ZScore'],
        name='Z-Score',
        line=dict(color='purple')
    ),
    row=2, col=1
)

# Add Z-Score threshold lines
fig.add_shape(
    type="line",
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
    y0=2, y1=2,
    line=dict(dash="dash", color="red"),
    row=2, col=1
)
fig.add_shape(
    type="line",
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
    y0=-2, y1=-2,
    line=dict(dash="dash", color="green"),
    row=2, col=1
)
fig.add_shape(
    type="line",
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
    y0=0, y1=0,
    line=dict(dash="dot", color="gray"),
    row=2, col=1
)

# Add volume
fig.add_trace(
    go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker_color='lightblue',
        opacity=0.7
    ),
    row=3, col=1
)

# Add volume moving average
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Volume_SMA'],
        name='Volume SMA',
        line=dict(color='darkblue', width=2)
    ),
    row=3, col=1
)

fig.update_layout(
    title=f'{TICKER} - Mean Reversion Z-Score Strategy',
    xaxis_rangeslider_visible=False,
    height=900
)

fig.show()

# Backtest the strategy
initial_capital = INITIAL_CAPITAL
df['Position'] = 0
current_position = 0

# Generate position signals with exit logic
for i in range(len(df)):
    if df['Strong_Buy'].iloc[i] and current_position == 0:
        current_position = 1  # Enter long position
    elif df['Strong_Sell'].iloc[i] and current_position == 1:
        current_position = 0  # Exit long position
    elif df['Exit_Long'].iloc[i] and current_position == 1:
        current_position = 0  # Exit on return to normal
    
    df.loc[df.index[i], 'Position'] = current_position

# Shift position by 1 to avoid look-ahead bias
df['Position'] = df['Position'].shift(1).fillna(0)

# Calculate portfolio performance with realistic trading costs
df['Holdings'] = 0.0
df['Cash'] = float(initial_capital)
df['Transaction_Costs'] = 0.0  # Track total transaction costs
shares_held = 0
total_transaction_costs = 0

for i in range(1, len(df)):
    prev_position = df['Position'].iloc[i-1]
    curr_position = df['Position'].iloc[i]
    
    if curr_position != prev_position:
        if curr_position == 1:  # Buy signal
            # Apply trading costs to buy price
            buy_price = apply_trading_costs(df['Close'].iloc[i], is_buy=True)
            shares_held = int(df['Cash'].iloc[i-1] // buy_price)
            trade_cost = shares_held * buy_price
            total_transaction_costs += trade_cost - (shares_held * df['Close'].iloc[i])
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] - trade_cost
        else:  # Sell signal
            # Apply trading costs to sell price
            sell_price = apply_trading_costs(df['Close'].iloc[i-1], is_buy=False)
            trade_value = shares_held * sell_price
            total_transaction_costs += (shares_held * df['Close'].iloc[i-1]) - trade_value
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] + trade_value
            shares_held = 0
    
    # Update holdings value
    df.loc[df.index[i], 'Holdings'] = shares_held * df['Close'].iloc[i]
    df.loc[df.index[i], 'Transaction_Costs'] = total_transaction_costs
    if i < len(df) - 1:
        if curr_position == prev_position:
            df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

# Calculate total portfolio value
df['Portfolio_Value'] = df['Holdings'] + df['Cash']

# Performance metrics
total_return = (df['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
num_trades = df['Position'].diff().abs().sum()

print(f"\n=== Mean Reversion Z-Score Strategy Performance (with Trading Costs) ===")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
print(f"Number of trades: {num_trades:.0f}")
print(f"Total Transaction Costs: ${total_transaction_costs:,.2f}")
print(f"Transaction Costs as % of Initial Capital: {(total_transaction_costs/initial_capital)*100:.2f}%")

# Calculate additional metrics
df['Strategy_Returns'] = df['Portfolio_Value'].pct_change()
strategy_volatility = df['Strategy_Returns'].std() * np.sqrt(252) * 100  # Annualized
sharpe_ratio = (total_return / 100) / (strategy_volatility / 100) if strategy_volatility > 0 else 0

# Calculate maximum drawdown
rolling_max = df['Portfolio_Value'].expanding().max()
drawdown = (df['Portfolio_Value'] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

print(f"Strategy Volatility (annualized): {strategy_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")

# Create portfolio performance chart
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Portfolio_Value'], 
    name='Z-Score Mean Reversion',
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
    title=f'{TICKER} - Mean Reversion Z-Score Strategy vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    xaxis_rangeslider_visible=False
)

fig2.show() 