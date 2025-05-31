import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# Load data using centralized fetcher with fallback options
print("Downloading data...")
df = fetch_stock_data(TICKER, START_DATE, END_DATE)

# Check if data was downloaded successfully
if df is None or df.empty:
    raise ValueError("Failed to download data from all sources")

# Data is already in the correct format from our fetcher
# No need to flatten MultiIndex columns or reset index

# Calculate Bollinger Bands
window = 20  # Standard period for Bollinger Bands
std_dev = 2  # Standard deviation multiplier

df['BB_Middle'] = df['Close'].rolling(window=window).mean()  # Simple Moving Average
df['BB_Std'] = df['Close'].rolling(window=window).std()     # Rolling standard deviation
df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * std_dev)  # Upper band
df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * std_dev)  # Lower band

# Calculate Bollinger Band position (where price is relative to bands)
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

# Create figure
fig = go.Figure()

# Add candlestick chart
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name=f'{TICKER}'
))

# Add Bollinger Bands
fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['BB_Upper'], 
    name='BB Upper',
    line=dict(color='red', dash='dash')
))
fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['BB_Middle'], 
    name='BB Middle (SMA 20)',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['BB_Lower'], 
    name='BB Lower',
    line=dict(color='red', dash='dash'),
    fill='tonexty',  # Fill between upper and lower bands
    fillcolor='rgba(255, 0, 0, 0.1)'
))

fig.update_layout(
    title=f'{TICKER} Stock Price with Bollinger Bands',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    xaxis_rangeslider_visible=False
)

# Calculate trading signals
# Buy when price touches or goes below lower band (oversold)
# Sell when price touches or goes above upper band (overbought)
df['Buy_Signal'] = (df['Close'] <= df['BB_Lower']) & (df['Close'].shift(1) > df['BB_Lower'].shift(1))
df['Sell_Signal'] = (df['Close'] >= df['BB_Upper']) & (df['Close'].shift(1) < df['BB_Upper'].shift(1))

# Alternative: Use BB_Position for more nuanced signals
# Buy when BB_Position <= 0.1 (near lower band)
# Sell when BB_Position >= 0.9 (near upper band)
df['Buy_Signal_Alt'] = (df['BB_Position'] <= 0.1) & (df['BB_Position'].shift(1) > 0.1)
df['Sell_Signal_Alt'] = (df['BB_Position'] >= 0.9) & (df['BB_Position'].shift(1) < 0.9)

# Use the alternative signals for better performance
buy_signals = df[df['Buy_Signal_Alt']]
sell_signals = df[df['Sell_Signal_Alt']]

# Add buy signals (green up arrows)
fig.add_trace(go.Scatter(
    x=buy_signals['Date'],
    y=buy_signals['Close'],
    mode='markers+text',
    marker=dict(symbol='triangle-up', size=15, color='green'),
    text='BUY',
    textposition='bottom center',
    name='Buy Signal'
))

# Add sell signals (red down arrows)
fig.add_trace(go.Scatter(
    x=sell_signals['Date'],
    y=sell_signals['Close'],
    mode='markers+text',
    marker=dict(symbol='triangle-down', size=15, color='red'),
    text='SELL',
    textposition='top center',
    name='Sell Signal'
))

fig.show()

# Backtest the strategy
initial_capital = INITIAL_CAPITAL  # Starting with $10,000
df['Position'] = 0  # 0 = no position, 1 = long position

# Generate position signals
current_position = 0
for i in range(len(df)):
    if df['Buy_Signal_Alt'].iloc[i] and current_position == 0:
        current_position = 1  # Enter long position
    elif df['Sell_Signal_Alt'].iloc[i] and current_position == 1:
        current_position = 0  # Exit long position
    
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
    
    # Position change occurred
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
    if i < len(df) - 1:  # Don't overwrite cash for current row if it was just updated
        if curr_position == prev_position:
            df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

# Calculate total portfolio value
df['Portfolio_Value'] = df['Holdings'] + df['Cash']

# Calculate returns
df['Strategy_Returns'] = df['Portfolio_Value'].pct_change()
df['Buy_Hold_Returns'] = df['Close'].pct_change()

# Performance metrics
total_return = (df['Portfolio_Value'].iloc[-1] / initial_capital - 1) * 100
buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
num_trades = df['Position'].diff().abs().sum()

print(f"\n=== Bollinger Bands Strategy Performance (with Trading Costs) ===")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
print(f"Number of trades: {num_trades}")
print(f"Total Transaction Costs: ${total_transaction_costs:,.2f}")
print(f"Transaction Costs as % of Initial Capital: {(total_transaction_costs/initial_capital)*100:.2f}%")

# Calculate additional metrics
strategy_volatility = df['Strategy_Returns'].std() * np.sqrt(252) * 100  # Annualized
sharpe_ratio = (total_return / 100) / (strategy_volatility / 100) if strategy_volatility > 0 else 0

print(f"Strategy Volatility (annualized): {strategy_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Create a new figure for portfolio performance
fig2 = go.Figure()

# Add portfolio value
fig2.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Portfolio_Value'], 
    name=f'Bollinger Bands Strategy',
    line=dict(color='blue')
))

# Add buy & hold comparison
initial_shares = initial_capital / df['Close'].iloc[0]
df['Buy_Hold_Value'] = initial_shares * df['Close']

fig2.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Buy_Hold_Value'], 
    name=f'Buy & Hold',
    line=dict(color='gray', dash='dash')
))

fig2.update_layout(
    title=f'{TICKER} - Bollinger Bands Strategy vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    xaxis_rangeslider_visible=False
)

fig2.show() 