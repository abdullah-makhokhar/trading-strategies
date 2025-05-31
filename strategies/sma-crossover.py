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

# Visualize and understand the data
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df.info())
# print(df.describe())

# Create figure
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name=f'{TICKER}'
))

fig.update_layout(
    title=f'{TICKER} Stock Price',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    xaxis_rangeslider_visible=False
)

# Calculate SMA
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Plot the data
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA_50'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name='SMA_200'))
#fig.show()

# Calculate the strategy
# Calculate crossover points
df['Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
signal_changes = df['Signal'].diff()
crossover_up = df[signal_changes == 1]
crossover_down = df[signal_changes == -1]

# Add up arrows for golden cross (50 crosses above 200)
fig.add_trace(go.Scatter(
    x=crossover_up['Date'],
    y=crossover_up['Close'],
    mode='markers+text',
    marker=dict(symbol='triangle-up', size=15, color='green'),
    text='↑',
    textposition='top center',
    name='Golden Cross'
))

# Add down arrows for death cross (50 crosses below 200) 
fig.add_trace(go.Scatter(
    x=crossover_down['Date'],
    y=crossover_down['Close'],
    mode='markers+text',
    marker=dict(symbol='triangle-down', size=15, color='red'),
    text='↓',
    textposition='bottom center',
    name='Death Cross'
))

fig.show()

# Backtest the strategy with realistic trading costs
df['Position'] = df['Signal'].shift(1)  # Shift by 1 to avoid look-ahead bias
df['Position'] = df['Position'].fillna(0)  # Fill first row with 0

# Calculate holdings and portfolio value with trading costs
df['Holdings'] = 0.0
df['Cash'] = float(INITIAL_CAPITAL)
df['Transaction_Costs'] = 0.0  # Track total transaction costs
shares_held = 0
total_transaction_costs = 0

# Update cash based on trades
for i in range(1, len(df)):
    # If position changed, calculate trade
    if df['Position'].iloc[i] != df['Position'].iloc[i-1]:
        # Buy signal
        if df['Position'].iloc[i] == 1:
            # Apply trading costs to buy price
            buy_price = apply_trading_costs(df['Close'].iloc[i], is_buy=True)
            shares_held = int(df['Cash'].iloc[i-1] // buy_price)  # Integer division for whole shares
            trade_cost = shares_held * buy_price
            total_transaction_costs += trade_cost - (shares_held * df['Close'].iloc[i])
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] - trade_cost
            df.loc[df.index[i:], 'Holdings'] = shares_held * df['Close'].iloc[i:]
        # Sell signal
        else:
            # Apply trading costs to sell price
            sell_price = apply_trading_costs(df['Close'].iloc[i-1], is_buy=False)
            trade_value = shares_held * sell_price
            total_transaction_costs += (shares_held * df['Close'].iloc[i-1]) - trade_value
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] + trade_value
            df.loc[df.index[i:], 'Holdings'] = 0
            shares_held = 0
    else:
        # No position change, update holdings value and carry forward cash
        df.loc[df.index[i], 'Holdings'] = shares_held * df['Close'].iloc[i]
        df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
    
    df.loc[df.index[i], 'Transaction_Costs'] = total_transaction_costs

# Calculate total portfolio value
df['Portfolio_Value'] = df['Holdings'] + df['Cash']

# Performance metrics
total_return = (df['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
num_trades = df['Position'].diff().abs().sum()

print(f"\n=== SMA Crossover Strategy Performance (with Trading Costs) ===")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
print(f"Number of trades: {num_trades}")
print(f"Total Transaction Costs: ${total_transaction_costs:,.2f}")
print(f"Transaction Costs as % of Initial Capital: {(total_transaction_costs/INITIAL_CAPITAL)*100:.2f}%")

# Calculate additional metrics
df['Strategy_Returns'] = df['Portfolio_Value'].pct_change()
strategy_volatility = df['Strategy_Returns'].std() * np.sqrt(252) * 100  # Annualized
sharpe_ratio = (total_return / 100) / (strategy_volatility / 100) if strategy_volatility > 0 else 0

print(f"Strategy Volatility (annualized): {strategy_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot the portfolio value
fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Portfolio_Value'], 
    name=f'Portfolio Value (Initial ${INITIAL_CAPITAL:,.0f})',
    line=dict(color='purple', width=2)
))

# Add buy & hold comparison
initial_shares = INITIAL_CAPITAL / df['Close'].iloc[0]
df['Buy_Hold_Value'] = initial_shares * df['Close']

fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Buy_Hold_Value'], 
    name='Buy & Hold',
    line=dict(color='gray', dash='dash')
))

fig.update_layout(
    title=f'{TICKER} - SMA Crossover Strategy vs Buy & Hold (with Trading Costs)',
    xaxis_title='Date',
    yaxis_title='Price/Value ($)',
    xaxis_rangeslider_visible=False
)

fig.show()
