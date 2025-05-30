import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKER, START_DATE, END_DATE, INITIAL_CAPITAL

# Load data
print("Downloading data...")
df = yf.download(f'{TICKER}', start=START_DATE, end=END_DATE)

# Check if data was downloaded successfully
if df is None or df.empty:
    raise ValueError("Failed to download data")

# Flatten the MultiIndex columns
df.columns = df.columns.droplevel(1)  # Remove the 'AAPL' level
df = df.reset_index()  # Move the date index to a column

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

# Backtest the strategy
df['Position'] = df['Signal'].shift(1)  # Shift by 1 to avoid look-ahead bias
df['Position'] = df['Position'].fillna(0)  # Fill first row with 0

# Calculate holdings and portfolio value
df['Holdings'] = df['Position'] * df['Close']  # Value of stocks held
df['Cash'] = INITIAL_CAPITAL  # Initialize cash column

# Update cash based on trades
for i in range(1, len(df)):
    # If position changed, calculate trade
    if df['Position'].iloc[i] != df['Position'].iloc[i-1]:
        # Buy signal
        if df['Position'].iloc[i] == 1:
            shares = df['Cash'].iloc[i-1] // df['Close'].iloc[i]  # Integer division for whole shares
            df.loc[df.index[i:], 'Holdings'] = shares * df['Close']
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] - (shares * df['Close'].iloc[i])
        # Sell signal
        else:
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] + df['Holdings'].iloc[i-1]
            df.loc[df.index[i:], 'Holdings'] = 0

# Calculate total portfolio value
df['Portfolio_Value'] = df['Holdings'] + df['Cash']

# Plot the portfolio value
fig.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Portfolio_Value'], 
    name=f'Portfolio Value (Initial ${INITIAL_CAPITAL:,.0f})'
))
fig.show()
