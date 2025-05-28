import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


# Load data
ticker = 'JPM'
print("Downloading data...")
df = yf.download(f'{ticker}', start='2020-01-01', end='2025-01-01')

# Check if data was downloaded successfully
if df is None or df.empty:
    raise ValueError("Failed to download data")

# Flatten the MultiIndex columns
df.columns = df.columns.droplevel(1)  # Remove the ticker level
df = df.reset_index()  # Move the date index to a column

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
    name=f'{ticker}'
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
    title=f'{ticker} Stock Price with Bollinger Bands',
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
initial_capital = 10000  # Starting with $10,000
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

# Calculate portfolio performance
df['Holdings'] = 0.0
df['Cash'] = float(initial_capital)
shares_held = 0

for i in range(1, len(df)):
    prev_position = df['Position'].iloc[i-1]
    curr_position = df['Position'].iloc[i]
    
    # Position change occurred
    if curr_position != prev_position:
        if curr_position == 1:  # Buy signal
            shares_held = int(df['Cash'].iloc[i-1] // df['Close'].iloc[i])
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] - (shares_held * df['Close'].iloc[i])
        else:  # Sell signal
            df.loc[df.index[i:], 'Cash'] = df['Cash'].iloc[i-1] + (shares_held * df['Close'].iloc[i-1])
            shares_held = 0
    
    # Update holdings value
    df.loc[df.index[i], 'Holdings'] = shares_held * df['Close'].iloc[i]
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

print(f"\n=== Bollinger Bands Strategy Performance ===")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")

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
    title=f'{ticker} - Bollinger Bands Strategy vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    xaxis_rangeslider_visible=False
)

fig2.show() 