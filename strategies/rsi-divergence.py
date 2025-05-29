import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_peaks_and_troughs(data, window=5):
    """
    Find local peaks and troughs in a series
    """
    peaks = []
    troughs = []
    
    for i in range(window, len(data) - window):
        # Check if current point is a peak
        if all(data.iloc[i] >= data.iloc[i-j] for j in range(1, window+1)) and \
           all(data.iloc[i] >= data.iloc[i+j] for j in range(1, window+1)):
            peaks.append(i)
        
        # Check if current point is a trough
        if all(data.iloc[i] <= data.iloc[i-j] for j in range(1, window+1)) and \
           all(data.iloc[i] <= data.iloc[i+j] for j in range(1, window+1)):
            troughs.append(i)
    
    return peaks, troughs


def detect_divergence(price_peaks, price_troughs, rsi_peaks, rsi_troughs, price_data, rsi_data):
    """
    Detect bullish and bearish divergences
    """
    bullish_divergence = []
    bearish_divergence = []
    
    # Bullish divergence: Price makes lower lows, RSI makes higher lows
    for i in range(1, len(price_troughs)):
        if i < len(rsi_troughs):
            curr_price_trough = price_troughs[i]
            prev_price_trough = price_troughs[i-1]
            curr_rsi_trough = rsi_troughs[i] if i < len(rsi_troughs) else None
            prev_rsi_trough = rsi_troughs[i-1] if i-1 < len(rsi_troughs) else None
            
            if curr_rsi_trough and prev_rsi_trough:
                # Price lower low, RSI higher low
                if (price_data.iloc[curr_price_trough] < price_data.iloc[prev_price_trough] and
                    rsi_data.iloc[curr_rsi_trough] > rsi_data.iloc[prev_rsi_trough]):
                    bullish_divergence.append(curr_price_trough)
    
    # Bearish divergence: Price makes higher highs, RSI makes lower highs
    for i in range(1, len(price_peaks)):
        if i < len(rsi_peaks):
            curr_price_peak = price_peaks[i]
            prev_price_peak = price_peaks[i-1]
            curr_rsi_peak = rsi_peaks[i] if i < len(rsi_peaks) else None
            prev_rsi_peak = rsi_peaks[i-1] if i-1 < len(rsi_peaks) else None
            
            if curr_rsi_peak and prev_rsi_peak:
                # Price higher high, RSI lower high
                if (price_data.iloc[curr_price_peak] > price_data.iloc[prev_price_peak] and
                    rsi_data.iloc[curr_rsi_peak] < rsi_data.iloc[prev_rsi_peak]):
                    bearish_divergence.append(curr_price_peak)
    
    return bullish_divergence, bearish_divergence


# Load data
ticker = 'JPM'
print("Downloading data...")
df = yf.download(f'{ticker}', start='2020-01-01', end='2025-01-01')

# Check if data was downloaded successfully
if df is None or df.empty:
    raise ValueError("Failed to download data")

# Flatten the MultiIndex columns
df.columns = df.columns.droplevel(1)
df = df.reset_index()

# Calculate RSI
df['RSI'] = calculate_rsi(df['Close'])

# Find peaks and troughs
price_peaks, price_troughs = find_peaks_and_troughs(df['Close'])
rsi_peaks, rsi_troughs = find_peaks_and_troughs(df['RSI'])

# Detect divergences
bullish_div, bearish_div = detect_divergence(
    price_peaks, price_troughs, rsi_peaks, rsi_troughs, df['Close'], df['RSI']
)

# Create trading signals
df['Buy_Signal'] = False
df['Sell_Signal'] = False

# Mark divergence points
for idx in bullish_div:
    df.loc[idx, 'Buy_Signal'] = True

for idx in bearish_div:
    df.loc[idx, 'Sell_Signal'] = True

# Add RSI overbought/oversold levels as additional signals
df['RSI_Oversold'] = df['RSI'] < 30
df['RSI_Overbought'] = df['RSI'] > 70

# Combine divergence with RSI levels for stronger signals
df['Strong_Buy'] = df['Buy_Signal'] & df['RSI_Oversold']
df['Strong_Sell'] = df['Sell_Signal'] & df['RSI_Overbought']

# Create subplot figure
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(f'{ticker} Price with RSI Divergence', 'RSI'),
    row_heights=[0.7, 0.3]
)

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f'{ticker}'
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
        marker=dict(symbol='triangle-up', size=15, color='green'),
        text='BUY',
        textposition='bottom center',
        name='Bullish Divergence'
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
        marker=dict(symbol='triangle-down', size=15, color='red'),
        text='SELL',
        textposition='top center',
        name='Bearish Divergence'
    ),
    row=1, col=1
)

# Add RSI
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ),
    row=2, col=1
)

# Add RSI levels
fig.add_shape(
    type="line",
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
    y0=70, y1=70,
    line=dict(dash="dash", color="red"),
    row=2, col=1
)
fig.add_shape(
    type="line", 
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1],
    y0=30, y1=30,
    line=dict(dash="dash", color="green"),
    row=2, col=1
)
fig.add_shape(
    type="line",
    x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], 
    y0=50, y1=50,
    line=dict(dash="dot", color="gray"),
    row=2, col=1
)

fig.update_layout(
    title=f'{ticker} - RSI Divergence Strategy',
    xaxis_rangeslider_visible=False,
    height=800
)

fig.show()

# Backtest the strategy
initial_capital = 10000
df['Position'] = 0
current_position = 0

# Generate position signals based on strong signals
for i in range(len(df)):
    if df['Strong_Buy'].iloc[i] and current_position == 0:
        current_position = 1  # Enter long position
    elif df['Strong_Sell'].iloc[i] and current_position == 1:
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

print(f"\n=== RSI Divergence Strategy Performance ===")
print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
print(f"Number of trades: {df['Position'].diff().abs().sum()}")

# Create portfolio performance chart
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df['Date'], 
    y=df['Portfolio_Value'], 
    name='RSI Divergence Strategy',
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
    title=f'{ticker} - RSI Divergence Strategy vs Buy & Hold',
    xaxis_title='Date',
    yaxis_title='Portfolio Value ($)',
    xaxis_rangeslider_visible=False
)

fig2.show() 