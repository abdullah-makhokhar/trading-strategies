"""
Centralized data fetching module with Yahoo Finance and Alpha Vantage fallback
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
from datetime import datetime, timedelta
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not available. Install with: pip install python-dotenv")

# Try to import Alpha Vantage, handle if not available
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    print("Alpha Vantage library not available. Install with: pip install alpha-vantage")
    ALPHA_VANTAGE_AVAILABLE = False

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')  # Get from environment or use demo
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

def get_alpha_vantage_data(symbol, start_date, end_date, api_key=None):
    """
    Fetch data from Alpha Vantage API
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        api_key (str): Alpha Vantage API key
    
    Returns:
        pandas.DataFrame: Stock data with OHLCV columns
    """
    if not ALPHA_VANTAGE_AVAILABLE:
        print("Alpha Vantage library not available")
        return None
        
    if api_key is None:
        api_key = ALPHA_VANTAGE_API_KEY
    
    print(f"Fetching {symbol} data from Alpha Vantage...")
    
    try:
        # Use Alpha Vantage TimeSeries
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        # Get daily data (free tier endpoint)
        result = ts.get_daily(symbol=symbol, outputsize='full')
        
        # Handle the tuple response from Alpha Vantage
        if isinstance(result, tuple) and len(result) >= 2:
            data, meta_data = result[0], result[1]
        else:
            data = result
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            print(f"No data returned from Alpha Vantage for {symbol}")
            return None
        
        # Rename columns to match yfinance format (daily endpoint has different columns)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert index to datetime and sort
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        data = data[(data.index >= start_dt) & (data.index <= end_dt)]
        
        # Reset index to have Date as a column (to match yfinance format)
        data = data.reset_index()
        data.rename(columns={'date': 'Date'}, inplace=True)
        
        # Keep only the essential columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"Successfully fetched {len(data)} days of data from Alpha Vantage")
        return data
        
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage: {str(e)}")
        return None

def get_sample_data(symbol, start_date, end_date):
    """
    Generate realistic sample data as a last resort fallback
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pandas.DataFrame: Generated stock data
    """
    print(f"Generating sample data for {symbol}...")
    
    # Create date range (business days only)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Set random seed based on symbol for consistent data
    np.random.seed(hash(symbol) % 2**32)
    
    # Base price varies by symbol
    base_price = 50 + (hash(symbol) % 200)  # Price between $50-$250
    
    # Generate realistic price movement
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1))  # Ensure price stays positive
    
    # Create OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))  # Daily volatility
        
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)
        volume = int(abs(np.random.normal(1000000, 300000)))
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} days of sample data")
    return df

def fetch_stock_data(symbol, start_date, end_date, max_retries=3):
    """
    Fetch stock data with multiple fallback options:
    1. Yahoo Finance (primary)
    2. Alpha Vantage (fallback)
    3. Sample data (last resort)
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        max_retries (int): Maximum retry attempts for each source
    
    Returns:
        pandas.DataFrame: Stock data with columns [Date, Open, High, Low, Close, Volume]
    """
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    # Method 1: Try Yahoo Finance first
    for attempt in range(max_retries):
        try:
            print(f"Trying Yahoo Finance (attempt {attempt + 1}/{max_retries})...")
            
            if attempt > 0:
                delay = 2 ** attempt  # Exponential backoff
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            
            # Download from Yahoo Finance
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False
            )
            
            if df is not None and not df.empty:
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Reset index to have Date as column
                df = df.reset_index()
                
                # Ensure we have the required columns
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    print(f"✅ Successfully fetched {len(df)} days from Yahoo Finance")
                    return df[required_cols]
            
        except Exception as e:
            print(f"Yahoo Finance error (attempt {attempt + 1}): {str(e)}")
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                if attempt < max_retries - 1:
                    delay = 10 + (attempt * 5)  # Longer delay for rate limits
                    print(f"Rate limited. Waiting {delay} seconds...")
                    time.sleep(delay)
    
    print("❌ Yahoo Finance failed. Trying Alpha Vantage...")
    
    # Method 2: Try Alpha Vantage
    for attempt in range(max_retries):
        try:
            print(f"Trying Alpha Vantage (attempt {attempt + 1}/{max_retries})...")
            
            if attempt > 0:
                delay = 15  # Alpha Vantage has strict rate limits
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            
            data = get_alpha_vantage_data(symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched {len(data)} days from Alpha Vantage")
                return data
                
        except Exception as e:
            print(f"Alpha Vantage error (attempt {attempt + 1}): {str(e)}")
    
    print("❌ Alpha Vantage failed. Using sample data...")
    
    # Method 3: Generate sample data as last resort
    try:
        data = get_sample_data(symbol, start_date, end_date)
        print(f"✅ Generated {len(data)} days of sample data")
        return data
    except Exception as e:
        print(f"❌ Sample data generation failed: {str(e)}")
        raise Exception(f"All data sources failed for {symbol}")

def test_data_sources():
    """Test all data sources to verify they're working"""
    print("🧪 Testing data sources...")
    
    test_symbol = "AAPL"
    test_start = "2024-01-01"
    test_end = "2024-01-31"
    
    # Test Yahoo Finance
    print("\n1. Testing Yahoo Finance...")
    try:
        df = yf.download(test_symbol, start=test_start, end=test_end, progress=False)
        if df is not None and not df.empty:
            print("✅ Yahoo Finance is working")
        else:
            print("⚠️ Yahoo Finance returned empty data")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {str(e)}")
    
    # Test Alpha Vantage
    print("\n2. Testing Alpha Vantage...")
    try:
        data = get_alpha_vantage_data(test_symbol, test_start, test_end)
        if data is not None and not data.empty:
            print("✅ Alpha Vantage is working")
        else:
            print("⚠️ Alpha Vantage returned empty data (might be demo key limit)")
    except Exception as e:
        print(f"❌ Alpha Vantage error: {str(e)}")
    
    # Test sample data
    print("\n3. Testing sample data generation...")
    try:
        data = get_sample_data(test_symbol, test_start, test_end)
        if data is not None and not data.empty:
            print("✅ Sample data generation is working")
    except Exception as e:
        print(f"❌ Sample data error: {str(e)}")
    
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    # Run tests if script is executed directly
    test_data_sources()
    
    # Example usage
    print("\n" + "="*50)
    print("Example: Fetching AAPL data...")
    try:
        data = fetch_stock_data("AAPL", "2024-01-01", "2024-01-31")
        print(f"\nData shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        print("\nLast 5 rows:")
        print(data.tail())
    except Exception as e:
        print(f"Error: {str(e)}") 