"""
Configuration module for trading strategies.
Reads ticker symbol from config.txt file.
"""

import os

def get_ticker():
    """
    Read ticker symbol from config.txt file.
    Returns 'AAPL' as default if file doesn't exist or is empty.
    """
    config_file = 'config.txt'
    
    # Check if config.txt exists in the current directory
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                ticker = f.read().strip().upper()
                if ticker:
                    return ticker
        except Exception as e:
            print(f"Error reading config file: {e}")
    
    # Check if config.txt exists in the parent directory (for strategies folder)
    parent_config = os.path.join('..', config_file)
    if os.path.exists(parent_config):
        try:
            with open(parent_config, 'r') as f:
                ticker = f.read().strip().upper()
                if ticker:
                    return ticker
        except Exception as e:
            print(f"Error reading parent config file: {e}")
    
    # Default ticker if file doesn't exist or is empty
    print("Config file not found or empty. Using default ticker: AAPL")
    return 'AAPL'

# Make ticker available as a module variable
TICKER = get_ticker()

# Additional configuration options (can be expanded later)
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
INITIAL_CAPITAL = 10000 