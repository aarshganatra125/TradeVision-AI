import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
import requests
from alpha_vantage.timeseries import TimeSeries
import os

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

    
# Function to calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    short_ema = prices.ewm(span=short_period, adjust=False).mean()
    long_ema = prices.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

# Function to calculate Stochastic Oscillator
def calculate_stochastic(prices, high, low, period=14):
    lowest_low = low.rolling(window=period, min_periods=1).min()
    highest_high = high.rolling(window=period, min_periods=1).max()
    stochastic_k = 100 * (prices - lowest_low) / (highest_high - lowest_low)
    return stochastic_k

# Function to calculate Aroon Indicator
def calculate_aroon(high, low, period=25):
    aroon_up = 100 * high.rolling(window=period, min_periods=1).apply(lambda x: x.argmax(), raw=True) / period
    aroon_down = 100 * low.rolling(window=period, min_periods=1).apply(lambda x: x.argmin(), raw=True) / period
    return aroon_up, aroon_down

# Function to calculate ADX
def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx

# Function to calculate OBV
def calculate_obv(prices, volume):
    obv = (volume * np.sign(prices.diff())).fillna(0).cumsum()
    return obv

# Lorentzian Classification Functions
def calculate_lorentzian_distance(x1, x2, gamma):
    """Calculate Lorentzian distance between two points."""
    return np.log(1 + np.abs(x1 - x2) / gamma)

def calculate_lorentzian_classification(prices, window=20, gamma=1.0):
    """Calculate Lorentzian classification for price action."""
    returns = prices.pct_change()
    classification = np.zeros(len(prices))
    
    for i in range(window, len(prices)):
        window_data = returns[i-window:i]
        distances = np.array([
            calculate_lorentzian_distance(returns[i], x, gamma)
            for x in window_data
        ])
        classification[i] = np.mean(distances)
    
    classification = pd.Series(classification, index=prices.index)
    
    # Normalize the classification scores
    min_vals = classification.rolling(window=window).min()
    max_vals = classification.rolling(window=window).max()
    
    # Avoid division by zero
    denominator = max_vals - min_vals
    denominator = denominator.replace(0, 1)  # Replace zeros with 1 to avoid division by zero
    
    classification = (classification - min_vals) / denominator
    
    return classification

# Function to fetch and preprocess data using Alpha Vantage
def fetch_advanced_metrics(ticker, api_key):
    try:
        # Initialize Alpha Vantage TimeSeries
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        # Fetch daily data (full)
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        # Rename columns to match our previous format
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Sort by date (most recent last)
        data = data.sort_index()
        
        # Use only last 6 months of data
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=180)
        data = data[data.index >= cutoff_date]
        
        if data.empty:
            raise ValueError("No data found for ticker.")

        # Calculate all technical indicators
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD_Line'], data['Signal_Line'] = calculate_macd(data['Close'])
        data['Stochastic'] = calculate_stochastic(data['Close'], data['High'], data['Low'])
        data['Aroon_Up'], data['Aroon_Down'] = calculate_aroon(data['High'], data['Low'])
        data['ADX'] = calculate_adx(data['High'], data['Low'], data['Close'])
        data['OBV'] = calculate_obv(data['Close'], data['Volume'])
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        data['Lorentzian'] = calculate_lorentzian_classification(data['Close'])
        
        return data
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")

# Function to predict stock movement
def predict_stock_movement(data):
    latest = data.iloc[-1]
    score = 0
    reasons = []

    # RSI Analysis
    if latest['RSI'] < 30:
        score += 3
        reasons.append("RSI indicates strong oversold conditions.")
    elif latest['RSI'] > 70:
        score -= 3
        reasons.append("RSI indicates strong overbought conditions.")

    # MACD Analysis
    if latest['MACD_Line'] > latest['Signal_Line']:
        score += 3
        reasons.append("MACD line is above the signal line, indicating strong bullish momentum.")
    else:
        score -= 3
        reasons.append("MACD line is below the signal line, indicating strong bearish momentum.")

    # EMA Analysis
    if latest['EMA_50'] > latest['EMA_200']:
        score += 4
        reasons.append("50-day EMA is above 200-day EMA, indicating a very strong bullish trend.")
    else:
        score -= 4
        reasons.append("50-day EMA is below 200-day EMA, indicating a very strong bearish trend.")

    # ADX Trend Strength Analysis
    if latest['ADX'] > 25:
        score += 2
        reasons.append("ADX confirms a strong trend.")
    
    # OBV Analysis
    if latest['OBV'] > 0:
        score += 2
        reasons.append("On-Balance Volume suggests strong buying pressure.")
    else:
        score -= 2
        reasons.append("On-Balance Volume suggests strong selling pressure.")

    # Lorentzian Analysis
    if not np.isnan(latest['Lorentzian']):
        if latest['Lorentzian'] > 0.7:
            score -= 2
            reasons.append("Lorentzian classification indicates potential trend reversal (overbought).")
        elif latest['Lorentzian'] < 0.3:
            score += 2
            reasons.append("Lorentzian classification indicates potential trend reversal (oversold).")
        elif 0.45 <= latest['Lorentzian'] <= 0.55:
            score += 1
            reasons.append("Lorentzian classification indicates stable trend continuation.")

    # Movement and Confidence
    movement = "Buy" if score > 0 else "Sell"
    confidence = min(abs(score) / 14, 1.0)  # Normalize confidence to a maximum of 100%
    return movement, confidence, reasons

# Main execution block
if __name__ == "__main__":
    print("Welcome to the TradeVision AI")
    try:
        # Get API key
        api_key = "Aarsh"
        if not api_key:
            print("Error: No API key entered. You can get a free key at https://finnhub.io/register")
            exit()
            
        ticker = input("Enter the stock ticker (e.g., AAPL, TSLA): ").strip().upper()
        if not ticker:
            print("Error: No ticker entered. Exiting.")
            exit()

        print(f"\nFetching data for {ticker}...")
        data = fetch_advanced_metrics(ticker, api_key)

        print("\nAnalyzing stock movement...")
        movement, confidence, reasons = predict_stock_movement(data)

        print(f"\nResults for {ticker}:")
        print(f"Prediction: {movement}")
        print(f"Confidence Level: {confidence * 100:.2f}%")
        print("\nAnalysis Reasons:")
        for reason in reasons:
            print(f"- {reason}")
            
        # Print current price
        current_price = data['Close'].iloc[-1]
        print(f"\nCurrent Price: ${current_price:.2f}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"Error: {e}")