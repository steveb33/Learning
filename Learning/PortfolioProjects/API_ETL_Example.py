import requests
import time

# Base API settings
API_KEY = ""
API_HOST = "real-time-finance-data.p.rapidapi.com"
BASE_URL = "https://real-time-finance-data.p.rapidapi.com/stock-quote"


def fetch_stock_data(symbols, exchange="NASDAQ"):
    """Fetch stock data from RapidAPI for a list of stock symbols."""
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST
    }

    stock_results = {}

    for symbol in symbols:
        # Format the URL with the stock symbol and exchange
        url = f"{BASE_URL}?symbol={symbol}%3A{exchange}&language=en"

        try:
            response = requests.get(url, headers=headers, timeout=10)  # Set timeout for request

            if response.status_code == 200:
                try:
                    stock_results[symbol] = response.json()
                except ValueError:
                    print(f"JSON Decode Error for {symbol}")
            else:
                print(f"Error for {symbol}: {response.status_code}, {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed for {symbol}: {e}")

        time.sleep(1)  # Sleep for 1 second to avoid hitting rate limits

    return stock_results


# Stock symbols to search for
stock_symbols = ['AAPL', 'TSLA', 'GOOGL', 'META', 'NVDA', 'AMD']

# Fetch data
stock_data = fetch_stock_data(stock_symbols)

# Print results
if stock_data:
    for symbol, data in stock_data.items():
        print(f"\nStock: {symbol}")
        print(data)
