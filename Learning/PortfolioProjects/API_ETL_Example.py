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

    stock_results = []

    for symbol in symbols:
        url = f"{BASE_URL}?symbol={symbol}%3A{exchange}&language=en"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                try:
                    data = response.json().get("data", {})

                    # Extract relevant fields into a structured format
                    stock_info = {
                        "symbol": data.get("symbol", "N/A"),
                        "name": data.get("name", "N/A"),
                        "price": data.get("price", None),
                        "open": data.get("open", None),
                        "high": data.get("high", None),
                        "low": data.get("low", None),
                        "volume": data.get("volume", None),
                        "previous_close": data.get("previous_close", None),
                        "change": data.get("change", None),
                        "change_percent": data.get("change_percent", None),
                        "last_update_utc": data.get("last_update_utc", None)
                    }

                    stock_results.append(stock_info)

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
    for stock in stock_data:
        print(stock)
