import requests
import time
import pyodbc

# Base API settings
API_KEY = ""
API_HOST = "real-time-finance-data.p.rapidapi.com"
BASE_URL = "https://real-time-finance-data.p.rapidapi.com/stock-quote"

# Extract & Transform
def fetch_stock_data(symbols, exchange="NASDAQ"):
    """Fetch stock data from RapidAPI for a list of stock symbols."""
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": API_HOST
    }

    stock_results = []  # Initiate a list to store stock data dictionaries

    for symbol in symbols:
        url = f"{BASE_URL}?symbol={symbol}%3A{exchange}&language=en"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                try:
                    data = response.json().get("data", {})

                    # Extract relevant fields into a dictionary
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

    return stock_results    # Returns a list of dictionaries to prep for DB loading

# Load the data into the DB
def insert_stock_data(stock_data):
    # Connection parameters
    server = ''
    database = ''
    username = ''
    password = ''
    driver = ''

    # Connection string
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

    try:
        # Connect to SQL Server
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # SQL INSERT query
        insert_query = """
        INSERT INTO Stocks (symbol, name, price, open_price, high, low, volume, previous_close, change_amount, change_percent, last_update_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Iterate through stock data and insert each stock
        for stock in stock_data:
            cursor.execute(insert_query, (
                stock['symbol'], stock['name'], stock['price'], stock['open'],
                stock['high'], stock['low'], stock['volume'], stock['previous_close'],
                stock['change'], stock['change_percent'], stock['last_update_utc']
            ))

        # Commit transaction and close connection
        conn.commit()
        cursor.close()
        conn.close()

        print(f"Successfully inserted {len(stock_data)} records into the database!")

    except Exception as e:
        print(f"Data insertion failed: {e}")

# Stock symbols to search for
stock_symbols = ['AAPL', 'TSLA', 'GOOGL', 'META', 'NVDA', 'AMD']

# Fetch data
stock_data = fetch_stock_data(stock_symbols)

# Print results to verify
if stock_data:
    for stock in stock_data:
        print(stock)

# Insert stock data into SQL
if stock_data:
    insert_stock_data(stock_data)