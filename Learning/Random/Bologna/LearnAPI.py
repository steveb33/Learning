"""
This example script is to figure out how to get stock data via API.
"""
import pandas as pd
import requests

# My API key - find key at https://site.financialmodelingprep.com/developer/docs/dashboard
api_key = 'find key at website'

# Some example stocks
stocks = ['AMZN', 'AAPL', 'META', 'GOOGL']

# Endpoint to get the quote for Apple (AAPL)
url = f'https://financialmodelingprep.com/api/v3/quote/{",".join(stocks)}?apikey={api_key}'

# Fetch the sample data
response = requests.get(url)

# Check if response is OK (status code 200)
if response.status_code == 200:
    data = response.json()

    # Extract the symbol, price, and volume
    stock_data = [{'symbol': stock['symbol'], 'price': stock['price'], 'volume': stock['volume']} for stock in data]

    # Create a df with the extracted data
    df = pd.DataFrame(stock_data)
    print(df)


else:
    print(f"Error fetching data: {response.status_code}")