"""I want to learn how to scrape yahoo finance so that I can easily pull stock data in the future.
If I want to get better at machine learning with financial data, I need a faster way to gather said data
"""

import pandas as pd
import yfinance as yf

# Get the S&P500 stock symbols (ironically from scraping wiki instead of yahoo)
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url)[0]        # Reads in the first table with the tickers
tickers = table['Symbol'].tolist()
tickers.append('SPY')     # Add in the S&P500 index
print(tickers[:10])

# Set the date range
start_date = '2018-10-24'
end_date = '2023-10-24'

# Create an empty DataFrame to hold all stock data
combined_data = pd.DataFrame()

# Iterate through each ticker and get its adjusted closing price
for ticker in tickers:
    try:
        # Download data for the given ticker
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')[['Adj Close']]
        # Rename the 'Adj Close' to the ticker symbol
        data = data.rename(columns={'Adj Close': ticker})
        # Join the data to the combined DataFrame
        if combined_data.empty:
            combined_data = data
        else:
            combined_data = combined_data.join(data, how='outer')
        print(f'Data for {ticker} collected')
    except Exception as e:
        print(f'Error fetching data for {ticker}: {e}')

# Save the combined DataFrame as a csv
combined_data.to_csv('/Users/stevenbarnes/Desktop/Resources/Data/Asset Pricing Round 2/5yr_S&P500_data.csv')
print("All data saved to '5yr_S&P500_data.csv'")