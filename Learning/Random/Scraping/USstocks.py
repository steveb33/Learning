"""Now that I have an idea of how to scrape tickers and thier prices, I want to collect all US stocks"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import yfinance as yf

# Set up the Selenium WebDriver (no need for executable_path argument if chromedriver is in PATH)
driver = webdriver.Chrome()

# Open the TradingView page
url = 'https://www.tradingview.com/markets/stocks-usa/market-movers-all-stocks/'
driver.get(url)

# Give the page some time to load the dynamic content
time.sleep(5)

# Find elements for Symbol, Market Cap, and Sector
symbols = []
market_caps = []
sectors = []

rows = driver.find_elements(By.CLASS_NAME, 'tv-data-table__row')  # Adjust based on the actual class

for row in rows:
    symbol = row.find_element(By.CLASS_NAME, 'tv-screener__symbol').text.strip()
    market_cap = row.find_elements(By.CLASS_NAME, 'tv-screener-table__cell--big')[2].text.strip()
    sector = row.find_elements(By.CLASS_NAME, 'tv-screener-table__cell--big')[4].text.strip()
    symbols.append(symbol)
    market_caps.append(market_cap)
    sectors.append(sector)

# Create a DataFrame of this data and save it for review
companies = pd.DataFrame({
    'ticker': symbols,
    'market cap': market_caps,
    'sector': sectors
})
companies.to_csv('/Users/stevenbarnes/Desktop/Resources/Data/Asset Pricing Round 2/companies', index=False)

# Close the driver
driver.quit()

print(symbols[:100])  # Check if tickers were scraped correctly


# Establish the list of tickers for the Yahoo scrapping
tickers = companies['ticker'].tolist()
tickers.extend(['SPY', 'QQQ'])     # Add in the S&P500 index


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

"""This is outputting a really ugly multi-level table so I need to clean it up"""

# Reset the index to move Date into a column
combined_data_cleaned = combined_data.reset_index()

# If there is a multi-level column, flatten it to avoid multi-level headers
if isinstance(combined_data_cleaned.columns, pd.MultiIndex):
    combined_data_cleaned.columns = combined_data_cleaned.columns.get_level_values(1)

# Rename the first column to Date
combined_data_cleaned.rename(columns={combined_data_cleaned.columns[0]: 'Date'}, inplace=True)

# Save the DataFrame with the correct structure (no index column)
combined_data_cleaned.to_csv('/Users/stevenbarnes/Desktop/Resources/Data/Asset Pricing Round 2/5yr_S&P500_data.csv', index=False)