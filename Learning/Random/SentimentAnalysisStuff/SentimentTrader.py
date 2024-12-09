"""
Stopiing on this for the meeantime while I figure out how to get EDGAR docs to read in
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import pandas as pd

# Correct EDGAR Base URL for company filings
BASE_URL = "https://data.sec.gov/submissions/CIK{CIK}.json"

# Define the CIKs and filing types to be pulled
COMPANIES = {
    'TSLA': '0001318605',
    'AMZN': '0001018724'
}
FILING_TYPES = ['10-K', '10-Q', '8-K']
OUTPUT_PATH = '/Users/stevenbarnes/Desktop/Resources/Data/SentimentTrader/Filings/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Function to fetch filings from the SEC JSON endpoint
def fetch_filings(cik, filing_type, start_year=2021, end_year=2023):
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code != 200:
        print(f"Failed to fetch filings for {cik}")
        return []

    data = response.json()
    filings = []

    # Extract recent filings
    recent_filings = data.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    filed_dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])

    for form, date, accession in zip(forms, filed_dates, accession_numbers):
        filing_year = int(date.split('-')[0])
        if form == filing_type and start_year <= filing_year <= end_year:
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/index.html"
            filings.append({"Date": date, "Type": form, "Link": filing_url})
    return filings


def download_filings(filings, company, filing_type):
    for filing in filings:
        index_url = filing["Link"]
        response = requests.get(index_url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            print(f"Failed to download index page {index_url}")
            continue

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            # Find the primary document link
            primary_doc = soup.find("a", href=True, text=True)
            if not primary_doc:
                print(f"No primary document found for {index_url}")
                continue

            doc_url = f"https://www.sec.gov{primary_doc['href']}"
            doc_response = requests.get(doc_url, headers={"User-Agent": "Mozilla/5.0"})
            if doc_response.status_code != 200:
                print(f"Failed to download {doc_url}")
                continue

            file_name = f"{company}_{filing_type}_{filing['Date']}.html"
            file_path = os.path.join(OUTPUT_PATH, file_name)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(doc_response.text)
            print(f"Downloaded {file_name}")
            time.sleep(1)  # Respect SEC rate limits
        except Exception as e:
            print(f"Error processing filing at {index_url}: {e}")


# Main script
for company, cik in COMPANIES.items():
    for filing_type in FILING_TYPES:
        print(f'Fetching {filing_type} for {company}')
        filings = fetch_filings(cik, filing_type)
        if filings:
            download_filings(filings, company, filing_type)
