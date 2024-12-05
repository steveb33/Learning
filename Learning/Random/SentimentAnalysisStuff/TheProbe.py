"""
This will be an exploratory script that I will use to create visuals and tables to learn about the tools
that can be used alongside sentiment analysis. I will be using books that have been preprocessed through the
Books2Sentiment.py script within this folder.

The goal is to them apply these tools to financial contexts to improve my finance & data analytics skills
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Output path
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/SentimentAnalysisOutputs/'
os.makedirs(output_path, exist_ok=True)

# Define input paths for books
BOOKS = {
    "American Psycho": "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/American_Psycho_Sentiment.csv",
    "Power of Now": "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/Power_of_Now_Sentiment.csv",
    "Smart Football": "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/Smart_Football_Sentiment.csv",
    "Into the Wild": "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/Into_the_Wild_Sentiment.csv"
}

# Load data into DataFrames and preprocess them
def load_and_preprocess_books(book_paths):
    dfs = {}
    for title, path in book_paths.items():
        df = pd.read_csv(path)
        dfs[title] = df
    return dfs

books_data = load_and_preprocess_books(BOOKS)

# Normalize the Sentence Index for Moving Averages
def preprocess_sentiment_data(df):
    """
    Normalizes sentence indices so that the books can be on the same x-axis.
    Adds a normalized ranking (1-100) of the DataFrames based on the compound sentiment score
    """
    df['Normalized_Index'] = df.index / len(df)

    # Calculate the min and max of the Compound scores
    min_compound = df['Compound'].min()
    max_compound = df['Compound'].max()

    # Normalize the scores to a scale of 1 to 100
    df['Ranking'] = 1 + 99 * ((df['Compound'] - min_compound) / (max_compound - min_compound))
    return df


# Apply the combined function to each book's DataFrame
for title, df in books_data.items():
    books_data[title] = preprocess_sentiment_data(df)


# Visualization of cumulative sentiment trends
def plot_cumulative_trends(data, score_column, output_file, title):
    """
    Plots cumulative sentiment trends for all books based on a specified column (e.g., Compound, Ranking).

    Doesn't really have any difference in the visual besides what numbers are on the y-axis
    """
    plt.figure(figsize=(12, 6))
    for label, df in data.items():
        df[f'Cumulative_{score_column}'] = df[score_column].expanding().mean()
        plt.plot(df['Normalized_Index'], df[f'Cumulative_{score_column}'], label=label)

    plt.title(title)
    plt.xlabel('Percentage of Sentences Covered')
    plt.ylabel(f'Cumulative {score_column} Score')
    plt.legend(title="Books", loc="best")
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    file_path = os.path.join(output_path, output_file)
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f'{title} plot saved to {file_path}')


# Plot cumulative sentiment trends
plot_cumulative_trends(books_data, 'Compound', 'Cumulative_Sentiment_Trends.png', 'Cumulative Sentiment Trend (Compound)')
plot_cumulative_trends(books_data, 'Ranking', 'Cumulative_Ranking_Trends.png', 'Cumulative Sentiment Trend (Ranking)')

