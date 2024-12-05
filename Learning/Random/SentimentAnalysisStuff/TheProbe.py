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

# Load sentiment-labeled data for all books
ap_df = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/American_Psycho_Sentiment.csv')
pn_df = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/Power_of_Now_Sentiment.csv')
sf_df = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/Smart_Football_Sentiment.csv')
itw_df = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/Into_the_Wild_Sentiment.csv')

# Normalize the Sentence Index for Moving Averages
def preprocess_sentiment_data(df):
    """
    Normalize sentence indices so that the books can be on the same x-axis
    """
    df['Normalized_Index'] = df.index / len(df)
    return df


# Apply the combined function to each book's DataFrame
ap_df = preprocess_sentiment_data(ap_df)
pn_df = preprocess_sentiment_data(pn_df)
sf_df = preprocess_sentiment_data(sf_df)
itw_df = preprocess_sentiment_data(itw_df)

def plot_sentiment_trends(dfs, labels, colors, output_path):
    """
    Plot cumulative sentiment trends of the books and save the result

    Parameters:
        - dfs: List of DataFrames for each book
        - labels: List of labels for each book (e.g., book titles)
        - colors: List of colors for each book's line
    """
    plt.figure(figsize=(12, 6))

    for df, label, color in zip(dfs, labels, colors):
        # Calculate the cumulative moving average
        df['Cumulative_Mean'] = df['Compound'].expanding().mean()
        # Plot cumulative mean
        plt.plot(df['Normalized_Index'], df['Cumulative_Mean'], label=label, color=color)

    plt.title('Cumulative Sentiment Trend')
    plt.xlabel('Percentage of Sentences Covered')
    plt.ylabel('Cumulative Sentiment Score')
    plt.legend(title="Books", loc="best")  # Adding a title to the legend
    plt.grid(True)

    # Save the plot to the output path
    file_path = os.path.join(output_path, 'SentimentTrends.png')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    print(f'Sentiment trends plot saved to {file_path}')

# DataFrames, labels, and colors for the books
dfs = [ap_df, pn_df, sf_df, itw_df]
labels = ['American Psycho', 'Power of Now', 'Smart Football', 'Into the Wild']
colors = ['red', 'blue', 'green', 'orange']

# Plot the sentiment trends
plot_sentiment_trends(dfs, labels, colors, output_path)

