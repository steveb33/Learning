"""
I will be comparing the classification averages of 3 very different books to work on building sentiment analysis
models.

The objective is to get an introduction to this topic of ML. After completing this project, I will be working on
a scale ranking of positivity (1-100) with the same texts.
"""

import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Output Paths
output_path = '/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks'
os.makedirs(output_path, exist_ok=True)


# Read in the sentence csvs
ap_sentences = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/American_Psycho_sentences.csv')
pn_sentences = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/Power_of_Now_sentences.csv')
sf_sentences = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/Smart_Football_sentences.csv')
itw_sentences = pd.read_csv('/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/Into_the_Wild_sentences.csv')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Function to process a book with VADER
def process_sentences(sentences_df):
    results = []
    for sentence in sentences_df['Sentence']:
        score = sia.polarity_scores(sentence)

        # Assign sentiment labels
        if score['compound'] >= 0.05:
            sentiment = 'Positive'
        elif score['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Append results as a dictionary
        results.append({
            'Sentence': sentence,
            'Compound': score['compound'],
            'Sentiment': sentiment
        })
    return pd.DataFrame(results)

# Process and save the results to a csv
def process_and_save(sentences, output_path, book_name):
    # Process sentences to get sentiment results
    results = process_sentences(sentences)

    # Construct the output file path
    output_file = os.path.join(output_path, f'{book_name}_Sentiment.csv')

    # Save the results to a csv
    results.to_csv(output_file, index=False)
    print(f"Results for '{book_name}' saved to {output_file}")

# Process the books
process_and_save(ap_sentences, output_path, 'American_Psycho')
process_and_save(pn_sentences, output_path, 'Power_of_Now')
process_and_save(sf_sentences, output_path, 'Smart_Football')
process_and_save(itw_sentences, output_path, 'Into_the_Wild')