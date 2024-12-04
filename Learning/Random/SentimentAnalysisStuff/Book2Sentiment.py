"""
This code builds from the work done on Sentiment Analysis so far.

This code will preprocess and label book PDFs for sentiment analysis
"""

import os
import re
import csv
import argparse
import pdfplumber
import fitz
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Clean up the text data
def fix_words(text):
    fixed_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return fixed_text

# Extract from a PDF
def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()

    # Fallback to PyMuPDF if pdfplumber fails
    if len(text) == 0:
        with fitz.open(pdf_path) as pdf:
            text = ''
            for page in pdf:
                text += page.get_text()

    text = fix_words(text)
    return text

# Slice text based on start and end markers
def slice_text(text, start_marker, end_marker):
    start_pos = text.find(start_marker)
    end_pos = text.find(end_marker)
    if start_pos == -1 or end_pos == -1:
        print('Markers not found. Returning full text')
        return text
    return text[start_pos:end_pos]

# Analyze sentiment for each sentence
def analyze_sentiment(sentences):
    labeled_sentence = []
    for sentence in sentences:
        scores = sia.polarity_scores(sentence)
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        labeled_sentence.append({
            'Sentence': sentence,
            'Compound': scores['compound'],
            'Sentiment': sentiment
        })
    return labeled_sentence

# Save tokenized and labeled sentences to a csv
def save_tokenized_sentences(sentences, book_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f'{book_name}_sentences.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence'])
        for sentence in sentences:
            writer.writerow([sentence])
    print(f'Tokenized sentences for {book_name} saved to {file_path}')

# Save sentiment-labeled sentences to a csv
def save_labeled_sentences(labeled_sentences, book_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f'{book_name}_Sentiment.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Sentence', 'Compound', 'Sentiment'])
        writer.writeheader()
        for entry in labeled_sentences:
            writer.writerow(entry)
    print(f'Labeled sentences for {book_name} saved to {file_path}')

# Main Function
def main(pdf_path, book_name, start_marker, end_marker, cleaned_output_path, labeled_output_path):
    print(f'Processing book: {book_name}...')
    text = extract_text(pdf_path)
    print(f'Extracted text length: {len(text)}')

    # Slice text based on markers
    sliced_text = slice_text(text, start_marker, end_marker)

    # Tokenize the sliced text into sentences
    sentences = sent_tokenize(sliced_text)

    # Save tokenized sentences
    save_tokenized_sentences(sentences, book_name, cleaned_output_path)

    # Analyze sentiment
    labeled_sentences = analyze_sentiment(sentences)

    # Save labeled sentences
    save_labeled_sentences(labeled_sentences, book_name, labeled_output_path)

    print(f'Processing for {book_name} completed.\n')

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and label a book PDF for sentiment analysis')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('book_name', help='Title of the book (used for output file naming)')
    parser.add_argument('start_marker', help='Start marker to slice the text')
    parser.add_argument('end_marker', help='End marker to slice the text')
    parser.add_argument('--cleaned_output_path',
                        default='/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/',
                        help='Path to save the tokenized sentences CSV output')
    parser.add_argument('--labeled_output_path',
                        default='/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/LabeledBooks/',
                        help='Path to save the sentiment-labeled CSV output')

    args = parser.parse_args()

    main(args.pdf_path, args.book_name, args.start_marker, args.end_marker, args.cleaned_output_path,
         args.labeled_output_path)
