"""
This code will be used to clean and preprocess individual book pdfs for future Sentiment Analysis.

The code is based off of the learning done in the SplittingBooks.py script
"""

import os
import re
import csv
import argparse
import pdfplumber
import fitz
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are donwloaded
import nltk
nltk.download('punkt')

# Clean up the text data
def fix_words(text):
    # Add space between concatenated words (e.g., "wordWord")
    fixed_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return fixed_text

# Extract text from a PDF
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

    # Clean the extracted text
    text = fix_words(text)
    return text

# Slice text based on start and end markers
def slice_text(text, start_marker, end_marker):
    start_pos = text.find(start_marker)
    end_pos = text.find(end_marker)
    if start_pos == -1 or end_pos == -1:
        print('Markers not found. Returning full text.')
        return text
    return text[start_pos:end_pos]

# Save tokenized sentences to a csv
def save_to_csv(sentences, book_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f'{book_name}_sentences.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence'])
        for sentence in sentences:
            writer.writerow([sentence])
    print(f'Sentences for {book_name} saved to {file_path}')

# Main function
def main(pdf_path, book_name, start_marker, end_marker, output_path):
    print(f'Processing book: {book_name}...')
    text = extract_text(pdf_path)
    print(f'Extracted text length: {len(text)}')

    # Slice text based on markers
    sliced_text = slice_text(text, start_marker, end_marker)

    # Tokenize the sliced text into sentences
    sentences = sent_tokenize(sliced_text)

    # Save the sentences to a csv
    save_to_csv(sentences, book_name, output_path)

    print(f'Processing for {book_name} completed.\n')

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and clean a book PDF for sentiment analysis')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('book_name', help='Title of the book (used for output file naming)')
    parser.add_argument('start_marker', help='Start marker to slice the text')
    parser.add_argument('end_marker', help='End marker to slice the text')
    parser.add_argument('--output_path', default='/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/',
                        help='Path to save the cleaned CSV output (default: CleanedBooks directory)')

    args = parser.parse_args()

    main(args.pdf_path, args.book_name, args.start_marker, args.end_marker, args.output_path)