"""
This code is to preprocess book pdfs before being used in sentiment analysis
"""

from nltk.tokenize import sent_tokenize
import pdfplumber
import fitz
import re
import csv
import os

output_path = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/CleanedBooks/"

# Path to the texts
ampyscho_pdf = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/American Psycho - Ellis B.E..pdf"
powerofnow_pdf = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/The Power Of Now - Eckhart Tolle.pdf"
smartfootball_pdf = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/The Art of Smart Football - Chris B. Brown.pdf"

# Clean up the text data - embed in the extract text function
def fix_words(text):
    fixed_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return fixed_text

# Extract text from pdfs
def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # If PDF cannot be read by pdfplumber, use PyMuPDF as a fallback
    if len(text) == 0:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()

    # Clean the words
    text = fix_words(text)

    print(f"Extracted text length: {len(text)}")
    return text

# Create the datasets
ampyscho = extract_text(ampyscho_pdf)
powerofnow = extract_text(powerofnow_pdf)
smartfootball = extract_text(smartfootball_pdf)

# ID the start and end markers for all texts
ap_start_marker = 'ABANDON ALL HOPE YE WHO ENTER HERE'
ap_end_marker = 'picador.com'
pn_start_marker = 'I have little use for the past and rarely think about it'
pn_end_marker = 'When you no longer need to ask the question'
sf_start_marker = 'INTRODUCTION'
sf_end_marker = 'ACKNOWLEDGEMENTS'

# Locate the start and end positions
ap_start_pos = ampyscho.find(ap_start_marker)
ap_end_pos = ampyscho.find(ap_end_marker)
pn_start_pos = powerofnow.find(pn_start_marker)
pn_end_pos = powerofnow.find(pn_end_marker)
sf_start_pos = smartfootball.find(sf_start_marker)
sf_end_pos = smartfootball.find(sf_end_marker)

# Slice the texts
ap_filtered = ampyscho[ap_start_pos:ap_end_pos]
pn_filtered = powerofnow[pn_start_pos:pn_end_pos]
sf_filtered = smartfootball[sf_start_pos:sf_end_pos]

# # Print a preview of the filtered content for each book
# print("\n--- American Psycho Preview ---\n")
# print(ap_filtered[:200])
#
# print("\n--- The Power of Now Preview ---\n")
# print(pn_filtered[:200])
#
# print("\n--- Smart Football Preview ---\n")
# print(sf_filtered[:200])

# Tokenize the texts
ap_sentences = sent_tokenize(ap_filtered)
pn_sentences = sent_tokenize(pn_filtered)
sf_sentences = sent_tokenize(sf_filtered)

# # Print a previews of the first 5 sentences of each book
# print("\n--- American Psycho Sentence Preview ---\n")
# print(ap_sentences[:5])
#
# print("\n--- The Power of Now Sentence Preview ---\n")
# print(pn_sentences[:5])
#
# print("\n--- Smart Football Sentence Preview ---\n")
# print(sf_sentences[:5])

# Function to save the cleaned and tokenized sentences
def save_to_csv(sentences, book_name, output_path):
    os.makedirs(output_path, exist_ok=True) # Ensures the output directory exits
    file_path = os.path.join(output_path, f'{book_name}_sentences.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence'])   # Write a header row
        for sentence in sentences:
            writer.writerow([sentence])
    print(f'Sentences for {book_name} saved to {file_path}')

save_to_csv(ap_sentences, "American_Psycho", output_path)
save_to_csv(pn_sentences, "Power_of_Now", output_path)
save_to_csv(sf_sentences, "Smart_Football", output_path)
