"""
This code is to split books into individual sentences
"""

from nltk.tokenize import sent_tokenize
import pdfplumber
import fitz
from pdf2image import convert_from_path
from pytesseract import image_to_string

# # Testing sent_tokenize
# text = "Hello there! How are you? I am fine. Let's split this text."
# sentences = sent_tokenize(text)
# print(sentences)

# Path to the texts
ampyscho_pdf = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/American Psycho - Ellis B.E..pdf"
powerofnow_pdf = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/The Power Of Now - Eckhart Tolle.pdf"
smartfootball_pdf = "/Users/stevenbarnes/Desktop/Resources/Data/Text4SentimentAnalysis/The Art of Smart Football - Chris B. Brown.pdf"

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

    # If neither of the PDF readers work, use OCR
    if len(text) == 0:
        try:
            pages = convert_from_path(pdf_path)
            for i, page in enumerate(pages):
                # Perform OCR on each page
                page_text = image_to_string(page)
                print(f"OCR Page {i + 1} text length: {len(page_text)}")  # Debugging: print text length per page
                text += page_text
        except Exception as e:
            print(f"OCR failed for {pdf_path}: {e}")

    print(f"Extracted text length: {len(text)}")
    return text


ampyscho = extract_text(ampyscho_pdf)
powerofnow = extract_text(powerofnow_pdf)
smartfootball = extract_text(smartfootball_pdf)

print(ampyscho[:200])

# ID the start and end markers for all texts
ap_start_marker = "ABANDON ALL HOPE YE WHO ENTER HERE"
ap_end_marker = "picador.com"

# Locate the start and end positions
ap_start_pos = ampyscho.find(ap_start_marker)
ap_end_pos = ampyscho.find(ap_end_marker)

# Slice the texts
ap_filtered = ampyscho[ap_start_pos:ap_end_pos]

print(ap_filtered[:200])
