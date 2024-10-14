import requests
import string
import random

# The link to a book
url = 'https://gutenberg.org/cache/epub/74571/pg74571-images.html'

# # # Exercise 13.1 # # #
# # Function to read in the file used
def read_txt_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        fin = response.text.splitlines()
        return fin
    else:
        print('Failed to retrieve the data. Status code {response.status_code}')
        return []

# Function to clean and process each word
def process_word(word):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation) # Create translation table
    return word.translate(translator).lower().strip()

# # Function to count word frequencies
# def word_frequency(url):
#     fin = read_txt_file(url)    # Read in the file
#     word_count = {}     # Initialize an empty dictionary for word frequencies
#     for line in fin:
#         words = line.split()    # Break the line into words
#         for word in words:
#             clean_words = process_word(word)    # Process each word (remove punctuation, lowercase)
#             if clean_words:
#                 word_count[clean_words] = word_count.get(clean_words, 0) + 1    # Update word frequency
#     return word_count
#
# # Example to show that the functions are working (Exercise 13.3)
# frequencies = word_frequency(url)
# top_10 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
# print('Top 10 most frequent words')
# for word, count in top_10:
#     print(f'{word}: {count}')
#
# # 13.2 - Count of different numbers used in the book link
# def count_of_words(url):
#     fin = read_txt_file(url)
#     word_count = {}     # Dictionary to store word frequencies
#     total_words = 0     # Running count of total words
#     for line in fin:
#         words = line.split()    # Split each line into words
#         for word in words:
#             clean_word = process_word(word) # Clean each word
#             if clean_word:
#                 total_words += 1    # Increase count of words
#                 word_count[clean_word] = word_count.get(clean_word, 0) + 1  # Count of word frequency
#     return word_count, total_words
#
# word_count, total_words = count_of_words(url)
# unique_words = len(word_count)
# print(f'Total words: {total_words}')
# print(f'Unique words: {unique_words}')
#
# 13.5 -  Choose from histogram
def histogram(s):
    d = dict()
    for c in s:
        d[c] = d.get(c, 0) + 1
    return d
def choose_from_hist(s):
    