import requests
import string

# # # Exercise 13.1 # # #
# The link to a book
url = 'https://gutenberg.org/cache/epub/74565/pg74565-images.html'

# Function to read in the file used
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

# Function to count word frequencies
def word_frequency(url):
    fin = read_txt_file(url)    # Read in the file
    word_count = {}     # Initialize an empty dictionary for word frequencies
    for line in fin:
        words = line.split()    # Break the line into words
        for word in words:
            clean_words = process_word(word)    # Process each word (remove punctuation, lowercase)
            if clean_words:
                word_count[clean_words] = word_count.get(clean_words, 0) + 1    # Update word frequency
    return word_count

# Example to show that the functions are working
frequencies = word_frequency(url)
top_10 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
print('Top 10 most frequent words')
for word, count in top_10:
    print(f'{word}: {count}')

