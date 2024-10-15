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

# Example to show that the functions are working (Exercise 13.3)
frequencies = word_frequency(url)
top_10 = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
print('Top 10 most frequent words')
for word, count in top_10:
    print(f'{word}: {count}')

# 13.2 - Count of different numbers used in the book link
def count_of_words(url):
    fin = read_txt_file(url)
    word_count = {}     # Dictionary to store word frequencies
    total_words = 0     # Running count of total words
    for line in fin:
        words = line.split()    # Split each line into words
        for word in words:
            clean_word = process_word(word) # Clean each word
            if clean_word:
                total_words += 1    # Increase count of words
                word_count[clean_word] = word_count.get(clean_word, 0) + 1  # Count of word frequency
    return word_count, total_words

word_count, total_words = count_of_words(url)
unique_words = len(word_count)
print(f'Total words: {total_words}')
print(f'Unique words: {unique_words}')

# 13.5 -  Choose from histogram
def histogram(s):
    d = dict()
    for c in s:
        d[c] = d.get(c, 0) + 1
    return d

def choose_from_hist(hist): # Function to choose from histogram
    total = sum(hist.values())  # Total number of Items
    cumulative_prob = []    # To store cumualtive probabilities
    items = []   # To store histogram keys

    current_prob = 0
    for key, freq in hist.items():
        current_prob += freq / total    # Add the proportional frequency to cumulative probability
        cumulative_prob.append(current_prob)    # Add cumulative probability to the list
        items.append(key)   # Add the key (item) to the list

    # Generage a random number between 0 and 1
    r = random.random()

    # Find where the random number falls in the cumulative probability list
    for i, prob in enumerate(cumulative_prob):
        if r < prob:
            return items[i]

# Example usage
t = ['a', 'a', 'b']
hist = histogram(t)
print(hist)
for _ in range(10):
    print(choose_from_hist(hist))

import os

# 14.1 - os module exploration
def walk(dirname):
    """Prints the names of all files in dirname and its subdirectories.

    This is the version in the book.

    dirname: string name of directory
    """
    for name in os.listdir(dirname):
        path = os.path.join(dirname, name)

        if os.path.isfile(path):
            print(path)
        else:
            walk(path)

walk('/Users/stevenbarnes/Desktop/pythonProject/Learning/Random/MonikasCompSciBook')

# Trying out a bad file
try:
    fin = open('bad_file')
    for line in fin:
        print(line)
    fin.close()
except:
    print('Something went wrong')

# 14.2 Sed function
def sed(pattern, replacement, input_file, output_file):
    try:
        # Open the input file for reading
        with open(input_file, 'r') as infile:
            # Read the content of the file
            content = infile.read()

        # Replace all occurances of the pattern with the replacement string
        updated_content = content.replace(pattern, replacement)

        # Open (or create) the output file for writing
        with open(output_file, 'w') as outfile:
            # Write the updated content to the output file
            outfile.write(updated_content)
    except FileNotFoundError:
        print(f'Error: {input_file} does not exist.')
    except Exception as e:
        print(f'An error occured: {e}')

sed('old_string', 'new_string', 'input.txt', 'output.txt')  # Example test

