import requests

# The link to the txt file
url = 'https://raw.githubusercontent.com/AllenDowney/ThinkPython2/master/code/words.txt'

# Histogram function to create a dictionary from a single string
def histogram(s):
    d = dict()
    for c in s:
        d[c] = d.get(c, 0) + 1
    return d

# 11.1 - store words.txt as keys in a dictionary
def read_txt_file(url): # Function to read in the file used
    response = requests.get(url)
    if response.status_code == 200:
        fin = response.text.splitlines()
        return fin
    else:
        print('Failed to retrieve the data. Status code {response.status_code}')

def read_words(url): # Function to store the words in a dict
    fin = read_txt_file(url)
    words = {word: None for word in fin} # Creates a dict with the words as keys and no value
    return words

words_dict = read_words(url)
# print('hello' in words_dict)

# 11.4 - Reverse lookup
def reverse_lookup(d, v):
    keys = []   # Initialize an empty list to store matching keys
    for key in d:
        if d[key] == v:
            keys.append(key)  # Add the key to the list if the value matches
    return keys # Return the list of keys
print(reverse_lookup(histogram('parrot'), 2))

# 11.5 - Function that inverts a dictionary
def invert_dict(d):
    inverse = dict()
    for key in d:
        val = d[key]
        inverse.setdefault(val, []).append(key)
    return inverse
hist = histogram('parrot')
print(hist)
inverse = invert_dict(hist)
print(inverse)

# Memoized Fibonacci
known = {0:0, 1:1}
def fibonacci(n):
    if n in known:
        return known[n]
    res = fibonacci(n-1) + fibonacci(n-2)
    known[n] = res
    return res

# 12.1 - Sumall
def sumall(*args):
    return sum(args)
print(sumall(1, 2, 3))

#  12.3 - Most Frequent
def most_frequent(s):
    freq = dict()
    for letter in s:
        if letter.isalpha():
            letter = letter.lower()
            freq[letter] = freq.get(letter, 0) + 1
    sorted_letters = sorted(freq, key=freq.get, reverse=True)
    for letter in sorted_letters:
        print(f'{letter}: {freq[letter]}')
example_text = 'Hello World'
most_frequent(example_text)
