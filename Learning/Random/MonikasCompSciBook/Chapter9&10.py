import requests

# The link to the txt file used in chapter 9
url = 'https://raw.githubusercontent.com/AllenDowney/ThinkPython2/master/code/words.txt'

# Function to read in the file used
def read_txt_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        fin = response.text.splitlines()
        return fin
    else:
        print('Failed to retrieve the data. Status code {response.status_code}')

fin = read_txt_file(url)
# print(words[:10])

# 9.1 - Print the word if it has more than 20 characters
for line in fin:
    word = line.strip()
    if len(word) > 20:
        print(word)

# 9.2 - No e's allowed and embed in 9.1 function
def has_no_e(word):
    return 'e' not in word

'''My attempt'''
for line in fin:
    word = line.strip()
    if len(word) > 20:
        if has_no_e(word) == True:
            print(word)
'''Cleaned up by chatgpt'''
for line in fin:
    word = line.strip()
    if len(word) > 20 and has_no_e(word):
        print(word)

# 9.3 - Forbidden letters
def avoids(word, forbidden_letter):
    for letter in forbidden_letter:
        if letter in word:
            return False
    return True

def avoid_count():
    forbidden_letters = input('Enter a string of forbidden letters: ').strip()
    count = 0
    for line in fin:
        word = line.strip()
        if avoids(word, forbidden_letters):
            count += 1
    print(f'There are {count} words that do not contain any of the forbidden letters')
avoid_count()

# 9.4 - Only these letters
def uses_only(word, required_letters):
    for letter in word:
        if letter not in required_letters:
            return False
    return True

def word_finder():
    required_letters = input('Enter a string of letters that a word must contain: ').strip()
    for line in fin:
        word = line.strip()
        if uses_only(word, required_letters):
            print(word)
word_finder()

# 9.5 - Use all of these letters
def uses_all(word, required_letters):
    for letter in required_letters:
        if letter not in word:
            return False
    return True

def these_letters_only():
    required_letters = input('Enter a string of letters which must all be in the word: ').strip()
    for line in fin:
        word = line.strip()
        if uses_all(word, required_letters):
            print(word)
these_letters_only()

# 9.6 - Abecedarion words
def is_abecedarion(word):
    return word == ''.join(sorted(word))

def count_abecedarion():
    count = 0
    for line in fin:
        word = line.strip()
        if is_abecedarion(word):
            count += 1
    print(f"Number of abecedarion words: {count}")
count_abecedarion()

