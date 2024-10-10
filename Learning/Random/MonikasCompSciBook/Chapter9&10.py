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

# 10.2 - Nested listed with capitalization
def capitalize_all(t):
    res = []
    for s in t:
        res.append(s.capitalize())
    return res
def capitalize_nested(nested_list):
    res = []
    for element in nested_list:
        if isinstance(element, list):
            # Recursively capitalize elements in nested lists
            res.append(capitalize_nested(element))
        else:
            #Capitalize the string if it's not a list
            res.append(element.capitalize())
    return res
nested_list = [['let', 'us', 'see'], ['if', 'this'], ['function', 'works'], 'correctly']
capitalize_nested(nested_list)

# 10.3 - Cumulative sum of list
def cumulative_sum(t):
    res = []
    csum = 0
    for num in t:
        csum += num # Add the current number to the running total
        res.append(csum)
    print(res)
    return res
cumulative_sum_list = [1, 2, 3]
cumulative_sum(cumulative_sum_list)

# 10.4 - Middle function
def middle(t):
    return t[1:-1]
print(middle([1, 2, 3, 4]))

# 10.5 - Chopping the middle but return None
def chop(t):
    if len(t) > 1:
        del t[0]
        del t[-1]
print(chop([1, 2, 3, 4]))

