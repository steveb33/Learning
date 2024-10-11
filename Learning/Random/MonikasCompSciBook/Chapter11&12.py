import requests

# The link to the txt file
url = 'https://raw.githubusercontent.com/AllenDowney/ThinkPython2/master/code/words.txt'

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
print('hello' in words_dict)

# 