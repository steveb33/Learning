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
#
# # 9.1 - Print the word if it has more than 20 characters
# for line in fin:
#     word = line.strip()
#     if len(word) > 20:
#         print(word)
#
# # 9.2 - No e's allowed and embed in 9.1 function
# def has_no_e(word):
#     return 'e' not in word
#
# '''My attempt'''
# for line in fin:
#     word = line.strip()
#     if len(word) > 20:
#         if has_no_e(word) == True:
#             print(word)
# '''Cleaned up by chatgpt'''
# for line in fin:
#     word = line.strip()
#     if len(word) > 20 and has_no_e(word):
#         print(word)

# 9.3 - Forbidden letters
def avoids(letters):
    for line in fin:
        word = line.strip()
    if letters in word:
        return False
    else:
        return True

def avoid_count():
    