import pandas as pd
import math

# 7.5 - Testing the square root
def newton_sqrt(a):
    x = a
    for _ in range(10):
        x = (x + a/x) / 2
    return x

def test_square_root():
    data = {
        'a': [],
        'Newton sqrt': [],
        'math.sqrt': []
    }

    for a in range(1, 10):
        newton_result = newton_sqrt(a)
        math_result = math.sqrt(a)

        data['a'].append(a)
        data['Newton sqrt'].append(newton_result)
        data['math.sqrt'].append(math_result)

    df = pd.DataFrame(data)
    df.set_index('a', inplace=True)
    print(df)
test_square_root()

# 8.1 - Displaying letters backwards
def backward_string(word):
    index = len(word) - 1
    while index > -1:
        letter = word[index]
        print(letter)
        index = index - 1
backward_string('banana')


# 8.2 - Concatenation example
prefixes = 'JKLMNOPQ'
suffix = 'ack'

for letter in prefixes:
    if letter == 'O' or letter == 'Q':
        print(letter + 'u' + suffix)
    else:
        print(letter + suffix)

# 8.5 - Letter counter
def letter_counter(word, letter):
    count = 0
    for _ in word:
        if _ == letter:
            count = count + 1
    print(count)
letter_counter('banana', 'a')