def linecount(filename):
    """Counts the number of lines in the given file."""
    count = 0
    # Open the file and iterate through each line to count
    for line in open(filename):
        count += 1
    return count

# If the script is run directly, print the number of lines in the file
if __name__ == '__main__':
    print(linecount('wc.py'))