# Tokenization function
# I used the Byte-Pair encoding method
# 
def bpe(training_corpus, k=10):
    """
    vocabulary = set()                  # Set to maintain vocabulary
    pairs = dict()                      # Dictionary to maintain counts
    all_chars = list(training_corpus)   # Assumes training corpus passed as large string

    for i in range(0, len(all_chars) - 1):      # Slide along the whole list
        pair = all_chars[i] + all_chars[i + 1]  # Put the 2 characters together

        if pair in pairs:       # If pairing is aleady in
            pairs[pair] += 1    #   Increment the pair value
        else:                   # Else pairing is not in
            pairs[pair] = 1     #   Set pair value to 1
    
    max_val = float('-inf')
    max_pair = None
    for p in pairs:
        if pairs[p] > max_val:
            max_val = pairs[p]
            max_pair = p
    """
    corpus = training_corpus.strip().split()
    corpus = [' '.join(list(word)) + ' </w>' for word in corpus]
    corpus = [word.split() for word in corpus]