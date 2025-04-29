from collections import defaultdict

# Tokenization function
# I used the Byte-Pair encoding method
# 
def bpe(training_corpus, k=10):
    vocab = defaultdict(int)

    for sentence in training_corpus.split("."):
        for word in sentence.strip().split():
            chars = ['<'] + list(word) + ['>']
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                vocab[pair] += 1

    merges = []
    for _ in range(k):
        if not vocab:
            break
        
        most_frequent = max(vocab, key=lambda x: vocab[x])
        merges.append(most_frequent)

       
        new_char = ''.join(most_frequent)
        new_vocab = defaultdict(int)
        for pair in vocab:
            count = vocab[pair]
            if pair == most_frequent:
                continue
            new_pair = list(pair)
            if new_pair[0] == most_frequent[0] and new_pair[1] == most_frequent[1]:
                new_pair[0] = new_char
                new_pair.pop(1)
            new_vocab[tuple(new_pair)] += count
        vocab = new_vocab
    return merges

def apply_bpe(text, merges):
    chars = ['<'] + list(text) + ['>']  
    for merge in reversed(merges):  
        merged = ''.join(merge)
        new_chars = []
        i = 0
        while i < len(chars) - 1:
            if (chars[i], chars[i+1]) == merge:  
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        if i < len(chars):
            new_chars.append(chars[-1])
        chars = new_chars
    
    return chars

def main():
    corpus = "ab bc bcd cde"
    merges = bpe(corpus, k=3) 
    print("Learned Merges:", merges)
    bpe_representation = apply_bpe("bcd", merges)
    print("BPE Representation for 'bcd':", bpe_representation)

if __name__ == "__main__":
    main()