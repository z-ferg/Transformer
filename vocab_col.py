from datasets import load_dataset
from string import ascii_letters, digits, punctuation
from collections import Counter

DESIRED_VOCAB_SIZE = 2000

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
all_text = ds[:5]['text']
single_text = "[END]".join(all_text)

# Initialize the vocabulary
tokens = list(single_text)
vocab = set(tokens + ["<|endoftext|>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"])

def get_frequency(tokens):
    ngrams = Counter()
    for i in range(len(tokens)-1):
        ngrams[(tokens[i], tokens[i+1])] += 1
    return ngrams

ngram_counts = get_frequency(tokens)

# Repeat until vocab is desired length
while len(vocab) < DESIRED_VOCAB_SIZE and ngram_counts:
    best_ngram, freq = ngram_counts.most_common(1)[0]
    merged_tok = "".join(best_ngram)
    vocab.add(merged_tok)
    
    i = 0
    new_tokens = []
    
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_ngram:
            new_tokens.append(merged_tok)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    
    tokens = new_tokens
    ngram_counts = get_frequency(tokens)

    print(f"New Token: {merged_tok} -> Vocab Size: {len(vocab)}")

print("--------------------------------------")
for line in vocab:
    print(line)