from datasets import load_dataset
from string import ascii_letters, digits, punctuation
from collections import Counter

DESIRED_VOCAB_SIZE = 2000

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
all_text = ds[:5]['text']
single_text = "[END]".join(all_text)

# Initialize the vocabulary
text = list(single_text)
vocab = set(text + ["<|endoftext|>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"])

# Repeat until vocab is desired length
while len(vocab) < DESIRED_VOCAB_SIZE:
    ngrams = []
    
    # 
    pairs = Counter()
    for i in range(len(text) - 1):
        pairs[(text[i], text[i + 1])] += 1
    
    if not pairs:
        break
    
    best_pair = pairs.most_common(1)[0][0]
    merged_tok = "".join(best_pair)
    
    i = 0
    new_text = []
    
    while i < len(text):
        if i < len(text) - 1 and (text[i], text[i + 1]) == best_pair:
            new_text.append(merged_tok)
            i += 2
        else:
            new_text.append(text[i])
            i += 1
    
    text = new_text
    
    vocab.add(merged_tok)
    print(f"New Token: {merged_tok} -> Vocab Size: {len(vocab)}")

print("--------------------------------------")
for line in vocab:
    print(line)