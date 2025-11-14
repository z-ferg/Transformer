
from string import ascii_letters, digits, punctuation
from collections import Counter

DESIRED_VOCAB_SIZE = 2000

vocab = ascii_letters + digits + punctuation
specials = ["<|endoftext|>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]

print(vocab)
print(type(vocab))