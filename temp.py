from wrapper_config import tokenizer
from datasets import load_dataset
from string import ascii_letters, digits, punctuation
from collections import Counter

DESIRED_VOCAB_SIZE = 2000

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
text = "[END]".join(ds[:3]['text'])
text_ascii = "".join(c for c in text if c in (ascii_letters + digits + punctuation + " "))
vocab = ascii_letters + digits + punctuation
specials = ["<|endoftext|>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]

result_vocab = tokenizer.tokenize(vocab, text_ascii, DESIRED_VOCAB_SIZE, specials)

for v in result_vocab:
    if len(v) > 2:
        print(v)