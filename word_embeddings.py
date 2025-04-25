from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import gensim

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)


def main():
    ds = load_dataset("fancyzhx/ag_news", split='train[:1%]')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab) - 1
    embedding_dim = 8 

    words = ds["text"].join().split(' ')
    vocab = Counter(words)
    vocab = sorted(vocab, key=vocab.get, reverse=True)
    vocab_size = len(vocab)

    print(vocab)


if __name__ == "__main__":
    main()