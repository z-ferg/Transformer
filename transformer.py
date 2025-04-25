from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn

def main():
    ds = load_dataset("fancyzhx/ag_news")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(len(tokenizer.vocab))

if __name__ == "__main__":
    main()
