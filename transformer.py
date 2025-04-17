from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Only convert numeric fields to tensors
        tensor_keys = ["input_ids", "attention_mask"]  # add "token_type_ids" if needed
        return {key: torch.tensor(item[key]) for key in tensor_keys}


def tokenize_function(example):
    return tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

def main():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = load_dataset("fka/awesome-chatgpt-prompts")

    # Tokenize only the "prompt" field
    tokenized = ds["train"].map(tokenize_function, batched=True)

    dataset = PromptDataset(tokenized)

    sample = dataset[0]
    print("Token IDs:", sample["input_ids"])
    print("Decoded:", tokenizer.decode(sample["input_ids"]))

if __name__ == "__main__":
    main()
