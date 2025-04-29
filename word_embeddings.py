import torch
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from collections import Counter

import numpy as np
from scipy import spatial # type: ignore

CONTEXT_SIZE = 2        # The number of words before and after the context word
EMBEDDING_DIM = 128     # The number of dimensions to track for dimensionality
BATCH_SIZE = 32         # The size to batch by

DIR = "C:\\Users\\Zach\\Desktop\\HI BOYFRIEND\\Transformer"
FILENAME = "model.pt"
PATH = os.path.join(DIR, FILENAME)

GLOVE_DIR = "C:\\Users\\Zach\\Desktop\\HI BOYFRIEND\\Transformer\\glove.840B.300d.txt\\glove.840B.300d.txt"

"""
    Helper function for testing purposes
    args:
        n    => this is the number of sentences to pick from the dataset
        word => the word to test for similarity at the end
"""
def ngram_gen(text):
    vocab = Counter()

    ngrams = [] 
    for sentence in text:
        tokens = sentence.split()
        vocab.update(tokens)

        if len(tokens) < CONTEXT_SIZE * 2 + 1:  # If sentence is too small then skip it
            continue

        sentence_ngram = []
        for i in range(CONTEXT_SIZE, len(tokens) - CONTEXT_SIZE):
            ngrams.append((
                [tokens[i - j] for j in range(CONTEXT_SIZE, 0, -1)] +   # Get CONTEXT_SIZE words prior to the target word
                [tokens[i + j + 1] for j in range(CONTEXT_SIZE)],       # Get CONTEXT_SIZE words after the target word
                tokens[i]
            ))
    
    word_to_index = {word: i for i, word in enumerate(vocab)}   # Currently using index positional encoding

    """
        What does n-grams look like at this point? (Assuming context size of 2)
            Text => "This is a test sentence.".split()
            ngrams list => [(['This', 'is', 'test', 'sentence.'], 'a')]
                [0] -> Context words ranging (i - CONTEXT_SIZE) - (i + CONTEXT_SIZE)
                [1] -> The word in context
    """

    return ngrams, word_to_index, list(vocab)

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)         # Establish the embedding matrix (lookup table)
        self.linear1 = nn.Linear((context_size * 2) * embedding_dim, 128) # Project large input vector into vector of embedding_dim
        self.linear2 = nn.Linear(128, vocab_size)                         # Map the projection to vector of size vocab
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(inputs.shape[0], -1)  # Get vectors of the inputs and flatten using view
        h1 = self.linear1(embeds)                                   # Get first hidden layer (inputs -> hidden layer 1)
        r = F.relu(h1)                                              # Apply non-linearity (h1 -> relu)
        out = self.linear2(r)                                       # Pass through linear2 to get final layer
        return F.log_softmax(out, dim=1)                            # Softmax the final layer to get the probability


def find_most_similar(word, word_to_ix, embeddings, n=5):
    norm = embeddings / embeddings.norm(dim=1, keepdim=True)    # Normalize the embeddings
    word_embed = norm[word_to_ix[word]].unsqueeze(0)            # Get the word embedding and store in size (1, EMBEDDING_DIM)
    cos_sim = torch.mm(word_embed, norm.t()).squeeze(0)         # Matrix multiply word embeddings and squeeze into (vocab_size,)
    top_n = torch.topk(cos_sim, n + 1)                          # Get the k largest elements of the cosine similarities tensor

    similar_words = []
    ix_to_word = {i: w for w, i in word_to_ix.items()}          # Convert the indices back to the words
    
    for idx in top_n.indices:       # Get the index of the top_n words
        w = ix_to_word[idx.item()]  # Get the word given the index
        if w != word:               # Skip the word when it is the target word itself
            similar_words.append(w) 

    return similar_words


def glove_similarity(glove_file_path, target_word, n, embedding_dim=300):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = ' '.join(parts[:-embedding_dim])    # Word might contain spaces
            vector = np.asarray(parts[-embedding_dim:], dtype='float32')
            embeddings[word] = vector
    
    if target_word not in embeddings:
        raise ValueError(f"Word '{target_word}' not found in GloVe vocabulary.")

    target_vector = embeddings[target_word]

    # Compute cosine distances
    distances = {
        w: spatial.distance.cosine(vec, target_vector)
        for w, vec in embeddings.items()
    }

    # Sort by closest
    most_similar = sorted(distances.items(), key=lambda x: x[1])[1:n+1]  # [1:] to skip the word itself
    return [word for word, _ in most_similar]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)

    ds = load_dataset("fancyzhx/ag_news")   # Using this simple "lil" dataset for now
    data_text = ds['train']['text']         # Get all of the text in the training set (disregard labels for now)

    ngrams, word_to_index, vocab = ngram_gen(data_text)

    loss_function = nn.NLLLoss()                                                        # Use the negative log likelihood loss 
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(device)    # Create the model to be used for Ngrams
    optimizer = optim.SGD(model.parameters(), lr=0.001)                                 # Using pytorch optimizer

    for epoch in range(1):
        print(f'Epoch: {epoch + 1}')

        random.shuffle(ngrams)
        
        for start in range(0, len(ngrams), BATCH_SIZE):
            batch = ngrams[start : start + BATCH_SIZE]

            context_batch = []
            target_batch = []

            for context, target in batch:
                context_indexes = [word_to_index[w] for w in context]
                context_batch.append(context_indexes)
                target_batch.append(word_to_index[target])

            context_batch = torch.tensor(context_batch, dtype=torch.long).to(device)
            target_batch = torch.tensor(target_batch, dtype=torch.long).to(device)

            model.zero_grad()
            log_probs = model(context_batch)
            loss = loss_function(log_probs, target_batch)
            loss.backward()
            optimizer.step()
            
            torch.save(model, PATH) # Is this how to checkpoint? Save after every epoch
    
    self_sim = find_most_similar("oil", word_to_index, model.embeddings.weight, n=50)
    print(f'Words similar to "oil": {self_sim}')

    print()

    glove_sim = glove_similarity(GLOVE_DIR, "oil", 50)
    print(f'Words similar to "oil": {glove_sim}')

    count = 0
    for item in self_sim:
        if item in glove_sim:
            count += 1
    
    print(f'Total overlap between glove and ag-news dataset: {count}')


if __name__ == "__main__":
    main()