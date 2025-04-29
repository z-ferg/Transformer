import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset

torch.manual_seed(1)

ds = load_dataset("fancyzhx/ag_news")   # Using this simple lil dataset for now
data_text = ds['train']['text']         # Get all of the text in the training set (disregard labels for now)

"""
    Helper function for testing purposes
    args:
        n    => this is the number of sentences to pick from the dataset
        word => the word to test for similarity at the end

    ** DELETE THIS FUNCTION LATER **
"""
def big_loop(n, word):
    text = data_text[:n]    # Store n sentences of the dataset in text
    vocab = set()           # A set to keep track of the vocab in this dataset (not certain if this is the most optimal method)

    CONTEXT_SIZE = 2        # The number of words before and after the context word
    EMBEDDING_DIM = 10      # The number of dimensions to track for dimensionality

    ngrams = [] 
    for sentence in text:
        tokens = sentence.split()
        
        for t in tokens:    
            if t not in vocab:
                vocab.add(t)    # Add token to vocab if not already noted

        if len(tokens) < CONTEXT_SIZE * 2 + 1:  # If sentence is too small then skip it
            continue

        sentence_ngram = []
        for i in range(CONTEXT_SIZE, len(tokens) - CONTEXT_SIZE):
            ngrams.append((
                [tokens[i - j] for j in range(CONTEXT_SIZE, 0, -1)] +   # Get CONTEXT_SIZE words prior to the target word
                [tokens[i + j + 1] for j in range(CONTEXT_SIZE)],       # Get CONTEXT_SIZE words after the target word
                tokens[i]
            ))

    """
        What does n-grams look like at this point? (Assuming context size of 2)
            Text => "This is a test sentence.".split()
            ngrams list => [(['This', 'is', 'test', 'sentence.'], 'a')]
                [0] -> Context words ranging (i - CONTEXT_SIZE) - (i + CONTEXT_SIZE)
                [1] -> The word in context
    """

    word_to_index = {word: i for i, word in enumerate(vocab)}   # Currently using index positional encoding

    class NGramLanguageModeler(nn.Module):
        def __init__(self, vocab_size, embedding_dim, context_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)         # Establish the embedding matrix (lookup table)
            self.linear1 = nn.Linear((context_size * 2) * embedding_dim, 128) # Project large input vector into vector of 128
            self.linear2 = nn.Linear(128, vocab_size)                         # Map the projection to vector of size vocab
        
        def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1, -1))      # Get vectors of the inputs and flatten using view
            h1 = self.linear1(embeds)                           # Get first hidden layer (inputs -> hidden layer 1)
            r = F.relu(h1)                                      # Apply non-linearity (h1 -> relu)
            out = self.linear2(r)                               # Pass through linear2 to get final layer
            return F.log_softmax(out, dim=1)                    # Softmax the final layer to get the log probability

    loss_function = nn.NLLLoss()                                            # Use the negative log likelihood loss 
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)   # Create the model to be used for Ngrams
    optimizer = optim.SGD(model.parameters(), lr=0.001)                     # Using pytorch optimizer

    for epoch in range(10):
        for context, target in ngrams:  
            # Example:
            #   context = (This, is, test, sentence)
            #   target = a

            context_idxs = torch.tensor([word_to_index[w] for w in context], dtype=torch.long)          # Get the positions of the context words
            model.zero_grad()                                                                           # Zero out the gradient of the model
            log_probs = model.forward(context_idxs)                                                     # Forward pass to get log probabilities
            loss = loss_function(log_probs, torch.tensor([word_to_index[target]], dtype=torch.long))    # Compute the loss function

            loss.backward()             # Perform backward pass
            optimizer.step()            # Update the optimizer
    
    similar = find_most_similar(word, word_to_index, model.embeddings.weight, n=5)
    print(f'Words similar to {word} on corpus size {n}: {similar}')
    print()

def find_most_similar(word, word_to_ix, embeddings, n=5):
    norm = embeddings / embeddings.norm(dim=1, keepdim=True)    # Normalize the embeddings
    word_embed = norm[word_to_ix[word]].unsqueeze(0)            # Get the word embedding and store in size (1, EMBEDDING_DIM)
    cos_sim = torch.mm(word_embed, norm.t()).squeeze(0)         # Matrix multipy word embeddings and squeeze into (vocab_size,)
    top_n = torch.topk(cos_sim, n + 1)                          # Get the k largest elements of the cosine similarities tensor

    similar_words = []
    ix_to_word = {i: w for w, i in word_to_ix.items()}          # Convert the indicies back to the words
    
    for idx in top_n.indices:       # Get the index of the top_n words
        w = ix_to_word[idx.item()]  # Get the word given the index
        if w != word:               # Skip the word when it is the target word itself
            similar_words.append(w) 

    return similar_words

"""
    Using a list of sizes and list of words to find similarity, iterate the helper function
    Will print out the 10 most similar words seperated by a large dashed line between words
"""
sizes = [5, 10, 25, 50, 100, 150, 200]
words = ["Private", "investment", "Oil", "oil"]

for word in words:
    for size in sizes:
        big_loop(size, word)
    print("-----------------------------------------------------------------------------")