import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset

torch.manual_seed(1)

ds = load_dataset("fancyzhx/ag_news")
text = ds['train']['text']
text = text[:5]
vocab = set()

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

ngrams = []
for sentence in text:
    tokens = sentence.split()
    
    for t in tokens:
        if t not in vocab:
            vocab.add(t)

    if len(tokens) < CONTEXT_SIZE * 2 + 1:
        continue
    sentence_ngram = [
        (
            [tokens[i - j] for j in range(CONTEXT_SIZE, 0, -1)] +
            [tokens[i + j + 1] for j in range(CONTEXT_SIZE)],
            tokens[i]
        )
        for i in range(CONTEXT_SIZE, len(tokens) - CONTEXT_SIZE)
    ]
    ngrams.extend(sentence_ngram)

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

losses = []                                                             # Just to track losses by eye for now

loss_function = nn.NLLLoss()                                            # Use the negative log likelihood loss 
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)   # Create the model to be used for Ngrams
optimizer = optim.SGD(model.parameters(), lr=0.001)                     # Using pytorch optimizer

for epoch in range(10):
    total_loss = 0

    for context, target in ngrams:  
        # Example:
        #   context = (This, is, test, sentence)
        #   target = a

        # Get the positions of the context words
        context_idxs = torch.tensor([word_to_index[w] for w in context], dtype=torch.long)

        # Zero out the gradient of the model
        model.zero_grad()

        # Forward pass to get log probabilities
        log_probs = model.forward(context_idxs)

        # Compute the loss function
        loss = loss_function(log_probs, torch.tensor([word_to_index[target]], dtype=torch.long))

        # Backward pass and update gradient
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    losses.append(total_loss)

# To get the embedding of a particular word, e.g. "beauty"
print(model.embeddings.weight[word_to_index["Wall"]])
print(model.embeddings.weight)