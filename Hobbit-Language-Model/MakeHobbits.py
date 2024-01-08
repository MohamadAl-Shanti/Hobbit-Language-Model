# Imports
import torch.nn.functional as F
import torch
import re
import string

# Reading file into String
with open('hobbit.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Creating list of unique words
text_for_words = text.translate(str.maketrans("", "", string.punctuation))
words = text_for_words.split()
unique_words = set(words)
unique_words = list(unique_words)
vocab = len(unique_words)

# Creating list of all sentences
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
for index, sentence in enumerate(sentences):
    sentences[index] = sentence.replace("\n", " ")

# Dictionaries stoi and itos map each word to an integer value.
stoi = {s: i + 1 for i, s in enumerate(sorted(unique_words))}
stoi['*'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3  # Number of words used to predict next word in the sequence.
X, Y = [], []  # List X: Neural network input. List Y: Labels for each element of X

# Populates lists X and Y
for sen in sentences:

    context = [0] * block_size
    for word in sen.split():
        ix = stoi.get(word, 0)
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn(vocab, 3)  # Look-up table: Each unique word represented by a 3-dimensional tensor.

# Weights and bias entering hidden layer.
W1 = torch.randn(9, 100)
b1 = torch.randn(100)

# Weights and bias entering output layer
W2 = torch.randn(100, vocab)
b2 = torch.randn(vocab)

# List of neural network's parameters
parameters = [C, W1, b1, W2, b2]

# Ensures parameters can be back-propagated through
for p in parameters:
    p.requires_grad = True

# Training loop
for i in range(1000):
    # Forward pass
    emb = C[X]
    h = torch.tanh(emb.view(-1, 9) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(loss.item())

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    for p in parameters:
        p.data += -0.1 * p.grad
