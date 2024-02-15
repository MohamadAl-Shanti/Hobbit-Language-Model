# Imports
import torch.nn.functional as F
import torch
import re
import string
import numpy as np

# Reading file into String
with open('hob2.txt', 'r', encoding='utf-8') as file:
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

xs, ys = [], []
for sen in sentences:
    words = ["*"] + sen.translate(str.maketrans("", "", string.punctuation)).split() + ["*"]
    for word1, word2 in zip(words, words[1:]):
        ix1 = stoi[word1]
        ix2 = stoi[word2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(1293090)
W = torch.randn((len(unique_words) + 1, len(unique_words) + 1), generator=g, requires_grad=True)

for k in range(100):
    xenc = F.one_hot(xs, num_classes=len(unique_words) + 1).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -1000 * W.grad

for i in range(5):
    out = []
    ix = 0

    while True:

        # Start with the initial word
        xenc = F.one_hot(torch.tensor([ix]), num_classes=len(unique_words) + 1).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break
    # Print or use the sampled sentence
    generated_sentence = " ".join(out)
    print(generated_sentence)
