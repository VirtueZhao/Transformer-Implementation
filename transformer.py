import torch
import torch.nn as nn
from torch.autograd import Variable

import math


# embedding = nn.Embedding(10, 3)
# input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
# print(embedding(input1))

# embedding = nn.Embedding(10, 3, padding_idx=0)
# input1 = torch.LongTensor([[0, 2, 0, 5]])
# print(embedding(input1))


class Embeddings(nn.Module):
    def __init__(self, model_dimension, vocab_size):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab_size, model_dimension)

        self.model_dimension = model_dimension

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.model_dimension)


model_dimension = 512
vocab_size = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

emb = Embeddings(model_dimension, vocab_size)
embr = emb(x)
print("embr: ", embr)
print(embr.shape)
