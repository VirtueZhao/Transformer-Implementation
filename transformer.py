import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable


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
# print("embr: ", embr)
print(embr.shape)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dimension)
        # print(pe.shape)

        position = torch.arange(0, max_len).unsqueeze(1)
        # print(position.shape)

        div_term = torch.exp(
            torch.arange(0, model_dimension, 2) * -(math.log(10000.0) / model_dimension)
        )
        # print(div_term.shape)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(pe.shape)

        pe = pe.unsqueeze(0)
        # print(pe.shape)

        self.register_buffer("pe", pe)

    def forward(self, x):
        print(x.size(1))
        print(self.pe[:, : x.size(1)].shape)
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)

        return self.dropout(x)


model_dimension = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(model_dimension, dropout, max_len)
pe_result = pe(x)
# print(pe_result)
print(pe_result.shape)

# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(15, 5))

# pe = PositionalEncoding(20, 0)

# y = pe(Variable(torch.zeros(1, 100, 20)))

# plt.plot(np.arange(100), y[0, :, :].data.numpy())
# # plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
# plt.savefig('plot.png')


def subsequent_mask(size):
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")

    return torch.from_numpy(1 - subsequent_mask)


plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
plt.savefig('subsequent_mask.pdf')
