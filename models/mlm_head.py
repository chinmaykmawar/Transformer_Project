'''
(batch, 128, 256)
            ↓
Linear(256,8000)
            ↓
(batch,128,8000)
'''

import torch
from torch import nn

class MLMHead(nn.Module):
    def __init__(self, dmodel, vocabSize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear=nn.Linear(dmodel, vocabSize)

    def forward(self, X):
        logits= self.linear(X)
        return logits
    