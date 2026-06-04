import math
import torch
from torch import nn

class SelfAttentionHead(nn.Module):
    def __init__(self, dmodel, dk, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Wq = nn.Linear(dmodel, dk, bias=False)
        self.Wk = nn.Linear(dmodel, dk, bias=False)
        self.Wv = nn.Linear(dmodel, dk, bias=False)
        self.dk = dk

    def forward(self, X):
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        att_score = Q @ K.transpose(-2, -1)
        att_score = att_score / math.sqrt(self.dk)

        att_weights = torch.softmax(att_score, dim=-1)
        output = att_weights @ V

        return output, att_weights