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
    
'''
Multi-Head Attention

Ctor : (embedding dim =256, no of heads = 8)

Forward Pass

For each head:
    Input:(batch, seq len, 256)
    Output:(batch, seq len, 32)
    Attention:(batch, seq len, seq len)

Concatenate head outputs:(batch, seq len, 256)

Output projection Matrix : (256, 256)

Apply output projection:(batch, seq len, 256)

Return: 
    output (batch, seq len, 256),
    tensor of attention matrices (batch, no of attention heads, seq len, seq len)
'''

class MultiHeadAttention (nn.Module):
    def __init__(self, embeddingDim, noOfHeads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noOfHeads=noOfHeads
        self.embeddingDim=embeddingDim
        self.Wo=nn.Linear(embeddingDim,embeddingDim, bias=False)
        self.heads = nn.ModuleList()
        if (embeddingDim%noOfHeads==0):
            self.dk=embeddingDim//noOfHeads
            for i in range(noOfHeads):
                self.heads.append(SelfAttentionHead(embeddingDim, self.dk))
        else:
            raise ValueError("no of Heads not a factor of embedding dimention")
    
    def forward(self, X):
        outputs=[]
        att_matrices=[]
        for head in self.heads:
            o,a=head(X)
            outputs.append(o)
            att_matrices.append(a)
        output = torch.cat(outputs, dim=-1)
        output = self.Wo(output)
        att_matrices = torch.stack(att_matrices)
        att_matrices=att_matrices.transpose(0,1)
        return output, att_matrices