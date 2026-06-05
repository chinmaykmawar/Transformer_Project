import torch
from models.attention import MultiHeadAttention
from torch import nn

'''
EncoderBlock

    Ctor:
        Input: dmodel, no of heads, ffn dim
        Vars to Initalize: 
            MHA Object (dmodel, no of heads)
            LayerNorm1 (dmodel)
            ffn class object ( dmodel, ffn dim)
            LayerNorm2 (dmodel)

    MultiHeadAttention
        Input : X (batch size, seq len , dmodel)

        Output : MHA Output(batch size, seq len , dmodel)
                 Attention weights (batch size, no of heads, seq len , seq len)

    Residual Connection + LayerNorm
        Input : X (batch size, seq len , dmodel)
                MHA Output(batch size, seq len , dmodel)
        Output : norm1 output (batch size, seq len , dmodel)

    FeedForward Network
        Input : norm1 output (batch size, seq len , dmodel)
        ffn.linear1=nn.Linear (dmodel, ffn dim)
        ffn.gelu=nn.Gelu()
        ffn.linear2=nn.Linear (ffn dim, dmodel)
        Output : ffn output (batch size, seq len , dmodel)

    Residual Connection + LayerNorm
        Input : norm1 output (batch size, seq len , dmodel)
                ffn output(batch size, seq len , dmodel)
        Output : block output (batch size, seq len , dmodel)
'''

class FeedForwardNetwok(nn.Module):
    def __init__(self,dmodel, dffn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1=nn.Linear(dmodel, dffn)
        self.gelu=nn.GELU()
        self.linear2=nn.Linear(dffn,dmodel)

    def forward(self, X):
        X=self.gelu(self.linear1(X))
        return self.linear2(X)
    

class EncoderBlock(nn.Module):
    def __init__(self, dmodel, noOfHeads, dffn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mha=MultiHeadAttention(dmodel, noOfHeads)
        self.ln1=nn.LayerNorm(dmodel)
        self.ffn=FeedForwardNetwok(dmodel, dffn)
        self.ln2=nn.LayerNorm(dmodel)
    
    def forward(self,X):
        mhaOutput,att_matrices=self.mha(X)
        norm1Output=self.ln1(X+mhaOutput)
        ffnOutput=self.ffn(norm1Output)
        blockOutput=self.ln2(norm1Output+ffnOutput)
        return blockOutput, att_matrices