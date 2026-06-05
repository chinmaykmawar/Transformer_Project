'''
Mini BERT

    Ctor:
        Input: vocab size, maxx len, dmodel, no of heads, ffn dim, no of layers
        Vars to Initalize: 
            token embedding = nn.Embedding(vocab size, dmodel)
            psoition embedding = nn.Embedding (max len, dmodel)
            encoder blocks = nn.ModuleList() of EncoderBlock 

    Token Embeddings
        Input : list of token IDs (batch size, seq len)
        Output : Token Embeddings (batch size, seq len, dmodel)

    Positinal Embeddings
        Input : list of positions (batch size, seq len)
        Output : Pos Embeddings (batch size, seq len, dmodel)

    Encoder 1
        Input: X = Token emb + Pos Emb (batch, seq len, dmodel)
        Output: X (batch size, seq len, dmodel)
                att_matrices (batch size, no of heads, seq len, seq len)  

    Encoder 2
        Input: out put of encoder 1 = X (batch, seq len, dmodel)
        Output: X (batch size, seq len, dmodel)
                att_matrices (batch size, no of heads, seq len, seq len)  

    .
    .
    .
    .
    .
    .    

    Encoder N
        Input: out put of encoder N-1 = X (batch, seq len, dmodel)
        Output: X (batch size, seq len, dmodel)
                att_matrices (batch size, no of heads, seq len, seq len)  
    
    Model Output : X (contexual word representations)
                   att matrices(batch size, no of layers, no of heads, seq len, seq len)
'''

import torch
from torch import nn
from models.encoder import EncoderBlock

class MiniBERT(nn.Module):
    def __init__(self, vocabSize, maxSeqLen, dmodel, noOfHeads, dffn, noOfLayers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel= dmodel
        self.noOfHeads= noOfHeads
        self.noOfLayers= noOfLayers
        self.maxSeqLen= maxSeqLen
        self.dffn=dffn
        self.token_emb = nn.Embedding(vocabSize, dmodel)
        self.pos_emb = nn.Embedding (maxSeqLen, dmodel)
        self.encoder_blocks = nn.ModuleList()
        for i in range(noOfLayers):
            self.encoder_blocks.append(EncoderBlock(dmodel,noOfHeads,dffn))
        
    def forward(self, tokenIDs):
        batch_size, seq_len = tokenIDs.shape
        posIDs = torch.arange(seq_len,device=tokenIDs.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeddings=self.token_emb(tokenIDs)
        pos_embeddings=self.pos_emb(posIDs)
        X=token_embeddings+pos_embeddings
        att_matrices=[]
        for encoder in self.encoder_blocks:
            X, att_matrix=encoder(X)
            att_matrices.append(att_matrix)
        
        att_matrices=torch.stack(att_matrices)
        att_matrices=att_matrices.transpose(0,1)
        return X, att_matrices
