'''
Input IDs
    ↓
MiniBERT
    ↓
Hidden States
(batch, seq_len, 256)
    ↓
MLMHead
    ↓
Logits
(batch, seq_len, 8000)
'''

from torch import nn
from models.bert import MiniBERT
from models.mlm_head import MLMHead

class BERTForMLM(nn.Module):
    def __init__(self, vocabSize, dmodel=256, noOfHeads=8, dffn=1024, noOfLayers=6, seqLen=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.BERT= MiniBERT(vocabSize, seqLen, dmodel, noOfHeads, dffn, noOfLayers)
        self.MLMHead= MLMHead(dmodel, vocabSize)

    def forward(self, inputIDs):
        X, att_matrices= self.BERT(inputIDs)
        logits=self.MLMHead(X)
        return (logits, att_matrices)

            
