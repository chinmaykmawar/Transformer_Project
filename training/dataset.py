import math
import torch
from torch.utils.data import Dataset, DataLoader

class MLMDataset(Dataset):
    def __init__(self, token_ids_file,seq_len=128, mask_prob=0.15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = torch.load(token_ids_file)
        self.seqLen=seq_len
        self.maskProb=mask_prob
        self.MASK_TOKEN_ID = 4
    
    def __len__(self):
        return math.floor(len(self.tokens)/self.seqLen) 

    def __getitem__(self, idx):
        tokenIDs= self.tokens[idx*self.seqLen:(idx+1)*self.seqLen].clone()
        labels = torch.full_like(tokenIDs,-100)
        prob=torch.rand((self.seqLen,2))
        for i in range(self.seqLen):
            if prob[i][0]<=self.maskProb:
                labels[i]=tokenIDs[i]
                if prob[i][1]<=0.8:
                    tokenIDs[i]=self.MASK_TOKEN_ID
                elif prob[i][1]<=0.9:
                    tokenIDs[i]=self.tokens[torch.randint(len(self.tokens),(1,1))]
        return {"input_ids": tokenIDs, "labels": labels}