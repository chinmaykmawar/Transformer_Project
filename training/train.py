'''
train.py should:


Load dataset
Create dataloader
Create model
Create optimizer
Create loss function
Train
Save checkpoints
'''

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from training.dataset import MLMDataset
from models.bert_for_mlm import BERTForMLM
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

tokenIDsFile=Path("./data/Gutenberg_Books+Metadata/processed/token_ids.pt")
checkpoint_folder=Path("./training/checkpoints")
checkpoint_folder.mkdir(parents=True,exist_ok=True)

seqLen=128
batchSize=32
dmodel=256
noOfHeads=8
dffn=1024
noOfLayers=6
lr=1e-4
noOfEpochs=1

dataset=MLMDataset(tokenIDsFile, seqLen)
#dataset = Subset(dataset,range(1000))
dataLoader=DataLoader(dataset, batch_size=batchSize, shuffle=True)
vocabSize=8000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=BERTForMLM(vocabSize, dmodel, noOfHeads, dffn, noOfLayers, seqLen).to(device)
lossFn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer=optim.AdamW(model.parameters(), lr=lr)

bestEpochLoss=torch.tensor(float('inf'))
losses=[]

start = time.time()
for epoch in range(noOfEpochs):
    epochLoss=0.0
    for batchIdx, batch in tqdm(enumerate(dataLoader)):
        inputIDs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, att_matrices= model(inputIDs)
        logits = logits.view(-1, vocabSize)
        labels = labels.view(-1)
        
        loss= lossFn(logits, labels)
        epochLoss+=loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batchIdx % 10 == 0:
            print(f"\nEpoch {epoch+1} ",f"Batch {batchIdx} ",f"Loss {loss.item():.4f}")
        
    epochLoss /= len(dataLoader)
    losses.append(epochLoss)
    checkpoint_path=checkpoint_folder/f'epoch_{epoch}.pt'
    torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epochLoss},
                checkpoint_path)
    
    if epochLoss<bestEpochLoss :
        bestEpochLoss = epochLoss
        checkpoint_path=checkpoint_folder/'best_model.pt'
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epochLoss},
                    checkpoint_path)
 
    print(f"Epoch {epoch+1}/{noOfEpochs} ", f"Loss: {epochLoss:.4f}")

    print(f"Epoch Time: ",f"{time.time()-start:.2f} sec")
    

plt.plot(losses)
        