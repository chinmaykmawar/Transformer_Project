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

import logging
import sys

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(message)s"
)

fileHandler = logging.FileHandler(
    "./training/train.log"
)
fileHandler.setFormatter(formatter)

consoleHandler = logging.StreamHandler(
    sys.stdout
)
consoleHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

tokenIDsFile=Path("./data/Gutenberg_Books+Metadata/processed/token_ids.pt")
checkpoint_folder=Path("./training/checkpoints")
checkpoint_folder.mkdir(parents=True,exist_ok=True)

seqLen=128
dmodel=256
noOfHeads=8
dffn=1024
noOfLayers=6

dataset=MLMDataset(tokenIDsFile, seqLen)
#dataset = Subset(dataset,range(1000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def runTraining(config):
    batchSize=config['batchSize']
    lr=config['lr']
    noOfEpochs=config['epochs']

    dataLoader=DataLoader(dataset, batch_size=batchSize, shuffle=True)
    vocabSize=8000
    startEpoch = 0

    model=BERTForMLM(vocabSize, dmodel, noOfHeads, dffn, noOfLayers, seqLen).to(device)
    lossFn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer=optim.AdamW(model.parameters(), lr=lr)
    bestEpochLoss=torch.tensor(float('inf'))
    losses=[]

    resumeTraining = True
    if resumeTraining:
        checkpointPath = (checkpoint_folder /"latest.pt")
        checkpoint = torch.load(checkpointPath,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        startEpoch = checkpoint["epoch"] + 1
        bestEpochLoss = checkpoint["loss"]
        #losses=checkpoint['losses']
        losses.append(checkpoint['loss'])
        print(f"Loaded checkpoint ", f"Epoch={checkpoint['epoch']} ", f"Loss={checkpoint['loss']:.4f}"
)    
    print(f"\n\nBatch Size {batchSize} ", f"Lr: {lr:.6f}", f"No Of Epochs: {noOfEpochs}")
    
    for epoch in range(startEpoch,noOfEpochs):
        start = time.time()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            if batchIdx % 8 == 0:
                print(f"\nEpoch {epoch+1} ",f"Batch {batchIdx} ",f"Loss {loss.item():.4f}")
        
        epochLoss /= len(dataLoader)
        losses.append(epochLoss)
        checkpoint_path=checkpoint_folder/f'epoch_{epoch}.pt'
        torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epochLoss},
                checkpoint_path)
        checkpoint_path=checkpoint_folder/f'latest.pt'
        torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epochLoss,
                'losses':losses},
                checkpoint_path)
    
        if epochLoss<bestEpochLoss :
            bestEpochLoss = epochLoss
            checkpoint_path=checkpoint_folder/'best_model.pt'
            torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epochLoss,
                    'losses':losses},
                    checkpoint_path)
 
        print(f"Epoch {epoch+1}/{noOfEpochs} ", f"Loss: {epochLoss:.4f}",f"Epoch Time: ",f"{time.time()-start:.2f} sec")

        



configs=[{'batchSize':64,'lr':8e-4 , 'epochs':10}]

for config in configs:
    runTraining(config)

