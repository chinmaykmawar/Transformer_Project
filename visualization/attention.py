import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
from models.bert_for_mlm import BERTForMLM
from tokenizers import Tokenizer

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# texts = [
#     "the king and the queen walked together",
#     "he went to the house",
#     "it was a good day",
#     "the dog chased the cat",
#     "the sun rose in the east"
# ]

texts = [
    "the king and the queen walked together",
]

tokenIDs_saveFile='./training/temp_tokenIDs.pt'
tokenizerJSON_file='./tokenizer/artifacts/tokenizer.json'
modelChekpoint_file='./training/checkpoints/best_model.pt'

tokenizer= Tokenizer.from_file(tokenizerJSON_file)
vocabSize = tokenizer.get_vocab_size()
model = BERTForMLM(vocabSize=vocabSize)
checkpoint = torch.load(modelChekpoint_file)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

tokens=[]
att_matrices=[]
for text in texts:
    encoding = tokenizer.encode(text)
    tokens.append(encoding.tokens)
    inputIDs = torch.tensor([encoding.ids],dtype=torch.long).to(device)

    with torch.no_grad():
        logits, temp_att_matrices = model(inputIDs)
    
    att_matrices.append(temp_att_matrices)

# f= open("attention_analysis.output", "w", encoding="utf-8") 

# for batch_Item in range(len(texts)):
#     for layer in range(6):
#         for head in range(8):
#             att = att_matrices[batch_Item][0,layer,head].cpu()
#             entropy = -(att * torch.log(att + 1e-9)).sum(dim=-1).mean()
#             print(f'BatchItem : {batch_Item}, layer:{layer}, head:{head} -> min:{att.min().item()}, max{att.max().item()}, entropy :{entropy.item()}', file=f)

#             for i in range(len(att)):
#                 j=att[i].argmax().item()
#                 print(f"token {tokens[batch_Item][i]:10s} -> {tokens[batch_Item][j]:10s} with weight {att[i,j]:.3f}", file=f)
            
# f.close()


def plot_att_matrices(att, batch_Item, layer, head):
    att = att.cpu().numpy()
    plt.figure(figsize=(8,6))
    plt.imshow(att)
    plt.xticks(range(len(tokens[batch_Item])),tokens[batch_Item],rotation=45)
    plt.yticks(range(len(tokens[batch_Item])),tokens[batch_Item])
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title(f"batch Item :{batch_Item}, Layer {layer+1} Head {head+1}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"attention_L{layer+1}_H{head+1}.png",dpi=300,bbox_inches="tight")
    plt.show()


plots = [
    (0, 0, 0),  # Sentence 1, Layer 1 Head 1
    (0, 1, 3),  # Sentence 1, Layer 2 Head 4
    (0, 2, 0),  # Sentence 1, Layer 3 Head 1
    (0, 5, 0),  # Sentence 1, Layer 6 Head 1
]

# for batchItem, layer, head in plots:
#     plot_att_matrices(att_matrices[batchItem][0,layer, head], batchItem, layer, head)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()
for ax, (batchItem, layer, head) in zip(axes, plots):
    im = ax.imshow(att_matrices[batchItem][0, layer, head].cpu().numpy())
    ax.set_title(f"Layer {layer+1}, Head {head+1}")
    ax.set_xlabel("Key Token")
    ax.set_ylabel("Query Token")
    fig.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.savefig(f"attention_heatmaps", dpi=300)
plt.show()
