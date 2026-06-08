# import sys
# from pathlib import Path

# project_root = Path(__file__).resolve().parent.parent

# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

import torch
from models.bert_for_mlm import BERTForMLM
from tokenizers import Tokenizer
from training.dataset import MLMDataset

from torch.utils.data import DataLoader
 

texts=['the [MASK] and the','he went to the [MASK]', 'it was a [MASK] day','the cat [MASK] on the mat']
tokenIDs_saveFile='./training/temp_tokenIDs.pt'
modelChekpoint_file='./training/checkpoints/best_model.pt'
tokenizerJSON_file='./tokenizer/artifacts/tokenizer.json'

tokenizer= Tokenizer.from_file(tokenizerJSON_file)
vocabSize = tokenizer.get_vocab_size()

model = BERTForMLM(vocabSize=vocabSize)
checkpoint = torch.load(modelChekpoint_file)
print (checkpoint["loss"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

for text in texts:
    encoding=tokenizer.encode(text)
    token_ids = encoding.ids

    inputIDs= torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        logits, att_matrices= model(inputIDs)

    
    mask_pos = token_ids.index(tokenizer.token_to_id("[MASK]"))
    masked_logits = logits[0, mask_pos]
    probs=torch.softmax(masked_logits, dim=-1)

    top_probs, top_ids = torch.topk(probs,k=5)
    print('\n\n',text)
    for token_id, prob in zip(top_ids,top_probs):
        print(tokenizer.id_to_token(token_id.item()),prob.item())
    
    print ('Probs:', top_probs.sum(), top_probs[0])
