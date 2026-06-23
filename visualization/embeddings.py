import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA

from models.bert_for_mlm import BERTForMLM
from tokenizers import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenIDs_saveFile='./training/temp_tokenIDs.pt'
tokenizerJSON_file='./tokenizer/artifacts/tokenizer.json'
modelChekpoint_file='./training/checkpoints/best_model.pt'

tokenizer= Tokenizer.from_file(tokenizerJSON_file)
vocabSize = tokenizer.get_vocab_size()
model = BERTForMLM(vocabSize=vocabSize)
checkpoint = torch.load(modelChekpoint_file)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

embedding_matrix = (model.BERT.token_emb.weight.detach().cpu().numpy())

print("Embedding Matrix Shape:")
print(embedding_matrix.shape)

words = [
    "king",
    "queen",
    "man",
    "woman",
    "dog",
    "cat",
    "house",
    "castle",
    "city",
    "day",
    "night",
    "sun",
    "moon",
    "good",
    "bad",
]

valid_words = []
word_vectors = []

for word in words:
    token_id = tokenizer.token_to_id(word)
    if token_id is not None:
        valid_words.append(word)
        word_vectors.append(embedding_matrix[token_id])
    else:
        print(f"Not found: {word}")

word_vectors = torch.tensor(word_vectors).numpy()
print("\nWords Found:")
print(valid_words)

pca = PCA(n_components=2)

coords = pca.fit_transform(word_vectors)
print("Explained Variance:",pca.explained_variance_ratio_)

plt.figure(figsize=(10,8))
plt.scatter(coords[:,0],coords[:,1])
for word, (x,y) in zip(valid_words,coords):
    plt.annotate(word,(x,y))

plt.title("MiniBERT Token Embeddings (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
output_file = "./visualization/embedding_pca.png"
plt.savefig(output_file,dpi=300,bbox_inches="tight")
print(f"\nSaved: {output_file}")
plt.show()
