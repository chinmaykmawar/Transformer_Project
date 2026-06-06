from pathlib import Path
from tokenizers import Tokenizer
import torch


def main():

    corpus_file = Path("./data/Gutenberg_Books+Metadata/processed/corpus_medium.txt")
    tokenizer_file = Path("./tokenizer/artifacts/tokenizer.json")
    output_file = Path("./data/Gutenberg_Books+Metadata/processed/token_ids.pt")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    print("Reading corpus...")
    with open(corpus_file,"r",encoding="utf-8") as f:
        text = f.read()

    print("Corpus characters:", len(text))

    print("Tokenizing corpus...")
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids
    print("Total tokens:", len(token_ids))

    token_tensor = torch.tensor(token_ids,dtype=torch.long)
    print("Tensor shape:", token_tensor.shape)

    print("Saving token_ids.pt ...")
    torch.save(token_tensor,output_file)
    print("Saved:", output_file)

if __name__ == "__main__":
    main()