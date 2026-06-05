import torch
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

def main():
    corpus_filepath='./data/Gutenberg_Books+Metadata/processed/corpus_medium.txt'
    artifact_dir = Path("./tokenizer/artifacts")

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(
        files=corpus_filepath,
        vocab_size=8000,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]
    )

    tokenizer.save_model(str(artifact_dir))
    tokenizer.save(str(artifact_dir / "tokenizer.json"))
    with open(artifact_dir/'tokenizer_stats.txt', 'w', encoding='utf-8') as f:
        f.write(f"Vocabulary Size: {tokenizer.get_vocab_size()}\n")
        f.write("Special Tokens:\n")
        f.write("[PAD]\n")
        f.write("[UNK]\n")
        f.write("[CLS]\n")
        f.write("[SEP]\n")
        f.write("[MASK]\n")

    print(f"Vocabulary Size: {tokenizer.get_vocab_size()}\n")
    
if __name__ == "__main__":
    main()