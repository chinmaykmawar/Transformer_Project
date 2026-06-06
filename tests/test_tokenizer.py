from tokenizers import BertWordPieceTokenizer
from pathlib import Path


def main():

    artifact_dir = Path("./tokenizer/artifacts")

    tokenizer = BertWordPieceTokenizer(
        str(artifact_dir / "vocab.txt")
    )

    print("\n=== Vocabulary ===")

    print(
        "Vocabulary Size:",
        tokenizer.get_vocab_size()
    )

    print("\n=== Special Tokens ===")

    special_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]

    for token in special_tokens:

        token_id = tokenizer.token_to_id(token)

        print(
            f"{token:<10} -> {token_id}"
        )

    print("\n=== Encode Test ===")

    text = "The cat sat on the mat."

    encoding = tokenizer.encode(text)

    print("Text:")
    print(text)

    print("\nTokens:")
    print(encoding.tokens)

    print("\nIDs:")
    print(encoding.ids)

    print("\n=== Decode Test ===")

    decoded = tokenizer.decode(
        encoding.ids
    )

    print(decoded)

    print("\n=== WordPiece Test ===")

    rare_word = "electromagnetohydrodynamics"

    encoding = tokenizer.encode(
        rare_word
    )

    print("Word:")
    print(rare_word)

    print("\nTokens:")
    print(encoding.tokens)

    print("\nIDs:")
    print(encoding.ids)

    print("\n=== Sentence Test ===")

    sentence = (
        "Transformers use multi-head attention "
        "to build contextual representations."
    )

    encoding = tokenizer.encode(
        sentence
    )

    print("Original:")
    print(sentence)

    print("\nTokens:")
    print(encoding.tokens)

    print("\nDecoded:")
    print(
        tokenizer.decode(
            encoding.ids
        )
    )


if __name__ == "__main__":
    main()