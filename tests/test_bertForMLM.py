import torch

from models.bert_for_mlm import BERTForMLM


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = BERTForMLM(
        vocabSize=8000
    ).to(device)

    input_ids = torch.randint(
        0,
        8000,
        (4, 128),
        device=device
    )

    logits, att_matrices = model(
        input_ids
    )

    print("\n=== Shape Check ===")

    print(
        "Input Shape :",
        input_ids.shape
    )

    print(
        "Logits Shape:",
        logits.shape
    )

    print(
        "Attention Shape:",
        att_matrices.shape
    )

    print("\n=== Expected Shapes ===")

    print(
        "Logits: (4, 128, 8000)"
    )

    print(
        "Attention: (4, 6, 8, 128, 128)"
    )

    print("\n=== Gradient Check ===")

    loss = logits.mean()

    loss.backward()

    print(
        "MLM Head Grad:",
        model.MLMHead.linear.weight.grad is not None
    )

    print(
        "Token Embedding Grad:",
        model.BERT.token_emb.weight.grad is not None
    )

    print("\n=== Device Check ===")

    print(
        "Input Device:",
        input_ids.device
    )

    print(
        "Logits Device:",
        logits.device
    )

    print(
        "Attention Device:",
        att_matrices.device
    )


if __name__ == "__main__":
    main()