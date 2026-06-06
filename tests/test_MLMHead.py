import torch

from models.mlm_head import MLMHead


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    head = MLMHead(
        dmodel=256,
        vocabSize=8000
    ).to(device)

    X = torch.randn(
        4,
        128,
        256,
        device=device
    )

    logits = head(X)

    print("\n=== Shape Check ===")

    print(
        "Input Shape  :",
        X.shape
    )

    print(
        "Output Shape :",
        logits.shape
    )

    print("\n=== Expected Shape ===")

    print(
        "(4, 128, 8000)"
    )

    print("\n=== Statistics ===")

    print(
        "Mean :",
        logits.mean().item()
    )

    print(
        "Std  :",
        logits.std().item()
    )

    print("\n=== Gradient Check ===")

    loss = logits.mean()

    loss.backward()

    print(
        "Linear Weight Grad:",
        head.linear.weight.grad is not None
    )

    print(
        "Linear Bias Grad:",
        head.linear.bias.grad is not None
    )

    print("\n=== Device Check ===")

    print(
        "Input Device :",
        X.device
    )

    print(
        "Output Device:",
        logits.device
    )

    print(
        "Weight Device:",
        head.linear.weight.device
    )


if __name__ == "__main__":
    main()