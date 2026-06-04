import torch
from models.attention import SelfAttentionHead


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    seq_len = 10
    d_model = 16
    d_k = 16

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    attention = SelfAttentionHead(
        dmodel=d_model,
        dk=d_k
    ).to(device)

    output, weights = attention(x)

    print("\n=== Shape Checks ===")
    print("Input Shape      :", x.shape)
    print("Output Shape     :", output.shape)
    print("Weights Shape    :", weights.shape)

    print("\n=== Softmax Check ===")
    print(weights.sum(dim=-1))

    print("\n=== Gradient Check ===")

    loss = output.mean()
    loss.backward()

    print(
        attention.Wq.weight.grad is not None,
        attention.Wk.weight.grad is not None,
        attention.Wv.weight.grad is not None
    )

    print("\n=== Device Check ===")
    print("Input Device     :", x.device)
    print("Output Device    :", output.device)
    print("Weights Device   :", weights.device)


if __name__ == "__main__":
    main()