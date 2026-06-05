import torch
from models.attention import MultiHeadAttention


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    seq_len = 10
    d_model = 256
    num_heads = 8

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device
    )

    mha = MultiHeadAttention(
        embeddingDim=d_model,
        noOfHeads=num_heads
    ).to(device)

    output, att_maps = mha(x)

    print("\n=== Shape Checks ===")
    print("Input Shape      :", x.shape)
    print("Output Shape     :", output.shape)
    print("Attention Shape  :", att_maps.shape)

    print("\n=== Expected Shapes ===")
    print("Expected Output  :", (batch_size, seq_len, d_model))
    print("Expected Attn    :", (batch_size, num_heads, seq_len, seq_len))

    print("\n=== Softmax Check ===")

    row_sums = att_maps.sum(dim=-1)

    print(
        "Min Row Sum:",
        row_sums.min().item()
    )

    print(
        "Max Row Sum:",
        row_sums.max().item()
    )

    print("\n=== Gradient Check ===")

    loss = output.mean()
    loss.backward()

    print(
        "Wo Grad:",
        mha.Wo.weight.grad is not None
    ) 

    for i, head in enumerate(mha.heads):
        print(
            f"Head {i} Wq:",
            head.Wq.weight.grad is not None
        )

    print("\n=== Device Check ===")

    print("Input Device     :", x.device)
    print("Output Device    :", output.device)
    print("Attention Device :", att_maps.device)


if __name__ == "__main__":
    main()