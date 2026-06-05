import torch
from models.encoder import EncoderBlock


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    seq_len = 10
    d_model = 256
    num_heads = 8
    d_ffn = 1024

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device
    )

    encoder = EncoderBlock(
        dmodel=d_model,
        noOfHeads=num_heads,
        dffn=d_ffn
    ).to(device)

    output, attention_maps = encoder(x)

    print("\n=== Shape Checks ===")
    print("Input Shape      :", x.shape)
    print("Output Shape     :", output.shape)
    print("Attention Shape  :", attention_maps.shape)

    print("\n=== Expected Shapes ===")
    print("Expected Output  :", (batch_size, seq_len, d_model))
    print("Expected Attn    :", (batch_size, num_heads, seq_len, seq_len))

    print("\n=== Softmax Check ===")

    row_sums = attention_maps.sum(dim=-1)

    print("Min Row Sum:", row_sums.min().item())
    print("Max Row Sum:", row_sums.max().item())

    print("\n=== Gradient Check ===")

    loss = output.mean()
    loss.backward()

    print(
        "LayerNorm1 Grad:",
        encoder.ln1.weight.grad is not None
    )

    print(
        "LayerNorm2 Grad:",
        encoder.ln2.weight.grad is not None
    )

    print(
        "FFN Linear1 Grad:",
        encoder.ffn.linear1.weight.grad is not None
    )

    print(
        "FFN Linear2 Grad:",
        encoder.ffn.linear2.weight.grad is not None
    )

    print(
        "MHA Wo Grad:",
        encoder.mha.Wo.weight.grad is not None
    )

    print("\n=== Device Check ===")

    print("Input Device     :", x.device)
    print("Output Device    :", output.device)
    print("Attention Device :", attention_maps.device)


if __name__ == "__main__":
    main()
    