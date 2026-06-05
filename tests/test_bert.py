import torch
from models.bert import MiniBERT


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 4
    seq_len = 10

    vocab_size = 1000
    max_seq_len = 128

    d_model = 256
    num_heads = 8
    d_ffn = 1024
    num_layers = 6

    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device
    )

    model = MiniBERT(
        vocabSize=vocab_size,
        maxSeqLen=max_seq_len,
        dmodel=d_model,
        noOfHeads=num_heads,
        dffn=d_ffn,
        noOfLayers=num_layers
    ).to(device)

    output, att_maps = model(token_ids)

    print("\n=== Shape Checks ===")
    print("Token IDs Shape  :", token_ids.shape)
    print("Output Shape     :", output.shape)
    print("Attention Shape  :", att_maps.shape)

    print("\n=== Expected Shapes ===")
    print(
        "Expected Output :",
        (batch_size, seq_len, d_model)
    )

    print(
        "Expected Attn   :",
        (
            batch_size,
            num_layers,
            num_heads,
            seq_len,
            seq_len
        )
    )

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
        "Token Embedding Grad:",
        model.token_emb.weight.grad is not None
    )

    print(
        "Position Embedding Grad:",
        model.pos_emb.weight.grad is not None
    )

    print(
        "First Encoder MHA Grad:",
        model.encoder_blocks[0]
             .mha
             .Wo
             .weight
             .grad is not None
    )

    print(
        "Last Encoder FFN Grad:",
        model.encoder_blocks[-1]
             .ffn
             .linear2
             .weight
             .grad is not None
    )

    print("\n=== Device Check ===")

    print("Token IDs Device :", token_ids.device)
    print("Output Device    :", output.device)
    print("Attention Device :", att_maps.device)

    print("\n=== Statistics ===")

    print(
        "Output Mean:",
        output.mean().item()
    )

    print(
        "Output Std:",
        output.std().item()
    )


if __name__ == "__main__":
    main()