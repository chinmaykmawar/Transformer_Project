import torch

from torch.utils.data import DataLoader

from training.dataset import MLMDataset


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dataset = MLMDataset(
        "./data/Gutenberg_Books+Metadata/processed/token_ids.pt",
        seq_len=128,
        mask_prob=0.15
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    batch = next(
        iter(dataloader)
    )

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print("\n=== Batch Shapes ===")

    print(
        "Input Shape :",
        input_ids.shape
    )

    print(
        "Label Shape :",
        labels.shape
    )

    print("\n=== Expected Shapes ===")

    print(
        "(32, 128)"
    )

    print("\n=== Dtype Check ===")

    print(
        "Input Dtype :",
        input_ids.dtype
    )

    print(
        "Label Dtype :",
        labels.dtype
    )

    print("\n=== Mask Statistics ===")

    total_positions = labels.numel()

    masked_positions = (
        labels != -100
    ).sum().item()

    print(
        "Masked Tokens :",
        masked_positions
    )

    print(
        "Mask Percentage :",
        masked_positions / total_positions
    )

    print(
        "Expected :",
        0.15
    )

    print("\n=== Device Transfer Check ===")

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    print(
        "Input Device :",
        input_ids.device
    )

    print(
        "Label Device :",
        labels.device
    )

    print("\n=== Sample Batch ===")

    print(
        "Input IDs:"
    )

    print(
        input_ids[0][:20]
    )

    print(
        "\nLabels:"
    )

    print(
        labels[0][:20]
    )


if __name__ == "__main__":
    main()