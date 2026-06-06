import torch
from training.dataset import MLMDataset


def main():

    dataset = MLMDataset(
        "./data/Gutenberg_Books+Metadata/processed/token_ids.pt",
        seq_len=128,
        mask_prob=0.15
    )

    print("\n=== Dataset Info ===")

    print(
        "Dataset Length:",
        len(dataset)
    )

    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    print("\n=== Shape Check ===")

    print(
        "Input Shape :",
        input_ids.shape
    )

    print(
        "Label Shape :",
        labels.shape
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

    print("\n=== Mask Count Check ===")

    masked_positions = (
        labels != -100
    ).sum().item()

    print(
        "Masked Tokens:",
        masked_positions
    )

    print(
        "Mask Percentage:",
        masked_positions / len(labels)
    )

    print("\n=== Label Consistency Check ===")

    valid_positions = (
        labels != -100
    )

    print(
        "Masked Positions:",
        valid_positions.sum().item()
    )

    print(
        "Labels Stored:",
        torch.all(
            labels[valid_positions] >= 0
        ).item()
    )

    print("\n=== Multi-Sample Statistics ===")

    counts = []

    for i in range(100):

        sample = dataset[i]

        count = (
            sample["labels"] != -100
        ).sum().item()

        counts.append(count)

    counts = torch.tensor(counts)

    print(
        "Average Masks Per Sample:",
        counts.float().mean().item()
    )

    print(
        "Expected:",
        128 * 0.15
    )

    print(
        "Min:",
        counts.min().item()
    )

    print(
        "Max:",
        counts.max().item()
    )

    print("\n=== Sample Output ===")

    print("Input IDs:")

    print(input_ids[:30])

    print("\nLabels:")

    print(labels[:30])


if __name__ == "__main__":
    main()