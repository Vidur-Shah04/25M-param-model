from pathlib import Path
import os
import sys
import csv

import torch
import matplotlib.pyplot as plt
import sentencepiece as spm

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path("/kaggle/working")
sys.path.insert(0, str(PROJECT_DIR))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


DATA_DIR = Path(os.environ.get("ABLATION_DATA_DIR", "/kaggle/input/datasets/vidurshah04/assam-data"))

TOKENIZER_PATH = Path("/kaggle/working/assamese_10k.model")

CHECKPOINT_PATH = Path(os.environ["ABLATION_CHECKPOINT"])

RESULTS_DIR = Path(os.environ["ABLATION_EVAL_DIR"]) / "attention"

FIGURES_DIR = Path(os.environ["ABLATION_EVAL_DIR"]) / "attention_figures"

SEQUENCE_LENGTH = 128
NUM_SEQUENCES = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("assamese AGGREGATED ATTENTION ANALYSIS")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(str(TOKENIZER_PATH)):
    raise RuntimeError(
        f"Could not load tokenizer: {TOKENIZER_PATH}"
    )

print(f"Vocabulary size: {sp.get_piece_size()}")


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

config = ModelConfig(**checkpoint["config"])

model = DecoderOnlyTransformer(config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(f"Checkpoint step: {checkpoint['step']}")
print(f"Parameters: {model.count_parameters():,}")


# ============================================================
# COLLECT VALIDATION SEQUENCES
# ============================================================

validation_files = sorted(
    DATA_DIR.glob("val_*.txt")
)

if not validation_files:
    raise FileNotFoundError(
        f"No validation files found in {DATA_DIR}"
    )


sequences = []

for file_path in validation_files:

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            ids = sp.encode(
                line,
                out_type=int,
            )

            if len(ids) >= SEQUENCE_LENGTH:

                sequences.append(
                    ids[:SEQUENCE_LENGTH]
                )

                if len(sequences) >= NUM_SEQUENCES:
                    break

    if len(sequences) >= NUM_SEQUENCES:
        break


if len(sequences) < NUM_SEQUENCES:
    raise RuntimeError(
        f"Only found {len(sequences)} usable sequences."
    )


print(
    f"Validation sequences selected: "
    f"{len(sequences)}"
)


# ============================================================
# STORAGE
# ============================================================

entropy_values = torch.zeros(
    NUM_SEQUENCES,
    config.n_layers,
    config.n_heads,
)

distance_values = torch.zeros(
    NUM_SEQUENCES,
    config.n_layers,
    config.n_heads,
)


# ============================================================
# DISTANCE MATRIX
# ============================================================

positions = torch.arange(
    SEQUENCE_LENGTH,
    device=device,
    dtype=torch.float32,
)

distance_matrix = torch.abs(
    positions[:, None]
    - positions[None, :]
)


# ============================================================
# RUN ALL SEQUENCES
# ============================================================

for sequence_index, ids in enumerate(sequences):

    print(
        f"Processing sequence "
        f"{sequence_index + 1}/{NUM_SEQUENCES}..."
    )

    input_tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        logits, loss, attention_maps = model(
            input_tensor,
            return_attention=True,
        )

    for layer_index, attention in enumerate(
        attention_maps
    ):

        # [heads, T, T]
        attention = attention[0]

        for head_index in range(
            config.n_heads
        ):

            matrix = attention[head_index]

            # ------------------------------------------------
            # ENTROPY
            # ------------------------------------------------

            probabilities = matrix.clamp(
                min=1e-12
            )

            entropy_per_position = -(
                probabilities
                * torch.log(probabilities)
            ).sum(dim=-1)

            entropy = (
                entropy_per_position
                .mean()
                .item()
            )

            entropy_values[
                sequence_index,
                layer_index,
                head_index,
            ] = entropy

            # ------------------------------------------------
            # MEAN ATTENTION DISTANCE
            # ------------------------------------------------

            distance_per_position = (
                matrix * distance_matrix
            ).sum(dim=-1)

            distance = (
                distance_per_position
                .mean()
                .item()
            )

            distance_values[
                sequence_index,
                layer_index,
                head_index,
            ] = distance


# ============================================================
# AGGREGATE
# ============================================================

mean_entropy = entropy_values.mean(
    dim=0
)

std_entropy = entropy_values.std(
    dim=0
)

mean_distance = distance_values.mean(
    dim=0
)

std_distance = distance_values.std(
    dim=0
)


# ============================================================
# SAVE CSV
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

csv_path = (
    RESULTS_DIR
    / "assamese_attention_aggregate.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "layer",
        "head",
        "mean_entropy",
        "std_entropy",
        "mean_attention_distance",
        "std_attention_distance",
    ])

    for layer in range(config.n_layers):

        for head in range(config.n_heads):

            writer.writerow([
                layer + 1,
                head,
                f"{mean_entropy[layer, head].item():.6f}",
                f"{std_entropy[layer, head].item():.6f}",
                f"{mean_distance[layer, head].item():.6f}",
                f"{std_distance[layer, head].item():.6f}",
            ])


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("AGGREGATED ATTENTION STATISTICS")
print("=" * 70)

for layer in range(config.n_layers):

    print(
        f"\nLayer {layer + 1}"
    )

    for head in range(config.n_heads):

        print(
            f"  Head {head}: "
            f"Entropy = "
            f"{mean_entropy[layer, head].item():.4f} "
            f"(±{std_entropy[layer, head].item():.4f}), "
            f"Distance = "
            f"{mean_distance[layer, head].item():.4f} "
            f"(±{std_distance[layer, head].item():.4f})"
        )


# ============================================================
# ENTROPY HEATMAP
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    mean_entropy.numpy(),
    aspect="auto",
    interpolation="nearest",
)

plt.title(
    "assamese Mean Attention Entropy"
)

plt.xlabel(
    "Attention Head"
)

plt.ylabel(
    "Transformer Layer"
)

plt.xticks(
    range(config.n_heads)
)

plt.yticks(
    range(config.n_layers),
    range(1, config.n_layers + 1),
)

plt.colorbar(
    label="Mean Entropy"
)

plt.tight_layout()

entropy_figure = (
    FIGURES_DIR
    / "assamese_attention_entropy_heatmap.png"
)

plt.savefig(
    entropy_figure,
    dpi=200,
)

plt.close()


# ============================================================
# DISTANCE HEATMAP
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    mean_distance.numpy(),
    aspect="auto",
    interpolation="nearest",
)

plt.title(
    "assamese Mean Attention Distance"
)

plt.xlabel(
    "Attention Head"
)

plt.ylabel(
    "Transformer Layer"
)

plt.xticks(
    range(config.n_heads)
)

plt.yticks(
    range(config.n_layers),
    range(1, config.n_layers + 1),
)

plt.colorbar(
    label="Mean Attention Distance"
)

plt.tight_layout()

distance_figure = (
    FIGURES_DIR
    / "assamese_attention_distance_heatmap.png"
)

plt.savefig(
    distance_figure,
    dpi=200,
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("assamese AGGREGATED ATTENTION ANALYSIS COMPLETE")
print("=" * 70)

print(f"CSV: {csv_path}")
print(f"Entropy figure: {entropy_figure}")
print(f"Distance figure: {distance_figure}")
