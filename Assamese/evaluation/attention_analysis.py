from pathlib import Path
import sys
import csv
import math

import torch
import matplotlib.pyplot as plt
import sentencepiece as spm

# ============================================================
# MAKE assamese/model IMPORTABLE
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.transformer import DecoderOnlyTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_DIR / "data" / "splits_v2"

PROJECT_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = (
    PROJECT_DIR
    / "tokenizer_clean"
    / "assamese_24k.model"
)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "checkpoint_030518_assamese.pt"
)

FIGURES_DIR = (
    Path(__file__).resolve().parent
    / "figures"
    / "attention"
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

SEQUENCE_LENGTH = 128

# Layers are zero-indexed internally:
# 0 = Layer 1
# 2 = Layer 3
# 4 = Layer 5
SELECTED_LAYERS = [0, 2, 4]

# Heads 0 and 4 provide two different heads per selected layer.
SELECTED_HEADS = [0, 4]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("assamese ATTENTION ANALYSIS")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(str(TOKENIZER_PATH)):
    raise RuntimeError(
        f"Could not load tokenizer: {TOKENIZER_PATH}"
    )

print(f"Vocabulary size: {sp.get_piece_size()}")


# ============================================================
# LOAD CHECKPOINT
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
# GET A HELD-OUT VALIDATION SEQUENCE
# ============================================================

validation_files = sorted(
    DATA_DIR.glob("val_*.txt")
)

if not validation_files:
    raise FileNotFoundError(
        f"No validation files found in {DATA_DIR}"
    )

input_ids = None

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
                input_ids = ids[:SEQUENCE_LENGTH]
                break

    if input_ids is not None:
        break


if input_ids is None:
    raise RuntimeError(
        "Could not find a validation sequence long enough."
    )


input_tensor = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=device,
)


print(f"Analysis sequence length: {SEQUENCE_LENGTH}")


# ============================================================
# RUN MODEL AND COLLECT ATTENTION
# ============================================================

with torch.no_grad():

    logits, loss, attention_maps = model(
        input_tensor,
        return_attention=True,
    )


print(
    f"Number of attention layers: "
    f"{len(attention_maps)}"
)

print(
    f"Attention shape per layer: "
    f"{tuple(attention_maps[0].shape)}"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ATTENTION STATISTICS
# ============================================================

statistics = []


for layer_index, attention in enumerate(
    attention_maps
):

    # attention shape:
    # [B, heads, T, T]
    attention = attention[0]

    for head_index in range(
        config.n_heads
    ):

        matrix = attention[head_index]

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------
        #
        # H_i = -sum_j p_ij log(p_ij)
        #
        # Calculate entropy for each query position,
        # then average across positions.
        #
        probabilities = matrix.clamp(
            min=1e-12
        )

        entropy_per_position = -(
            probabilities
            * torch.log(probabilities)
        ).sum(dim=-1)

        mean_entropy = (
            entropy_per_position
            .mean()
            .item()
        )

        # ----------------------------------------------------
        # Mean attention distance
        # ----------------------------------------------------
        #
        # D_i = sum_j A_ij |i-j|
        #
        # Then average over query positions.
        #
        positions = torch.arange(
            SEQUENCE_LENGTH,
            device=device,
            dtype=torch.float32,
        )

        distance_matrix = torch.abs(
            positions[:, None]
            - positions[None, :]
        )

        mean_distance_per_position = (
            matrix * distance_matrix
        ).sum(dim=-1)

        mean_attention_distance = (
            mean_distance_per_position
            .mean()
            .item()
        )

        statistics.append({
            "layer": layer_index + 1,
            "head": head_index,
            "mean_entropy": mean_entropy,
            "mean_attention_distance":
                mean_attention_distance,
        })


# ============================================================
# SAVE STATISTICS CSV
# ============================================================

csv_path = (
    RESULTS_DIR
    / "assamese_attention_statistics.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "layer",
            "head",
            "mean_entropy",
            "mean_attention_distance",
        ],
    )

    writer.writeheader()

    writer.writerows(statistics)


# ============================================================
# PRINT STATISTICS
# ============================================================

print()
print("=" * 70)
print("ATTENTION STATISTICS")
print("=" * 70)

for row in statistics:

    print(
        f"Layer {row['layer']} "
        f"Head {row['head']}: "
        f"Entropy = "
        f"{row['mean_entropy']:.4f}, "
        f"Distance = "
        f"{row['mean_attention_distance']:.4f}"
    )


# ============================================================
# GENERATE HEATMAPS
# ============================================================

tokens = [
    sp.id_to_piece(token_id)
    for token_id in input_ids
]


for layer_index in SELECTED_LAYERS:

    attention = (
        attention_maps[layer_index][0]
        .detach()
        .cpu()
    )

    for head_index in SELECTED_HEADS:

        matrix = attention[head_index].numpy()

        layer_number = layer_index + 1

        figure_path = (
            FIGURES_DIR
            / (
                f"assamese_layer"
                f"{layer_number}_head"
                f"{head_index}.png"
            )
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
        )

        plt.title(
            f"assamese Attention - "
            f"Layer {layer_number}, "
            f"Head {head_index}"
        )

        plt.xlabel(
            "Key / Attended Position"
        )

        plt.ylabel(
            "Query Position"
        )

        plt.colorbar(
            label="Attention Probability"
        )

        plt.tight_layout()

        plt.savefig(
            figure_path,
            dpi=200,
        )

        plt.close()

        print(
            f"Saved heatmap: {figure_path}"
        )


# ============================================================
# SAVE THE ANALYSIS TOKENS
# ============================================================

tokens_path = (
    RESULTS_DIR
    / "assamese_attention_sequence.txt"
)

with open(
    tokens_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        sp.decode(input_ids)
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("assamese ATTENTION ANALYSIS COMPLETE")
print("=" * 70)

print(f"Statistics: {csv_path}")
print(f"Sequence:   {tokens_path}")
print(f"Figures:    {FIGURES_DIR}")