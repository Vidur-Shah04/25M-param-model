from pathlib import Path
import os
import sys
import csv

import torch
import matplotlib.pyplot as plt
import sentencepiece as spm


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = Path(
    os.environ.get(
        "ABLATION_CHECKPOINT",
        ROOT / "checkpoint_030518.pt"
    )
)

TOKENIZER_PATH = Path(
    os.environ.get(
        "ABLATION_TOKENIZER",
        ROOT / "assamese_10k.model"
    )
)

DATA_DIR = Path(
    os.environ.get(
        "ABLATION_DATA_DIR",
        ROOT.parent.parent / "Assamese" / "data" / "splits_v2" / "val"
    )
)

EVAL_DIR = Path(
    os.environ.get(
        "ABLATION_EVAL_DIR",
        ROOT / "evaluation_results"
    )
)

RESULTS_DIR = (
    EVAL_DIR / "attention"
)

FIGURES_DIR = (
    EVAL_DIR / "attention_figures"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.insert(0, str(ROOT))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 128
NUM_SEQUENCES = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)
print("ASSAMESE AGGREGATED ATTENTION ANALYSIS")
print("=" * 80)

print("Root       :", ROOT)
print("Checkpoint :", CHECKPOINT_PATH)
print("Tokenizer  :", TOKENIZER_PATH)
print("Validation :", DATA_DIR)
print("Device     :", device)

if device.type == "cuda":

    print(
        "GPU        :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TOKENIZER
# ============================================================

sp = spm.SentencePieceProcessor()

if not sp.load(
    str(TOKENIZER_PATH)
):

    raise RuntimeError(
        f"Could not load tokenizer:\n{TOKENIZER_PATH}"
    )

print(
    "Vocabulary size:",
    sp.get_piece_size()
)


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

config = ModelConfig(
    **checkpoint["config"]
)

model = DecoderOnlyTransformer(
    config
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint step:",
    checkpoint["step"]
)

print(
    "Parameters:",
    model.count_parameters()
)


# ============================================================
# FIND VALIDATION FILES
# ============================================================

def get_text_files(directory):

    if directory.is_file():
        return [directory]

    if not directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n{directory}"
        )

    files = sorted(
        directory.rglob("*.txt")
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in:\n{directory}"
        )

    return files


# ============================================================
# COLLECT VALIDATION SEQUENCES
# ============================================================

validation_files = get_text_files(
    DATA_DIR
)

sequences = []


for file_path in validation_files:

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            ids = sp.encode(
                line,
                out_type=int
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
        f"Only found {len(sequences)} usable "
        f"validation sequences."
    )


print(
    "Validation sequences selected:",
    len(sequences)
)


# ============================================================
# STORAGE
# ============================================================

entropy_values = torch.zeros(
    NUM_SEQUENCES,
    config.n_layers,
    config.n_heads
)

distance_values = torch.zeros(
    NUM_SEQUENCES,
    config.n_layers,
    config.n_heads
)


# ============================================================
# DISTANCE MATRIX
# ============================================================

positions = torch.arange(
    SEQUENCE_LENGTH,
    device=device,
    dtype=torch.float32
)

distance_matrix = torch.abs(
    positions[:, None]
    - positions[None, :]
)


# ============================================================
# RUN ATTENTION ANALYSIS
# ============================================================

for sequence_index, ids in enumerate(
    sequences
):

    print(
        f"Processing sequence "
        f"{sequence_index + 1}/"
        f"{NUM_SEQUENCES}..."
    )

    input_tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        logits, loss, attention_maps = model(
            input_tensor,
            return_attention=True
        )

    # attention_maps:
    # list of [B, heads, T, T]

    if len(attention_maps) != config.n_layers:

        raise RuntimeError(
            "Unexpected number of attention maps: "
            f"{len(attention_maps)} "
            f"(expected {config.n_layers})"
        )

    for layer_index, attention in enumerate(
        attention_maps
    ):

        # [B, H, T, T] -> [H, T, T]

        attention = attention[0]

        for head_index in range(
            config.n_heads
        ):

            matrix = attention[
                head_index
            ]

            # ------------------------------------------------
            # ENTROPY
            # ------------------------------------------------

            probabilities = matrix.clamp(
                min=1e-12
            )

            entropy_per_position = -(
                probabilities *
                torch.log(probabilities)
            ).sum(dim=-1)

            entropy = (
                entropy_per_position
                .mean()
                .item()
            )

            entropy_values[
                sequence_index,
                layer_index,
                head_index
            ] = entropy

            # ------------------------------------------------
            # MEAN ATTENTION DISTANCE
            # ------------------------------------------------

            distance_per_position = (
                matrix *
                distance_matrix
            ).sum(dim=-1)

            distance = (
                distance_per_position
                .mean()
                .item()
            )

            distance_values[
                sequence_index,
                layer_index,
                head_index
            ] = distance


# ============================================================
# AGGREGATE
# ============================================================

mean_entropy = (
    entropy_values.mean(dim=0)
)

std_entropy = (
    entropy_values.std(dim=0)
)

mean_distance = (
    distance_values.mean(dim=0)
)

std_distance = (
    distance_values.std(dim=0)
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    RESULTS_DIR /
    "assamese_attention_aggregate.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "layer",
        "head",
        "mean_entropy",
        "std_entropy",
        "mean_attention_distance",
        "std_attention_distance"
    ])

    for layer in range(
        config.n_layers
    ):

        for head in range(
            config.n_heads
        ):

            writer.writerow([

                layer + 1,

                head,

                f"{mean_entropy[layer, head].item():.6f}",

                f"{std_entropy[layer, head].item():.6f}",

                f"{mean_distance[layer, head].item():.6f}",

                f"{std_distance[layer, head].item():.6f}"
            ])


# ============================================================
# PRINT STATISTICS
# ============================================================

print()
print("=" * 80)
print("AGGREGATED ATTENTION STATISTICS")
print("=" * 80)


for layer in range(
    config.n_layers
):

    print(
        f"\nLayer {layer + 1}"
    )

    for head in range(
        config.n_heads
    ):

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
    mean_entropy.cpu().numpy(),
    aspect="auto",
    interpolation="nearest"
)

plt.title(
    "Assamese Mean Attention Entropy"
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
    range(1, config.n_layers + 1)
)

plt.colorbar(
    label="Mean Entropy"
)

plt.tight_layout()

entropy_figure = (
    FIGURES_DIR /
    "assamese_attention_entropy_heatmap.png"
)

plt.savefig(
    entropy_figure,
    dpi=200
)

plt.close()


# ============================================================
# DISTANCE HEATMAP
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    mean_distance.cpu().numpy(),
    aspect="auto",
    interpolation="nearest"
)

plt.title(
    "Assamese Mean Attention Distance"
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
    range(1, config.n_layers + 1)
)

plt.colorbar(
    label="Mean Attention Distance"
)

plt.tight_layout()

distance_figure = (
    FIGURES_DIR /
    "assamese_attention_distance_heatmap.png"
)

plt.savefig(
    distance_figure,
    dpi=200
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("ATTENTION ANALYSIS COMPLETE")
print("=" * 80)

print(
    "CSV:",
    csv_path
)

print(
    "Entropy figure:",
    entropy_figure
)

print(
    "Distance figure:",
    distance_figure
)