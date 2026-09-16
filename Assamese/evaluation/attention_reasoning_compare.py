from pathlib import Path
import sys
import json
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

TOKENIZER_PATH = (
    PROJECT_DIR
    / "tokenizer_clean"
    / "assamese_10k.model"
)

PRETRAINED_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "checkpoint_030518_assamese.pt"
)

SFT_CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "phase3_assamese_sft.pt"
)

TEST_FILE = (
    PROJECT_DIR
    / "data_sft"
    / "test.jsonl"
)

FIGURES_DIR = (
    Path(__file__).resolve().parent
    / "figures"
    / "attention_reasoning"
)

SEQUENCE_LENGTH = 128

# Early and late layers
SELECTED_LAYERS = [0, 6]

# Multiple heads
SELECTED_HEADS = [0, 4]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("assamese REASONING ATTENTION COMPARISON")
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
# LOAD TEST EXAMPLE
# ============================================================

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}"
    )

test_example = None

with open(
    TEST_FILE,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        example = json.loads(line)

        # Prefer a comparative reasoning example.
        task = example.get("task", "")

        if task in [
            "direct_comparison",
            "greater_smaller",
            "two_hop",
            "three_hop",
            "ordering",
            "reversed_wording",
            "distractor",
        ]:
            test_example = example
            break

        if test_example is None:
            test_example = example


if test_example is None:
    raise RuntimeError(
        "Could not find an example in test.jsonl"
    )


# ============================================================
# GET INPUT TEXT
# ============================================================

print()
print("=" * 70)
print("SELECTED REASONING EXAMPLE")
print("=" * 70)

print(json.dumps(
    test_example,
    ensure_ascii=False,
    indent=2,
))


# ------------------------------------------------------------
# Find the input/question field without assuming one exact
# JSON schema.
# ------------------------------------------------------------

input_text = None

for key in [
    "input",
    "question",
    "prompt",
    "text",
]:

    if key in test_example:
        input_text = test_example[key]
        break


if input_text is None:
    raise KeyError(
        "Could not find input/question/prompt/text "
        "field in test example."
    )


print()
print("INPUT:")
print(input_text)


# ============================================================
# TOKENIZE
# ============================================================

input_ids = sp.encode(
    input_text,
    out_type=int,
)

if len(input_ids) > SEQUENCE_LENGTH:
    input_ids = input_ids[:SEQUENCE_LENGTH]

input_tensor = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=device,
)

actual_sequence_length = len(input_ids)

print()
print(f"Token count: {actual_sequence_length}")


# ============================================================
# MODEL CONFIG
# ============================================================

def create_config(checkpoint):

    if "config" in checkpoint:

        return ModelConfig(
            **checkpoint["config"]
        )

    # Phase 3 SFT checkpoint does not contain config.
    return ModelConfig(
        vocab_size=checkpoint["vocab_size"],
        d_model=512,
        n_layers=7,
        n_heads=8,
        d_ff=1792,
        context_length=512,
        dropout=0.1,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, name):

    print()
    print("-" * 70)
    print(f"Loading {name}")
    print("-" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = create_config(checkpoint)

    model = DecoderOnlyTransformer(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"Parameters: {model.count_parameters():,}")

    if "step" in checkpoint:
        print(f"Checkpoint step: {checkpoint['step']}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model, config


pretrained_model, config = load_model(
    PRETRAINED_CHECKPOINT,
    "PRETRAINED MODEL",
)

sft_model, _ = load_model(
    SFT_CHECKPOINT,
    "SFT MODEL",
)


# ============================================================
# RUN ATTENTION
# ============================================================

def get_attention(model):

    with torch.no_grad():

        logits, loss, attention_maps = model(
            input_tensor,
            return_attention=True,
        )

    return attention_maps


print()
print("=" * 70)
print("COLLECTING ATTENTION")
print("=" * 70)

pretrained_attention = get_attention(
    pretrained_model
)

sft_attention = get_attention(
    sft_model
)

print(
    f"Number of attention layers: "
    f"{len(pretrained_attention)}"
)

print(
    f"Attention shape per layer: "
    f"{tuple(pretrained_attention[0].shape)}"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TOKEN LABELS
# ============================================================

tokens = [
    sp.id_to_piece(token_id)
    for token_id in input_ids
]

print()
print("TOKENS:")
print(tokens)


# ============================================================
# GENERATE COMPARISON HEATMAPS
# ============================================================

for layer_index in SELECTED_LAYERS:

    pretrained_layer = (
        pretrained_attention[layer_index][0]
        .detach()
        .cpu()
    )

    sft_layer = (
        sft_attention[layer_index][0]
        .detach()
        .cpu()
    )

    layer_number = layer_index + 1

    for head_index in SELECTED_HEADS:

        pretrained_matrix = (
            pretrained_layer[head_index]
            .numpy()
        )

        sft_matrix = (
            sft_layer[head_index]
            .numpy()
        )

        # ----------------------------------------------------
        # PRETRAINED
        # ----------------------------------------------------

        pretrained_path = (
            FIGURES_DIR
            / (
                f"pretrained_layer"
                f"{layer_number}_head"
                f"{head_index}.png"
            )
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.imshow(
            pretrained_matrix,
            aspect="auto",
            interpolation="nearest",
        )

        plt.title(
            f"assamese Reasoning Attention - "
            f"PRETRAINED - Layer {layer_number}, "
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
            pretrained_path,
            dpi=200,
        )

        plt.close()

        print(
            f"Saved: {pretrained_path}"
        )

        # ----------------------------------------------------
        # SFT
        # ----------------------------------------------------

        sft_path = (
            FIGURES_DIR
            / (
                f"sft_layer"
                f"{layer_number}_head"
                f"{head_index}.png"
            )
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.imshow(
            sft_matrix,
            aspect="auto",
            interpolation="nearest",
        )

        plt.title(
            f"assamese Reasoning Attention - "
            f"SFT - Layer {layer_number}, "
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
            sft_path,
            dpi=200,
        )

        plt.close()

        print(
            f"Saved: {sft_path}"
        )


# ============================================================
# SAVE INPUT
# ============================================================

input_path = (
    FIGURES_DIR
    / "reasoning_prompt.txt"
)

with open(
    input_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(input_text)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("assamese REASONING ATTENTION COMPARISON COMPLETE")
print("=" * 70)

print(f"Prompt:  {input_path}")
print(f"Figures: {FIGURES_DIR}")