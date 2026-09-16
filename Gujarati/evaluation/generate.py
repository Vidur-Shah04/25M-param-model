from pathlib import Path
import sys
import math
import random

import sentencepiece as spm
import torch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "checkpoint_030518_gujarati.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer_clean"
    / "gujarati_10k.model"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits_v2"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "generations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

CONTEXT_LENGTH = 256
MAX_NEW_TOKENS = 100

# TEMPERATURES = [
#     0.5,
#     1.0,
#     1.5,
# ]
TEMPERATURES = [
    0.0,   # greedy
    0.5,
    1.0,
    1.5,
]

NUM_PROMPTS = 10

SEED = 42


# ---------------------------------------------------------
# Load validation prompts
# ---------------------------------------------------------

def load_prompts():

    prompts = []

    validation_files = sorted(
        VAL_DIR.glob("val_*.txt")
    )

    if not validation_files:
        raise FileNotFoundError(
            f"No validation files found in {VAL_DIR}"
        )

    for file_path in validation_files:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                # We want reasonably long natural-language
                # examples rather than tiny fragments.
                if len(line) < 100:
                    continue

                prompts.append(line)

                if len(prompts) >= NUM_PROMPTS:
                    return prompts

    if len(prompts) < NUM_PROMPTS:
        raise RuntimeError(
            f"Only found {len(prompts)} suitable prompts."
        )

    return prompts


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    temperature,
    max_new_tokens,
):

    encoded = tokenizer.encode(
        prompt,
        out_type=int,
    )

    # Keep only the last CONTEXT_LENGTH tokens.
    encoded = encoded[-CONTEXT_LENGTH:]

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device,
    )

    generated = input_ids.clone()

    for _ in range(max_new_tokens):

        # Transformer only accepts at most its context length.
        context = generated[:, -CONTEXT_LENGTH:]

        logits, _ = model(
            context
        )

        # Only predict the next token from the
        # final position.
        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

        else:
            scaled_logits = (
                next_token_logits / temperature
            )

            probabilities = torch.softmax(
                scaled_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    generated_ids = generated[0].tolist()

    return tokenizer.decode(
        generated_ids
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("GUJARATI GENERATION EVALUATION")
    print("=" * 70)

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.load(
        str(TOKENIZER_PATH)
    ):
        raise RuntimeError(
            f"Could not load tokenizer: {TOKENIZER_PATH}"
        )

    print(
        f"Tokenizer vocabulary: "
        f"{tokenizer.get_piece_size()}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    # config = ModelConfig()
    config = ModelConfig(**checkpoint["config"])
    model = DecoderOnlyTransformer(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint step: "
        f"{checkpoint['step']}"
    )

    # -----------------------------------------------------
    # Prompts
    # -----------------------------------------------------

    prompts = load_prompts()

    print(
        f"Using {len(prompts)} validation prompts."
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    for temperature in TEMPERATURES:

        print(
            f"\nGenerating at temperature "
            f"{temperature}..."
        )

        output_path = (
            OUTPUT_DIR
            / f"gujarati_temp_{temperature:.1f}.txt"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as out:

            for i, prompt in enumerate(prompts):

                generated_text = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    temperature=temperature,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                # Store the original prompt separately.
                out.write(
                    f"========== SAMPLE {i + 1} ==========\n"
                )

                out.write(
                    "PROMPT:\n"
                )

                out.write(
                    prompt + "\n\n"
                )

                out.write(
                    "GENERATED:\n"
                )

                out.write(
                    generated_text + "\n\n"
                )

        print(
            f"Saved: {output_path}"
        )

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()