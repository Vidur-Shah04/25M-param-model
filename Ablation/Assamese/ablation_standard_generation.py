from pathlib import Path
import os
import sys
import random

import torch
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

OUTPUT_DIR = EVAL_DIR / "generations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORT LOCAL MODEL
# ============================================================

sys.path.insert(0, str(ROOT))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


# ============================================================
# SETTINGS
# ============================================================

CONTEXT_LENGTH = 256
MAX_NEW_TOKENS = 100

TEMPERATURES = [
    0.0,
    0.5,
    1.0,
    1.5,
]

NUM_PROMPTS = 10
SEED = 42


# ============================================================
# FIND VALIDATION TEXT FILES
# ============================================================

def get_text_files(directory):
    if directory.is_file():
        return [directory]

    if not directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n{directory}"
        )

    files = sorted(directory.rglob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt validation files found in:\n{directory}"
        )

    return files


# ============================================================
# LOAD PROMPTS
# ============================================================

def load_prompts():

    prompts = []

    validation_files = get_text_files(DATA_DIR)

    for file_path in validation_files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if len(line) < 100:
                    continue

                prompts.append(line)

                if len(prompts) >= NUM_PROMPTS:
                    return prompts

    if len(prompts) < NUM_PROMPTS:
        raise RuntimeError(
            f"Only found {len(prompts)} suitable validation prompts."
        )

    return prompts


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    temperature,
    max_new_tokens
):

    encoded = tokenizer.encode(
        prompt,
        out_type=int
    )

    encoded = encoded[-CONTEXT_LENGTH:]

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    generated = input_ids.clone()

    for _ in range(max_new_tokens):

        context = generated[:, -CONTEXT_LENGTH:]

        logits, _ = model(context)

        next_token_logits = logits[:, -1, :]

        if temperature <= 0:

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

        else:

            scaled_logits = (
                next_token_logits / temperature
            )

            probabilities = torch.softmax(
                scaled_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    return tokenizer.decode(
        generated[0].tolist()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ASSAMESE STANDARD GENERATION")
    print("=" * 80)

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Root       :", ROOT)
    print("Checkpoint :", CHECKPOINT_PATH)
    print("Tokenizer  :", TOKENIZER_PATH)
    print("Validation :", DATA_DIR)
    print("Output     :", OUTPUT_DIR)
    print("Device     :", device)

    if device.type == "cuda":
        print(
            "GPU        :",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.load(
        str(TOKENIZER_PATH)
    ):
        raise RuntimeError(
            f"Could not load tokenizer:\n{TOKENIZER_PATH}"
        )

    print(
        "Vocabulary :",
        tokenizer.get_piece_size()
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PROMPTS
    # --------------------------------------------------------

    prompts = load_prompts()

    print(
        f"Using {len(prompts)} validation prompts."
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    for temperature in TEMPERATURES:

        print(
            f"\nGenerating T={temperature}..."
        )

        output_path = (
            OUTPUT_DIR /
            f"assamese_temp_{temperature:.1f}.txt"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as out:

            for i, prompt in enumerate(prompts):

                generated_text = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    temperature=temperature,
                    max_new_tokens=MAX_NEW_TOKENS
                )

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
            "Saved:",
            output_path
        )

    print()
    print("=" * 80)
    print("STANDARD GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()